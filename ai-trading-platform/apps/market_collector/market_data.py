import asyncio
import websockets
import json
import time
import numpy as np
from typing import List, Dict
from core.logging.logger import logger, set_log_file
from core.db.redis import redis_manager
from core.config.settings import settings
from apps.feature_engine.engine import FeatureEngine
from core.data_quality.validator import DataQualityValidator
from apps.market_collector.context_poller import market_data_is_mainnet, poll_context, seed_engines
from core.exchange.binance_streams import combined_stream_urls
from apps.market_collector.global_context import poll_global_context

class BinanceWebSocketCollector:
    """
    Phase 3: Connects to Binance Futures WebSocket, ingests multiple streams (Orderbook, Trades, Funding, Liq),
    and feeds them through the FeatureEngine before streaming dense feature vectors to Redis.
    """
    def __init__(self, symbols: List[str]):
        self.symbols = [s.lower() for s in symbols]
        self.mainnet = market_data_is_mainnet()
        self._running = False
        
        # Initialize a Feature Engine for each symbol
        self.feature_engines: Dict[str, FeatureEngine] = {
            s.upper(): FeatureEngine(s.upper()) for s in symbols
        }
        
        # Maintain raw latest state to feed into the engine
        self.latest_raw: Dict[str, Dict] = {
            s.upper(): {
                "mid_price": 0.0, "volume": 0.0, "buy_volume": 0.0, 
                "trade_count": 0.0, "best_bid": 0.0, "best_ask": 0.0,
                "bid_qty": 0.0, "ask_qty": 0.0, "funding_rate": 0.0, "liquidation_volume": 0.0,
                "open_interest": 0.0, "ls_ratio": 1.0, "ema_4h": 0.0, "ob_skew_l2": 0.0,
                "price_change_24h": 0.0, "volume_24h": 0.0, "mark_price_premium": 0.0
            } for s in symbols
        }
        
        self.last_publish_time = {s.upper(): 0.0 for s in symbols}
        
    async def publish_features(self, symbol: str):
        """Pulls features from the engine and publishes to Redis"""
        try:
            now = time.time()
            if now - self.last_publish_time[symbol] < 1.0:
                return
            self.last_publish_time[symbol] = now
            
            engine = self.feature_engines[symbol]
            raw = self.latest_raw[symbol]
            engine.add_tick(raw)
            # Trade / liquidation totals are per-tick accumulators: reset after each tick
            raw["volume"] = raw["buy_volume"] = raw["trade_count"] = raw["liquidation_volume"] = 0.0
            
            # The returned array is the 90-dim Phase 3 Market State
            ai_features_array = engine.compute_features().tolist()

            # raw["mid_price"] is whichever came last: a trade price (at the bid or the ask) or the book mid, so it
            # jumps by half the spread. The engine keeps that series (the brain was trained on it), but the bot
            # prices stops, PnL and orders off mid_price, so publish the true book mid there.
            bid, ask = raw["best_bid"], raw["best_ask"]
            book_mid = (bid + ask) / 2.0 if bid > 0 and ask >= bid else raw["mid_price"]

            payload = {
                "symbol": symbol,
                "timestamp": time.time(),
                "bid": raw["best_bid"] or raw["mid_price"],
                "ask": raw["best_ask"] or raw["mid_price"],
                "mid_price": book_mid,
                "last_price": raw["mid_price"],
                "spread": max(0.0, raw["best_ask"] - raw["best_bid"]),
                "order_book_imbalance": raw["ob_skew_l2"],
                "volume": float(sum(list(engine.volumes)[-60:])),
                "vwap": raw["mid_price"], 
                # 1h ATR as a fraction of price: drives the volatility-scaled exit policy
                "volatility": engine.volatility_1h(),
                "funding_rate": raw["funding_rate"],
                "features": ai_features_array,
                "data_quality": 1.0
            }
            
            channel = f"market:state:{symbol}"
            
            if redis_manager.redis is not None:
                # set_state handles both setting the key (for dashboard) and publishing (for trading bot)
                await redis_manager.set_state(channel, payload, ttl_seconds=60)
                
        except Exception as e:
            logger.error(f"Error publishing features for {symbol}: {e}")

    async def process_book_ticker(self, data: dict):
        try:
            symbol = data['s'].upper()
            best_bid = float(data['b'])
            best_ask = float(data['a'])
            
            
            # Phase 4: Data Quality Check
            raw_tick = {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_qty": float(data['B']),
                "ask_qty": float(data['A'])
            }
            # bookTicker event doesn't always have a timestamp, use current time
            is_valid, reason = DataQualityValidator.validate_tick(raw_tick, int(time.time() * 1000))
            if not is_valid:
                logger.warning(f"Data Quality Validation Failed: {symbol} - {reason}")
                return
                
            self.latest_raw[symbol]["best_bid"] = best_bid
            self.latest_raw[symbol]["best_ask"] = best_ask
            self.latest_raw[symbol]["mid_price"] = (best_bid + best_ask) / 2.0
            self.latest_raw[symbol]["bid_qty"] = raw_tick["bid_qty"]
            self.latest_raw[symbol]["ask_qty"] = raw_tick["ask_qty"]
            
            await self.publish_features(symbol)
        except Exception as e:
            pass

    async def process_agg_trade(self, data: dict):
        try:
            symbol = data['s'].upper()
            price = float(data['p'])
            qty = float(data['q'])
            is_buyer_maker = data['m'] # True if buyer is maker (meaning trade was a sell)
            
            
            # Phase 4: Data Quality Check
            raw_tick = {
                "mid_price": price,
                "volume": qty
            }
            event_time = data.get('E', int(time.time() * 1000)) # Event time
            is_valid, reason = DataQualityValidator.validate_tick(raw_tick, event_time)
            
            if not is_valid:
                logger.warning(f"Data Quality Validation Failed: {symbol} - {reason}")
                return

            # Accumulate every trade until the next tick (previously each trade OVERWROTE the last one,
            # so "1m volume" was a sum of 60 random single-trade sizes)
            raw = self.latest_raw[symbol]
            raw["mid_price"] = price # trades also update mid_price
            raw["volume"] += qty
            if not is_buyer_maker:
                raw["buy_volume"] += qty
            raw["trade_count"] += 1.0
            
            await self.publish_features(symbol)
        except Exception as e:
            pass

    async def process_funding_rate(self, data: dict):
        try:
            symbol = data['s'].upper()
            if 'r' in data:
                self.latest_raw[symbol]["funding_rate"] = float(data['r'])
            
            mark = float(data.get('p', 1.0))
            index = float(data.get('i', 1.0))
            if index > 0:
                self.latest_raw[symbol]["mark_price_premium"] = (mark - index) / index
            if data.get('T'):
                self.latest_raw[symbol]["next_funding_ms"] = float(data['T'])
                
            await self.publish_features(symbol)
        except Exception as e:
            pass
            
    async def process_ticker24h(self, data: dict):
        try:
            symbol = data['s'].upper()
            self.latest_raw[symbol]["price_change_24h"] = float(data.get('P', 0.0)) / 100.0  # P is percentage
            self.latest_raw[symbol]["volume_24h"] = float(data.get('q', 0.0)) # Quote volume in USDT
            await self.publish_features(symbol)
        except Exception as e:
            pass
            
    async def process_liquidation(self, data: dict):
        try:
            order = data['o']
            symbol = order['s'].upper()
            qty = float(order['q'])
            self.latest_raw[symbol]["liquidation_volume"] += qty
            await self.publish_features(symbol)
        except Exception as e:
            pass

    async def process_open_interest(self, data: dict):
        try:
            symbol = data['s'].upper()
            self.latest_raw[symbol]["open_interest"] = float(data['o'])
            await self.publish_features(symbol)
        except Exception as e:
            pass
            
    async def process_depth(self, data: dict, symbol: str):
        try:
            if 'b' in data and 'a' in data and isinstance(data['b'], list):
                bids = sum(float(b[1]) for b in data['b'])
                asks = sum(float(a[1]) for a in data['a'])
                tot = bids + asks
                if tot > 0:
                    self.latest_raw[symbol]["ob_skew_l2"] = (bids - asks) / tot
                await self.publish_features(symbol)
        except Exception as e:
            pass
            
    async def poll_rest_data(self):
        """Open interest, premium, funding, 24h stats, long/short ratio and 4h trend (shared poller)."""
        await poll_context([sym.upper() for sym in self.symbols], self.latest_raw, lambda: self._running)

    async def listen(self):
        self._running = True
        
        # Build streams: <symbol>@bookTicker, <symbol>@aggTrade, <symbol>@markPrice, <symbol>@forceOrder, <symbol>@depth5@100ms, <symbol>@ticker
        streams = []
        for s in self.symbols:
            streams.extend([f"{s}@bookTicker", f"{s}@aggTrade", f"{s}@markPrice", f"{s}@forceOrder", f"{s}@depth5@100ms", f"{s}@ticker"])
            
        logger.info(f"Connecting to {len(streams)} Binance WS streams...")
        await redis_manager.connect()
        # Pre-load ~16h of 1m bars so indicators / volatility are valid immediately
        await seed_engines(self.feature_engines)
        
        async def _listen_chunk(url):
            chunk = url.split("streams=", 1)[1].split("/")
            while self._running:
                try:
                    # Binance does not accept unsolicited PING frames from the client.
                    # We must set ping_interval=None so websockets only responds with PONGs to Binance's PINGs.
                    async with websockets.connect(url, ping_interval=None) as websocket:
                        logger.info(f"Connected to Binance WebSocket Chunk ({len(chunk)} streams)!")
                        while self._running:
                            # A half-open connection never closes: without a read timeout this chunk's coins
                            # silently stopped updating (stale prices for the bot's stops). Every chunk carries
                            # trades and book updates many times a second, so 60s of silence means it is dead.
                            try:
                                message = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                            except asyncio.TimeoutError:
                                logger.warning(f"WebSocket Chunk silent for 60s ({len(chunk)} streams). Reconnecting...")
                                break
                            data = json.loads(message)
                            
                            event_type = data.get('e')
                            if 'b' in data and 'a' in data and 'e' not in data:
                                if not isinstance(data.get('b'), list):
                                    await self.process_book_ticker(data)
                            if event_type == 'aggTrade':
                                await self.process_agg_trade(data)
                            elif event_type == 'markPriceUpdate':
                                await self.process_funding_rate(data)
                            elif event_type == 'forceOrder':
                                await self.process_liquidation(data)
                            elif event_type == 'openInterest':
                                await self.process_open_interest(data)
                            elif event_type == '24hrTicker':
                                await self.process_ticker24h(data)
                            elif 'stream' in data and '@depth5' in data['stream']:
                                sym = data['stream'].split('@')[0].upper()
                                await self.process_depth(data['data'], sym)
                            elif 'stream' in data and '@ticker' in data['stream']:
                                await self.process_ticker24h(data['data'])
                            elif 'stream' in data and '@bookTicker' in data['stream']:
                                await self.process_book_ticker(data['data'])
                            elif 'stream' in data and '@aggTrade' in data['stream']:
                                await self.process_agg_trade(data['data'])
                            elif 'stream' in data and '@markPrice' in data['stream']:
                                await self.process_funding_rate(data['data'])
                            elif 'stream' in data and '@forceOrder' in data['stream']:
                                await self.process_liquidation(data['data'])
                            elif 'stream' in data and '@openInterest' in data['stream']:
                                await self.process_open_interest(data['data'])
                            else:
                                if 'b' in data and 'a' in data and 'e' not in data and not isinstance(data.get('b'), list):
                                    await self.process_book_ticker(data)
                except websockets.ConnectionClosed as e:
                    logger.warning(f"WebSocket Chunk Closed ({e.code} - {e.reason}). Reconnecting in 5s...")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"WebSocket Chunk Error: {e}. Reconnecting...")
                    await asyncio.sleep(5)

        # Mainnet routes trades / mark price / ticker / liquidations and depth / bookTicker to different
        # endpoints (core/exchange/binance_streams.py); one connection per endpoint and chunk.
        tasks = []
        for url in combined_stream_urls(streams, self.mainnet):
            tasks.append(asyncio.create_task(_listen_chunk(url)))
            
        await asyncio.gather(*tasks)
                
    def stop(self):
        self._running = False

async def run_collector():
    from apps.market_collector.universe_screener import UniverseScreener
    symbols = settings.symbol_universe
    collector = BinanceWebSocketCollector(symbols)
    screener = UniverseScreener(target_symbol_count=10, poll_interval_seconds=1800)
    
    try:
        # Run the WebSocket listener, REST poller, and dynamic Universe Screener concurrently
        await redis_manager.connect()
        await asyncio.gather(
            collector.listen(),
            collector.poll_rest_data(),
            poll_global_context(redis_manager.redis),
            screener.start()
        )
    except asyncio.CancelledError:
        collector.stop()
        screener.stop()

if __name__ == "__main__":
    set_log_file("logs/market_collector.log")
    try:
        asyncio.run(run_collector())
    except KeyboardInterrupt:
        pass
