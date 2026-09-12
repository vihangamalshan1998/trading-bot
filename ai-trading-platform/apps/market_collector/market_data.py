import asyncio
import websockets
import json
import time
import numpy as np
from typing import List, Dict
from core.logging.logger import logger
from core.db.redis import redis_manager
from core.config.settings import settings
from apps.feature_engine.engine import FeatureEngine
from core.data_quality.validator import DataQualityValidator

class BinanceWebSocketCollector:
    """
    Phase 3: Connects to Binance Futures WebSocket, ingests multiple streams (Orderbook, Trades, Funding, Liq),
    and feeds them through the FeatureEngine before streaming dense feature vectors to Redis.
    """
    def __init__(self, symbols: List[str]):
        self.symbols = [s.lower() for s in symbols]
        self.base_url = "wss://fstream.binance.com/ws"
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
                "bid_qty": 0.0, "ask_qty": 0.0, "funding_rate": 0.0, "liquidation_volume": 0.0
            } for s in symbols
        }
        
    async def publish_features(self, symbol: str):
        """Pulls features from the engine and publishes to Redis"""
        try:
            engine = self.feature_engines[symbol]
            engine.add_tick(self.latest_raw[symbol])
            
            # The returned array is the 25-dim Phase 3 Market State
            market_state = engine.compute_features()
            
            channel = f"market:state:{symbol}"
            payload = market_state.tolist()
            
            if redis_manager.redis is not None:
                await redis_manager.redis.publish(channel, json.dumps(payload))
                
                # Also publish a human-readable state for the Dashboard UI
                trend = "UP" if self.latest_raw[symbol]["buy_volume"] > (self.latest_raw[symbol]["volume"] / 2) else "DOWN"
                dashboard_payload = {
                    "price": self.latest_raw[symbol]["mid_price"],
                    "trend": trend
                }
                await redis_manager.redis.set(f"dashboard:market_states:{symbol}", json.dumps(dashboard_payload))
                
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

            self.latest_raw[symbol]["mid_price"] = price # trades also update mid_price
            self.latest_raw[symbol]["volume"] = qty
            self.latest_raw[symbol]["buy_volume"] = 0.0 if is_buyer_maker else qty
            self.latest_raw[symbol]["trade_count"] = 1.0
            
            await self.publish_features(symbol)
        except Exception as e:
            pass

    async def process_funding_rate(self, data: dict):
        try:
            symbol = data['s'].upper()
            self.latest_raw[symbol]["funding_rate"] = float(data['r'])
            await self.publish_features(symbol)
        except Exception as e:
            pass
            
    async def process_liquidation(self, data: dict):
        try:
            order = data['o']
            symbol = order['s'].upper()
            qty = float(order['q'])
            self.latest_raw[symbol]["liquidation_volume"] = qty
            await self.publish_features(symbol)
        except Exception as e:
            pass

    async def listen(self):
        self._running = True
        
        # Build streams: <symbol>@bookTicker, <symbol>@aggTrade, <symbol>@markPrice, <symbol>@forceOrder
        streams = []
        for s in self.symbols:
            streams.extend([f"{s}@bookTicker", f"{s}@aggTrade", f"{s}@markPrice", f"{s}@forceOrder"])
            
        streams_path = "/".join(streams)
        url = f"{self.base_url}/{streams_path}"
        
        logger.info(f"Connecting to Binance WS: {url}")
        
        await redis_manager.connect()
        
        while self._running:
            try:
                async with websockets.connect(url) as websocket:
                    logger.info("Connected to Binance WebSocket!")
                    while self._running:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        event_type = data.get('e')
                        if 'b' in data and 'a' in data and 'e' not in data:
                            await self.process_book_ticker(data)
                        elif event_type == 'aggTrade':
                            await self.process_agg_trade(data)
                        elif event_type == 'markPriceUpdate':
                            await self.process_funding_rate(data)
                        elif event_type == 'forceOrder':
                            await self.process_liquidation(data)
                            
            except websockets.ConnectionClosed:
                logger.warning("WebSocket Connection Closed. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket Error: {e}. Reconnecting...")
                await asyncio.sleep(5)
                
    def stop(self):
        self._running = False

async def run_collector():
    symbols = settings.symbol_universe
    collector = BinanceWebSocketCollector(symbols)
    
    try:
        await collector.listen()
    except asyncio.CancelledError:
        collector.stop()

if __name__ == "__main__":
    try:
        asyncio.run(run_collector())
    except KeyboardInterrupt:
        pass
