import asyncio
from typing import Dict, Any, List
import time

from core.logging.logger import logger
from core.config.settings import settings
from core.exchange.binance_client import BinanceFuturesAdapter
from apps.orderbook.builder import OrderBookBuilder
from apps.feature_engine.calculators import FeatureEngine
from apps.feature_engine.engine import FeatureEngine as AIEngine
import json

import aiohttp

from apps.market_collector.context_poller import market_data_is_mainnet, market_data_rest_url, poll_context, seed_engines
from core.exchange.binance_streams import combined_stream_urls
from core.db.redis import redis_manager
from core.db.repository import MarketFeatureRepository
from core.exchange.symbol_registry import registry, SymbolConfig


def _empty_raw() -> Dict[str, float]:
    return {
        # accumulated since the previous AI tick (reset after every tick)
        "volume": 0.0, "buy_volume": 0.0, "trade_count": 0.0, "liquidation_volume": 0.0,
        # latest values from streams / REST context poller
        "funding_rate": 0.0, "open_interest": 0.0, "ls_ratio": 1.0, "ema_4h": 0.0, "ob_skew_l2": 0.0,
        "price_change_24h": 0.0, "volume_24h": 0.0, "mark_price_premium": 0.0,
    }


class MarketCollectorService:
    def __init__(self):
        self.adapter = BinanceFuturesAdapter()
        # Public market data from the configured market-data venue (see context_poller)
        self.adapter.base_url = market_data_rest_url()

        # Initialize registry with target universe
        for sym in settings.symbol_universe:
            registry.add_symbol(SymbolConfig(symbol=sym))
            registry.update_status(sym, "ENABLED")

        self.active_symbols = [s.symbol for s in registry.get_active_symbols()]

        # We want order book depth, aggregate trades, mark price/funding, and liquidations
        self.streams = ["depth@100ms", "aggTrade", "markPrice@1s", "forceOrder"]
        self.running = False
        self._task = None
        self._context_task = None

        # State per symbol
        self.order_books: Dict[str, OrderBookBuilder] = {sym: OrderBookBuilder(sym) for sym in self.active_symbols}
        self.feature_engines: Dict[str, FeatureEngine] = {sym: FeatureEngine() for sym in self.active_symbols}
        self.ai_engines: Dict[str, AIEngine] = {sym: AIEngine(sym) for sym in self.active_symbols}
        self._update_counters: Dict[str, int] = {sym: 0 for sym in self.active_symbols}
        self.raw: Dict[str, Dict[str, float]] = {sym: _empty_raw() for sym in self.active_symbols}

        self.repository = MarketFeatureRepository(batch_size=50)

    def _l2_skew(self, symbol: str, levels: int = 5) -> float:
        book = self.order_books[symbol]
        try:
            bids = sorted(book.bids.items(), key=lambda kv: kv[0], reverse=True)[:levels]
            asks = sorted(book.asks.items(), key=lambda kv: kv[0])[:levels]
            b = float(sum(q for _, q in bids))
            a = float(sum(q for _, q in asks))
            return (b - a) / (b + a) if (b + a) > 0 else 0.0
        except Exception:
            return 0.0

    async def _handle_market_data(self, data: Dict[str, Any]):
        """
        Callback for incoming multiplexed market data from the WebSocket.
        """
        payload = data.get("data", data)
        stream_name = data.get("stream", "")

        # Determine symbol from stream name (e.g. btcusdt@depth) or payload
        symbol = payload.get("s", "UNKNOWN").upper()
        if symbol == "UNKNOWN" and "@" in stream_name:
             symbol = stream_name.split("@")[0].upper()
        if symbol == "UNKNOWN" and isinstance(payload.get("o"), dict):
            symbol = payload["o"].get("s", "UNKNOWN").upper()

        if symbol not in self.active_symbols:
            return

        event_type = payload.get("e", "unknown")
        raw = self.raw[symbol]

        if event_type == "depthUpdate":
            self.order_books[symbol].process_update(payload)
            self._update_counters[symbol] += 1

            # Periodically extract features and push to Redis and MySQL
            if self._update_counters[symbol] % 10 == 0:
                # Dashboard UI features (calculators.py)
                features = self.feature_engines[symbol].extract_features(self.order_books[symbol])
                features["best_bid"] = self.order_books[symbol].get_best_bid()
                features["best_ask"] = self.order_books[symbol].get_best_ask()

                # AI Model features (engine.py). Trade / liquidation totals are what accumulated
                # since the previous tick (previously hard-coded to 0, leaving every volume feature dead).
                tick_data = {
                    "mid_price": features["mid_price"],
                    "best_bid": features["best_bid"],
                    "best_ask": features["best_ask"],
                    "bid_qty": float(self.order_books[symbol].bids.get(features["best_bid"], 0)),
                    "ask_qty": float(self.order_books[symbol].asks.get(features["best_ask"], 0)),
                    "ob_skew_l2": self._l2_skew(symbol),
                    **{k: v for k, v in raw.items() if k != "ob_skew_l2"},
                }
                raw["volume"] = raw["buy_volume"] = raw["trade_count"] = raw["liquidation_volume"] = 0.0
                engine = self.ai_engines[symbol]
                engine.add_tick(tick_data)
                ai_features_array = engine.compute_features().tolist()

                # Fill missing Pydantic fields for MarketState
                features["symbol"] = symbol
                features["timestamp"] = time.time()
                features["bid"] = features["best_bid"] or features["mid_price"]
                features["ask"] = features["best_ask"] or features["mid_price"]
                features["last_price"] = features["mid_price"]
                features["spread"] = max(0.0, features["ask"] - features["bid"])
                features["order_book_imbalance"] = features["imbalance"]
                features["volume"] = float(sum(list(engine.volumes)[-60:]))
                features["vwap"] = features["vwap_recent"] or features["mid_price"]
                # 1h ATR as a fraction of price: drives the volatility-scaled exit policy
                features["volatility"] = engine.volatility_1h()
                features["funding_rate"] = raw["funding_rate"]
                features["features"] = ai_features_array

                # Push to Redis asynchronously
                state_key = f"market:state:{symbol}"
                asyncio.create_task(redis_manager.set_state(state_key, features, ttl_seconds=60))

                # Add to DB buffer (will auto-flush when batch_size reached)
                timestamp = int(time.time() * 1000)
                asyncio.create_task(self.repository.add_feature(symbol, timestamp, features))

        elif event_type == "aggTrade":
            self.feature_engines[symbol].add_trade(payload)
            qty = float(payload.get("q", 0.0))
            raw["volume"] += qty
            if not payload.get("m", False):  # buyer is taker -> aggressive buy
                raw["buy_volume"] += qty
            raw["trade_count"] += 1.0

        elif event_type == "markPriceUpdate":
            # Extract funding rate and mark price, push to Redis AND update local cache
            funding_rate = float(payload.get("r", 0.0))
            raw["funding_rate"] = funding_rate
            mark, index = float(payload.get("p", 0.0)), float(payload.get("i", 0.0))
            if index > 0:
                raw["mark_price_premium"] = (mark - index) / index
            state_key = f"derivatives:state:{symbol}"
            deriv_state = {
                "mark_price": mark,
                "index_price": index,
                "funding_rate": funding_rate,
                "next_funding_time": payload.get("T", 0)
            }
            asyncio.create_task(redis_manager.set_state(state_key, deriv_state, ttl_seconds=60))

        elif event_type == "forceOrder":
            # Liquidation event: accumulate liquidated quantity for the liq_1m / liq_15m features
            order = payload.get("o", {})
            raw["liquidation_volume"] += float(order.get("q", 0.0))
            logger.debug("Liquidation Event Detected", extra={"symbol": symbol, "data": order})

    async def _snapshot_books(self):
        for sym in self.active_symbols:
            try:
                snapshot = await self.adapter._request("GET", "/fapi/v1/depth", params={"symbol": sym, "limit": 1000})
                self.order_books[sym].initialize_book(snapshot)
                logger.info(f"Initialized order book for {sym}")
            except Exception as e:
                logger.error(f"Failed to fetch order book snapshot for {sym}", exc_info=True)

    async def _consume(self, session: aiohttp.ClientSession, url: str):
        async with session.ws_connect(url, heartbeat=30) as ws:
            logger.info(f"Connected to market data stream: {url[:100]}...")
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_market_data(json.loads(msg.data))
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

    async def _stream_forever(self):
        """
        One connection per endpoint: on mainnet, depth is served on /public and trades / mark price /
        liquidations on /market (core/exchange/binance_streams.py). If any connection drops, all are
        reconnected with fresh order-book snapshots.
        """
        names = [f"{sym.lower()}@{st}" for sym in self.active_symbols for st in self.streams]
        urls = combined_stream_urls(names, market_data_is_mainnet())
        while self.running:
            await self._snapshot_books()
            async with aiohttp.ClientSession() as session:
                tasks = [asyncio.create_task(self._consume(session, u)) for u in urls]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                except Exception as e:
                    logger.error(f"Market data stream error: {e}")
                finally:
                    for t in tasks:
                        t.cancel()
            if self.running:
                logger.warning("Market data stream ended. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def start(self):
        logger.info("Starting Multi-Symbol Market Collector Service...", extra={"symbols": self.active_symbols, "streams": self.streams})
        self.running = True

        await redis_manager.connect()

        try:
            exchange_info = await self.adapter.get_exchange_info()
            logger.info("Successfully connected to Binance Futures REST API")
        except Exception as e:
            logger.error("Failed to connect to Binance REST API, exiting.")
            return

        # Pre-load ~16h of 1m bars so indicators / volatility are valid immediately
        await seed_engines(self.ai_engines)

        self._context_task = asyncio.create_task(poll_context(self.active_symbols, self.raw, lambda: self.running))
        self._task = asyncio.create_task(self._stream_forever())

        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Market Collector Service shutting down...")
        finally:
            self._task.cancel()
            self._context_task.cancel()
            await self.adapter.close()
            await redis_manager.disconnect()
            await self.repository.flush()

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
        if self._context_task:
            self._context_task.cancel()
