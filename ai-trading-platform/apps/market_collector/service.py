import asyncio
from typing import Dict, Any, List
import time

from core.logging.logger import logger
from core.config.settings import settings
from core.exchange.binance_client import BinanceFuturesAdapter
from apps.orderbook.builder import OrderBookBuilder
from apps.feature_engine.calculators import FeatureEngine
from apps.feature_engine.engine import FeatureEngine as AIEngine
from core.db.redis import redis_manager
from core.db.repository import MarketFeatureRepository
from core.exchange.symbol_registry import registry, SymbolConfig

class MarketCollectorService:
    def __init__(self):
        self.adapter = BinanceFuturesAdapter()
        
        # Initialize registry with target universe
        for sym in settings.symbol_universe:
            registry.add_symbol(SymbolConfig(symbol=sym))
            registry.update_status(sym, "ENABLED")
            
        self.active_symbols = [s.symbol for s in registry.get_active_symbols()]
        
        # We want order book depth, aggregate trades, mark price/funding, and liquidations
        self.streams = ["depth@100ms", "aggTrade", "markPrice@1s", "forceOrder"]
        self.running = False
        self._task = None
        
        # State per symbol
        self.order_books: Dict[str, OrderBookBuilder] = {sym: OrderBookBuilder(sym) for sym in self.active_symbols}
        self.feature_engines: Dict[str, FeatureEngine] = {sym: FeatureEngine() for sym in self.active_symbols}
        self.ai_engines: Dict[str, AIEngine] = {sym: AIEngine(sym) for sym in self.active_symbols}
        self._update_counters: Dict[str, int] = {sym: 0 for sym in self.active_symbols}
        self._funding_rates: Dict[str, float] = {sym: 0.0 for sym in self.active_symbols}  # Live funding rates
        
        self.repository = MarketFeatureRepository(batch_size=50)

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
             
        if symbol not in self.active_symbols:
            return
            
        event_type = payload.get("e", "unknown")
        
        if event_type == "depthUpdate":
            self.order_books[symbol].process_update(payload)
            self._update_counters[symbol] += 1
            
            # Periodically extract features and push to Redis and MySQL
            if self._update_counters[symbol] % 10 == 0:
                # Dashboard UI features (calculators.py)
                features = self.feature_engines[symbol].extract_features(self.order_books[symbol])
                features["best_bid"] = self.order_books[symbol].get_best_bid()
                features["best_ask"] = self.order_books[symbol].get_best_ask()
                
                # AI Model features (engine.py)
                # We need to simulate the 'tick' for engine.py
                tick_data = {
                    "mid_price": features["mid_price"],
                    "best_bid": features["best_bid"],
                    "best_ask": features["best_ask"],
                    "bid_qty": float(self.order_books[symbol].bids.get(features["best_bid"], 0)),
                    "ask_qty": float(self.order_books[symbol].asks.get(features["best_ask"], 0)),
                    "volume": 0.0, # Filled by aggTrade
                    "buy_volume": 0.0,
                    "trade_count": 0.0,
                    "funding_rate": self._funding_rates.get(symbol, 0.0)
                }
                self.ai_engines[symbol].add_tick(tick_data)
                ai_features_array = self.ai_engines[symbol].compute_features().tolist()
                
                # Fill missing Pydantic fields for MarketState
                features["symbol"] = symbol
                features["timestamp"] = time.time()
                features["bid"] = features["best_bid"] or features["mid_price"]
                features["ask"] = features["best_ask"] or features["mid_price"]
                features["last_price"] = features["mid_price"]
                features["spread"] = max(0.0, features["ask"] - features["bid"])
                features["order_book_imbalance"] = features["imbalance"]
                features["volume"] = 0.0
                features["vwap"] = features["vwap_recent"] or features["mid_price"]
                features["volatility"] = 0.0
                features["funding_rate"] = self._funding_rates.get(symbol, 0.0)  # Inject live funding rate
                features["features"] = ai_features_array
                
                # Push to Redis asynchronously
                state_key = f"market:state:{symbol}"
                asyncio.create_task(redis_manager.set_state(state_key, features, ttl_seconds=60))
                
                # Add to DB buffer (will auto-flush when batch_size reached)
                timestamp = int(time.time() * 1000)
                asyncio.create_task(self.repository.add_feature(symbol, timestamp, features))
                
        elif event_type == "aggTrade":
            self.feature_engines[symbol].add_trade(payload)
            
        elif event_type == "markPriceUpdate":
            # Extract funding rate and mark price, push to Redis AND update local cache
            funding_rate = float(payload.get("r", 0.0))
            self._funding_rates[symbol] = funding_rate  # Cache for next MarketState publish
            state_key = f"derivatives:state:{symbol}"
            deriv_state = {
                "mark_price": float(payload.get("p", 0.0)),
                "index_price": float(payload.get("i", 0.0)),
                "funding_rate": funding_rate,
                "next_funding_time": payload.get("T", 0)
            }
            asyncio.create_task(redis_manager.set_state(state_key, deriv_state, ttl_seconds=60))
            
        elif event_type == "forceOrder":
            # Liquidation event
            logger.warning("Liquidation Event Detected", extra={"symbol": symbol, "data": payload.get("o", {})})

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

        # Fetch initial order book snapshots for all symbols concurrently
        logger.info("Fetching order book snapshots...")
        # Note: In a real system, you'd want to rate-limit or batch these snapshot requests to avoid 429s.
        # But for 5 symbols, it's fine to run them sequentially or concurrently.
        for sym in self.active_symbols:
            try:
                snapshot = await self.adapter._request("GET", "/fapi/v1/depth", params={"symbol": sym, "limit": 1000})
                self.order_books[sym].initialize_book(snapshot)
                logger.info(f"Initialized order book for {sym}")
            except Exception as e:
                logger.error(f"Failed to fetch order book snapshot for {sym}", exc_info=True)

        # Start WebSocket listener
        self._task = asyncio.create_task(
            self.adapter.subscribe_market_data(
                symbols=self.active_symbols,
                streams=self.streams,
                callback=self._handle_market_data
            )
        )
        
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Market Collector Service shutting down...")
        finally:
            self._task.cancel()
            await self.adapter.close()
            await redis_manager.disconnect()
            await self.repository.flush()

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
