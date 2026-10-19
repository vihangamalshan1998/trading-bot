import asyncio
import json
import logging
from typing import List
from core.db.redis import redis_manager
from core.exchange.binance_client import BinanceFuturesAdapter

logger = logging.getLogger(__name__)

class UniverseScreener:
    """
    Phase 15: Dynamic Symbol Selection.
    Polls the exchange's 24hr ticker API to rank coins by trading volume and volatility.
    Dynamically swaps out lowest-performing symbols with hot ones on a regular schedule,
    updating the Redis 'active_symbols' set so the TradingBot and FeatureEngine adapt on the fly.
    """
    def __init__(self, target_symbol_count: int = 5, poll_interval_seconds: int = 3600):
        self.target_count = target_symbol_count
        self.poll_interval = poll_interval_seconds
        self.client = BinanceFuturesAdapter()
        self.redis = redis_manager
        self._running = False
        self.active_set_key = "system:active_symbols"
        
    async def get_top_symbols(self) -> List[str]:
        """Fetches 24hr ticker data and sorts curated symbols by quote volume."""
        from core.config.settings import settings
        curated = getattr(settings, "symbol_universe", [])
        excluded = getattr(settings, "excluded_trading_symbols", [])
        active_curated = [s for s in curated if s not in excluded]
        try:
            tickers = await self.client.get_24hr_ticker()
            valid_tickers = [t for t in tickers if t.get('symbol') in active_curated and float(t.get('quoteVolume', 0)) > 0]
            sorted_tickers = sorted(valid_tickers, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
            top_symbols = [t['symbol'] for t in sorted_tickers]
            # Ensure any curated symbol missing from ticker response is still included
            for s in active_curated:
                if s not in top_symbols:
                    top_symbols.append(s)
            return top_symbols
        except Exception as e:
            logger.error(f"Failed to fetch top symbols: {e}")
            return active_curated

    async def update_active_symbols(self):
        """Updates the Redis set of active symbols for the rest of the system to read."""
        top_symbols = await self.get_top_symbols()
        if not top_symbols:
            return
            
        await self.redis.connect()
        redis_conn = self.redis.redis
        
        # We store them as a JSON list in a simple key for easy reading
        # Or as a Redis SET. A JSON list is easier for consistent ordering.
        await redis_conn.set(self.active_set_key, json.dumps(top_symbols))
        logger.info(f"Dynamically updated universe to top {self.target_count} symbols by volume: {top_symbols}")
        
    async def start(self):
        self._running = True
        logger.info("UniverseScreener started.")
        while self._running:
            await self.update_active_symbols()
            await asyncio.sleep(self.poll_interval)
            
    def stop(self):
        self._running = False
        logger.info("UniverseScreener stopping...")

if __name__ == "__main__":
    screener = UniverseScreener()
    try:
        asyncio.run(screener.start())
    except KeyboardInterrupt:
        screener.stop()
