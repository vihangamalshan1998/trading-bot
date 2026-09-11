import asyncio
import aiohttp
from typing import List
from core.logging.logger import logger
from core.config.settings import settings

class UniverseScreener:
    """
    Dynamically screens Binance Futures for the most liquid and volatile pairs.
    """
    def __init__(self, top_n: int = 10, poll_interval_hours: int = 1):
        self.top_n = top_n
        self.poll_interval = poll_interval_hours * 3600
        self.base_url = "https://fapi.binance.com" # Always use live data for screening
        self.active_symbols: List[str] = []
        self._running = False
        
    async def fetch_24hr_ticker(self) -> List[dict]:
        url = f"{self.base_url}/fapi/v1/ticker/24hr"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                logger.error(f"Failed to fetch 24hr ticker: {response.status}")
                return []
                
    async def update_universe(self):
        tickers = await self.fetch_24hr_ticker()
        if not tickers:
            return
            
        # Filter for USDT perpetuals only
        usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and '_' not in t['symbol']]
        
        # Sort by quote volume (liquidity)
        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        
        # Select Top N
        top_pairs = usdt_pairs[:self.top_n]
        new_symbols = [p['symbol'] for p in top_pairs]
        
        if new_symbols != self.active_symbols:
            logger.info(f"Universe Shift! Old: {self.active_symbols} -> New: {new_symbols}")
            self.active_symbols = new_symbols
            
            # In a full system, this would publish to Redis: "universe:updated" 
            # so the TradingBot and Collectors can gracefully drop old symbols and subscribe to new ones.
            
    async def start(self):
        self._running = True
        logger.info(f"Starting Universe Screener (Top {self.top_n} every {self.poll_interval/3600} hours)")
        while self._running:
            try:
                await self.update_universe()
            except Exception as e:
                logger.error(f"Error in Universe Screener: {e}")
                
            await asyncio.sleep(self.poll_interval)
            
    def stop(self):
        self._running = False

async def test_screener():
    screener = UniverseScreener(top_n=5, poll_interval_hours=1)
    await screener.update_universe()
    logger.info(f"Top 5 Liquid Pairs Right Now: {screener.active_symbols}")

if __name__ == "__main__":
    asyncio.run(test_screener())
