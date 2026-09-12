import asyncio
import pandas as pd
from datetime import datetime, timedelta
from core.exchange.binance_client import BinanceFuturesAdapter
from apps.feature_engine.engine import FeatureEngine
from core.database.session import SessionLocal
from core.database.models.experience import Experience
from core.logging.logger import logger
from core.config.settings import settings

class HistoricalDownloader:
    """
    Downloads historical K-lines from Binance Futures, processes them through the 
    canonical FeatureEngine (producing 25-dim MarketState vectors), and saves them 
    into the Experience table for offline PPO training.
    """
    def __init__(self):
        self.binance = BinanceFuturesAdapter()
        self.feature_engine = FeatureEngine()
        self.SessionLocal = SessionLocal
        self.symbols = settings.symbol_universe
        
    async def fetch_klines(self, symbol: str, start_time: int, end_time: int, limit: int = 1500) -> list:
        # Binance API limits 1500 per request
        params = {
            "symbol": symbol,
            "interval": "1m", # 1 minute base resolution
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        # In a real setup, we'd use a robust HTTP client with retry logic.
        # For demonstration, we'll assume the public endpoint works.
        try:
            return await self.binance._request("GET", "/fapi/v1/klines", params)
        except Exception as e:
            logger.error(f"Failed to fetch historical klines: {e}")
            return []

    async def download_and_process(self, days_back: int = 30):
        logger.info(f"Starting historical data download for {days_back} days...")
        now = int(datetime.now().timestamp() * 1000)
        start_time = now - (days_back * 24 * 60 * 60 * 1000)
        
        for symbol in self.symbols:
            logger.info(f"Downloading historical data for {symbol}")
            current_start = start_time
            
            all_klines = []
            while current_start < now:
                # Chunk into 1500 minute segments (~25 hours)
                current_end = min(current_start + (1500 * 60 * 1000), now)
                
                klines = await self.fetch_klines(symbol, current_start, current_end)
                if not klines:
                    break
                    
                all_klines.extend(klines)
                current_start = current_end + 1
                await asyncio.sleep(0.5) # Rate limiting
                
            logger.info(f"Fetched {len(all_klines)} candles for {symbol}. Processing features...")
            
            # Format: [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore]
            df = pd.DataFrame(all_klines, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
            df = df.astype(float)
            
            # Process sequentially to build the feature state
            with self.SessionLocal() as session:
                for idx, row in df.iterrows():
                    # Mock a websocket payload
                    ws_data = {
                        "s": symbol,
                        "c": row["close"],
                        "h": row["high"],
                        "l": row["low"],
                        "v": row["volume"],
                        "q": row["quote_volume"],
                        "n": row["trades"]
                    }
                    
                    market_state = self.feature_engine.process_trade(ws_data)
                    if market_state is None:
                        continue # Still warming up
                        
                    # Create a dummy experience for historical state (Action = HOLD)
                    # We can use offline RL to learn from historical price movements
                    exp = Experience(
                        timestamp=row["close_time"] / 1000.0,
                        symbol=symbol,
                        market_state=market_state.features,
                        portfolio_state=[10000.0, 10000.0], # Dummy
                        position_state=[0.0, 0.0], # Dummy
                        action_type="HOLD",
                        confidence=0.0,
                        requested_size=0.0,
                        approved_size=0.0,
                        reward=0.0, # We'd need a labeler here to calculate forward returns
                        model_version="historical"
                    )
                    session.add(exp)
                    
                    if idx % 1000 == 0:
                        session.commit()
                        logger.info(f"[{symbol}] Inserted {idx}/{len(df)} historical states.")
                        
                session.commit()
                logger.info(f"[{symbol}] Finished historical insertion.")

if __name__ == "__main__":
    downloader = HistoricalDownloader()
    asyncio.run(downloader.download_and_process(days_back=7)) # Download last 7 days as example
