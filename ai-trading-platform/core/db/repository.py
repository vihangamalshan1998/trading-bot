import asyncio
import uuid
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from core.db.database import AsyncSessionLocal, async_sessionmaker
from core.database.models.market import MarketFeature
from core.database.models.ai import Experience
from core.logging.logger import logger

class MarketFeatureRepository:
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self._buffer: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def add_feature(self, symbol: str, timestamp: int, features: Dict[str, float]):
        """
        Add a feature snapshot to the buffer. If buffer reaches batch_size, flush it to DB.
        """
        record = {
            "symbol": symbol,
            "timestamp": timestamp,
            "best_bid": features.get("best_bid"),
            "best_ask": features.get("best_ask"),
            "spread_bps": features.get("spread_bps"),
            "mid_price": features.get("mid_price"),
            "micro_price": features.get("micro_price"),
            "imbalance": features.get("imbalance"),
            "vwap_recent": features.get("vwap_recent")
        }
        
        async with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self.batch_size:
                await self.flush()

    async def flush(self):
        """
        Flush the current buffer to the MySQL database.
        """
        async with self._lock:
            if not self._buffer:
                return
                
            records_to_insert = self._buffer.copy()
            self._buffer.clear()
            
        try:
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    # Bulk insert
                    session.add_all([MarketFeature(**record) for record in records_to_insert])
                logger.info(f"Flushed {len(records_to_insert)} market features to database")
        except Exception as e:
            logger.error("Failed to flush market features to database", exc_info=True)
            # Optionally put them back in the buffer if we want retry logic
            # async with self._lock:
            #     self._buffer.extend(records_to_insert)

class ExperienceRepository:
    """
    Handles batched async inserts for the massive Experience table.
    """
    def __init__(self, batch_size: int = 20):
        self.batch_size = batch_size
        self._buffer: List[Experience] = []
        self._lock = asyncio.Lock()
        
    async def add_experience(self, exp_data: Dict[str, Any]):
        """
        Add a single experience dictionary to the memory buffer.
        """
        # Ensure ID exists
        if "experience_id" not in exp_data:
            exp_data["experience_id"] = str(uuid.uuid4())
            
        exp = Experience(**exp_data)
        
        async with self._lock:
            self._buffer.append(exp)
            if len(self._buffer) >= self.batch_size:
                await self.flush_unlocked()
                
    async def flush(self):
        """Public flush method taking the lock."""
        async with self._lock:
            await self.flush_unlocked()
            
    async def flush_unlocked(self):
        if not self._buffer:
            return
            
        to_insert = self._buffer[:]
        self._buffer.clear()
        
        session = async_session_maker()
        try:
            session.add_all(to_insert)
            await session.commit()
            logger.info(f"Flushed {len(to_insert)} Experiences to MySQL")
        except Exception as e:
            logger.error("Failed to flush Experiences to MySQL", exc_info=True)
            await session.rollback()
        finally:
            await session.close()
