import json
import redis.asyncio as redis
from typing import Optional, Any, Dict
from core.config.settings import settings
from core.logging.logger import logger

class RedisManager:
    _instance: Optional['RedisManager'] = None
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        
    @classmethod
    def get_instance(cls) -> 'RedisManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    async def connect(self):
        if self.redis is None:
            try:
                self.redis = redis.from_url(settings.redis_url, decode_responses=True)
                # Test connection
                await self.redis.ping()
                logger.info("Successfully connected to Redis")
            except Exception as e:
                logger.error("Failed to connect to Redis", exc_info=True)
                self.redis = None
                
    async def disconnect(self):
        if self.redis is not None:
            await self.redis.aclose()
            self.redis = None
            logger.info("Disconnected from Redis")
            
    async def set_state(self, key: str, data: Dict[str, Any], ttl_seconds: int = 60):
        if self.redis is None:
            return
        try:
            payload = json.dumps(data)
            await self.redis.set(key, payload, ex=ttl_seconds)
        except Exception as e:
            logger.error("Failed to set state in Redis", extra={"key": key, "error": str(e)})

redis_manager = RedisManager.get_instance()
