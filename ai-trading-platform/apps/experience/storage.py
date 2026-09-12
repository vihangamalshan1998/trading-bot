import asyncio
import json
from core.db.redis import redis_manager
from core.logging.logger import logger
from core.ai.replay_buffer import ReplayBuffer

class ExperienceStorageService:
    """
    Phase 8: Listens for completed trade experiences on Redis and flushes them to MySQL.
    """
    def __init__(self):
        self.replay_buffer = ReplayBuffer()
        
    async def start(self):
        await redis_manager.connect()
        if not redis_manager.redis:
            logger.error("Could not connect to Redis for Experience Storage.")
            return
            
        pubsub = redis_manager.redis.pubsub()
        await pubsub.subscribe("experience:completed")
        logger.info("Experience Storage Service listening on 'experience:completed'")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    exp_data = json.loads(message['data'])
                    self.replay_buffer.add_experience(exp_data)
                    logger.info(f"Stored experience for {exp_data.get('symbol')} with reward {exp_data.get('reward')}")
                except Exception as e:
                    logger.error(f"Failed to process experience: {e}")
                    
if __name__ == "__main__":
    service = ExperienceStorageService()
    asyncio.run(service.start())
