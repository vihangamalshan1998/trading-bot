import json
import asyncio
from collections import deque
from core.logging.logger import logger

class ReplayBuffer:
    """
    Records live trades from the TradingBot for continuous online RL training.
    """
    def __init__(self, max_size: int = 10000):
        # We keep a rolling window in memory for quick sampling.
        self.memory = deque(maxlen=max_size)
        
    def add(self, state: list, action: int, reward: float, next_state: list, done: bool, symbol: str, model_version: str):
        """Adds a live experience tuple to the buffer."""
        experience_tuple = (state, action, reward, next_state, done)
        self.memory.append(experience_tuple)
        
        # In production, we push to MySQL asynchronously using ExperienceRepository
        logger.debug(f"Added experience to replay buffer. Total: {len(self.memory)}")
            
    def sample(self, batch_size: int) -> list:
        """Samples a batch of experiences for the online trainer."""
        import random
        if len(self.memory) < batch_size:
            return list(self.memory)
        return random.sample(self.memory, batch_size)
        
replay_buffer = ReplayBuffer()
