import json
from typing import List, Dict, Any
from core.db.redis import redis_manager
from core.logging.logger import logger

class EventMemorySystem:
    """
    Maintains a rolling buffer of major macroeconomic events for the AI's contextual awareness.
    """
    def __init__(self, max_events: int = 10):
        self.max_events = max_events
        self.redis_key = "macro:memory:events"
        
    async def add_event(self, event_data: Dict[str, Any]):
        """Pushes a new macro event into the historical memory buffer."""
        if redis_manager.redis is None:
            await redis_manager.connect()
            
        try:
            # LPUSH and LTRIM to keep only the last max_events
            await redis_manager.redis.lpush(self.redis_key, json.dumps(event_data))
            await redis_manager.redis.ltrim(self.redis_key, 0, self.max_events - 1)
            logger.info(f"Memory updated with event: {event_data.get('type', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to add event to memory: {e}")
            
    async def get_recent_events(self) -> List[Dict[str, Any]]:
        """Retrieves the chronological list of recent macro events."""
        if redis_manager.redis is None:
            await redis_manager.connect()
            
        try:
            # Lrange gets everything in the list
            events_json = await redis_manager.redis.lrange(self.redis_key, 0, -1)
            
            events = []
            for e_json in events_json:
                events.append(json.loads(e_json))
                
            # Reverse to return chronological order (oldest first)
            return list(reversed(events))
        except Exception as e:
            logger.error(f"Failed to fetch event memory: {e}")
            return []

memory_system = EventMemorySystem()

import numpy as np
import time

class EventMemoryBuffer:
    """
    Phase 11: Stores specific discrete events (liquidations, spikes, massive drawdowns).
    Provides an encoded 'memory state' vector to inject into the neural network,
    allowing it to remember recent traumatic or profitable events.
    """
    def __init__(self, memory_dim: int = 5, decay_rate: float = 0.99):
        self.memory_dim = memory_dim
        self.decay_rate = decay_rate
        
        # 5-dim memory vector:
        # [0]: Recent Liquidation intensity
        # [1]: Recent Massive PnL loss intensity
        # [2]: Recent Massive PnL win intensity
        # [3]: Recent Extreme Volatility spike intensity
        # [4]: Time since last major event (normalized)
        self.state_vector = np.zeros(self.memory_dim, dtype=np.float32)
        self.last_event_time = time.time()
        
    def add_event(self, event_type: str, intensity: float):
        """Register a new event to boost memory vector."""
        self.last_event_time = time.time()
        
        if event_type == "liquidation":
            self.state_vector[0] = min(1.0, self.state_vector[0] + intensity)
        elif event_type == "massive_loss":
            self.state_vector[1] = min(1.0, self.state_vector[1] + intensity)
        elif event_type == "massive_win":
            self.state_vector[2] = min(1.0, self.state_vector[2] + intensity)
        elif event_type == "volatility_spike":
            self.state_vector[3] = min(1.0, self.state_vector[3] + intensity)
            
    def step(self) -> np.ndarray:
        """Decays the memory state over time and returns the current vector."""
        self.state_vector[0:4] *= self.decay_rate
        
        time_elapsed = time.time() - self.last_event_time
        normalized_time = min(1.0, time_elapsed / 1000.0)
        self.state_vector[4] = normalized_time
        
        return self.state_vector.copy()
