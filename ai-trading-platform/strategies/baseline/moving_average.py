import numpy as np
from typing import Tuple

class MovingAverageStrategy:
    """
    Phase 6: Baseline Technical Strategy - VWAP Crossover.
    Uses canonical MarketState 25-dim vector.
    """
    def __init__(self):
        # Index 15 is vwap_deviation = (price - vwap) / vwap
        self.vwap_dev_idx = 15
        self.threshold = 0.001 # 0.1% deviation required to trigger signal
        
    def predict(self, market_state: np.ndarray, current_position: float = 0.0) -> Tuple[int, float, float]:
        """
        Returns (action_type, confidence, target_size).
        action_type mapped to our continuous [-1, 1] space:
        > 0.2: OPEN_LONG, < -0.2: OPEN_SHORT
        """
        if len(market_state) < 25:
            return 0.0, 0.0, 0.0 # HOLD
            
        vwap_dev = market_state[self.vwap_dev_idx]
        
        # Bullish signal
        if vwap_dev > self.threshold:
            if current_position <= 0:
                return 0.5, 0.8, 0.5 # OPEN_LONG (moderate size, high confidence)
            else:
                return 0.0, 0.0, 0.0 # Already long
                
        # Bearish signal
        elif vwap_dev < -self.threshold:
            if current_position >= 0:
                return -0.5, 0.8, 0.5 # OPEN_SHORT
            else:
                return 0.0, 0.0, 0.0 # Already short
                
        return 0.0, 0.0, 0.0 # HOLD
