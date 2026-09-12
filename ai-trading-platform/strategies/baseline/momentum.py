import numpy as np
from typing import Tuple

class MomentumStrategy:
    """
    Phase 6: Baseline Technical Strategy - Time-Series Momentum.
    Uses canonical MarketState 25-dim vector.
    """
    def __init__(self):
        # Index 3 is ret_15m, index 9 is trade_imbalance
        self.ret_15m_idx = 3
        self.trade_imb_idx = 9
        
    def predict(self, market_state: np.ndarray, current_position: float = 0.0) -> Tuple[float, float, float]:
        """
        Returns (action_type, confidence, target_size).
        action_type mapped to our continuous [-1, 1] space.
        """
        if len(market_state) < 25:
            return 0.0, 0.0, 0.0 # HOLD
            
        ret_15 = market_state[self.ret_15m_idx]
        imb = market_state[self.trade_imb_idx]
        
        # Strong uptrend and buying pressure
        if ret_15 > 0.005 and imb > 0.2:
            if current_position <= 0:
                return 0.8, 0.9, 1.0 # Aggressive LONG
                
        # Strong downtrend and selling pressure
        elif ret_15 < -0.005 and imb < -0.2:
            if current_position >= 0:
                return -0.8, 0.9, 1.0 # Aggressive SHORT
                
        # Stop loss / Mean reversion
        if current_position > 0 and ret_15 < -0.002:
            return 0.9, 0.7, 1.0 # CLOSE_LONG
            
        if current_position < 0 and ret_15 > 0.002:
            return -0.9, 0.7, 1.0 # CLOSE_SHORT
                
        return 0.0, 0.0, 0.0 # HOLD
