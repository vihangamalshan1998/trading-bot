import numpy as np
from typing import List

class BaseHeuristicStrategy:
    """
    Phase 7: Interface for baseline algorithms.
    Must return a continuous action vector of shape (num_symbols, 3) 
    where each symbol gets: [action_val, confidence, target_size].
    """
    def __init__(self, num_symbols: int):
        self.num_symbols = num_symbols
        self.history = [[] for _ in range(num_symbols)]
        
    def step(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class MomentumStrategy(BaseHeuristicStrategy):
    """
    Baseline 1: Trend Following (Momentum).
    Goes LONG if recent momentum is positive.
    Goes SHORT if recent momentum is negative.
    Outputs continuous actions.
    """
    def __init__(self, num_symbols: int, window: int = 10, threshold: float = 0.001):
        super().__init__(num_symbols)
        self.window = window
        self.threshold = threshold
        
    def step(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros((self.num_symbols, 3), dtype=np.float32)
        
        # obs structure: 9 Portfolio dims + N * 37 dims (25 Market + 12 Position) + Macro
        # The first market feature (idx 9 + i*37 + 0) could be treated as a proxy for price/momentum
        # For this baseline, we'll extract a naive synthetic price or use the VWAP deviation.
        
        for i in range(self.num_symbols):
            # Market features start at index 9 + i*37
            base_idx = 9 + (i * 37)
            # Assuming index 6 of market features (offset 6) is mid_price (based on earlier phases)
            # Or we can just use a simulated price from the obs
            current_price = obs[base_idx + 6] if len(obs) > base_idx + 6 else 100.0
            
            self.history[i].append(current_price)
            if len(self.history[i]) > self.window:
                self.history[i].pop(0)
                
            if len(self.history[i]) < self.window:
                actions[i] = [0.0, 0.0, 0.0] # HOLD
                continue
                
            momentum = (current_price - self.history[i][0]) / (self.history[i][0] + 1e-8)
            
            if momentum > self.threshold:
                # OPEN_LONG (action > 0.2), high confidence, 20% size
                actions[i] = [0.4, 0.8, 0.2]
            elif momentum < -self.threshold:
                # OPEN_SHORT (action < -0.2), high confidence, 20% size
                actions[i] = [-0.4, 0.8, 0.2]
            else:
                actions[i] = [0.0, 0.0, 0.0]
                
        return actions

class MeanReversionStrategy(BaseHeuristicStrategy):
    """
    Baseline 2: Mean Reversion / Stat Arb.
    Goes SHORT if price is significantly above the moving average.
    Goes LONG if price is significantly below the moving average.
    """
    def __init__(self, num_symbols: int, window: int = 20, deviation_threshold: float = 0.005):
        super().__init__(num_symbols)
        self.window = window
        self.threshold = deviation_threshold
        
    def step(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros((self.num_symbols, 3), dtype=np.float32)
        
        for i in range(self.num_symbols):
            base_idx = 9 + (i * 37)
            current_price = obs[base_idx + 6] if len(obs) > base_idx + 6 else 100.0
            
            self.history[i].append(current_price)
            if len(self.history[i]) > self.window:
                self.history[i].pop(0)
                
            if len(self.history[i]) < self.window:
                actions[i] = [0.0, 0.0, 0.0]
                continue
                
            moving_average = sum(self.history[i]) / len(self.history[i])
            deviation = (current_price - moving_average) / (moving_average + 1e-8)
            
            if deviation > self.threshold:
                # Overbought -> OPEN_SHORT
                actions[i] = [-0.4, 0.8, 0.2]
            elif deviation < -self.threshold:
                # Oversold -> OPEN_LONG
                actions[i] = [0.4, 0.8, 0.2]
            else:
                actions[i] = [0.0, 0.0, 0.0]
                
        return actions
