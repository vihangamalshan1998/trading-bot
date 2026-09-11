import numpy as np
from typing import List, Dict

class RegimeDetector:
    """
    Phase 9: Evaluates macro conditions over long time horizons to determine the market regime.
    Outputs a scalar: -1 (Bear), 0 (Chop/Crab), 1 (Bull).
    """
    def __init__(self, fast_window: int = 50, slow_window: int = 200, adx_window: int = 14):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.adx_window = adx_window
        self.price_history: Dict[str, List[float]] = {}
        
    def add_price(self, symbol: str, price: float):
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        
        # Keep only what we need
        if len(self.price_history[symbol]) > self.slow_window * 2:
            self.price_history[symbol].pop(0)
            
    def detect_regime(self, symbol: str) -> float:
        """
        Calculates the regime based on Moving Average crossover and a mock ADX (trend strength).
        Returns -1.0, 0.0, or 1.0
        """
        history = self.price_history.get(symbol, [])
        if len(history) < self.slow_window:
            return 0.0 # Default to chop if insufficient data
            
        fast_ma = np.mean(history[-self.fast_window:])
        slow_ma = np.mean(history[-self.slow_window:])
        
        # Extremely simplified trend strength (mock ADX)
        # In reality, this requires High/Low/Close calculations over the window
        # We proxy it by looking at the variance of the recent trend
        recent_prices = history[-self.adx_window:]
        trend_strength = abs(recent_prices[-1] - recent_prices[0]) / (np.std(recent_prices) + 1e-8)
        
        is_trending = trend_strength > 1.5 # Arbitrary threshold for "trending" vs "chopping"
        
        if not is_trending:
            return 0.0 # Chop
            
        if fast_ma > slow_ma:
            return 1.0 # Bull
        else:
            return -1.0 # Bear
