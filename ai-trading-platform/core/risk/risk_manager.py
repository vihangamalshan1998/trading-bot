import time
from typing import Dict, Any, List
from core.logging.logger import logger

class RiskManager:
    """
    Phase 6: Deterministic firewall that intercepts AI actions to enforce hard risk limits.
    """
    def __init__(self, 
                 max_daily_drawdown_pct: float = 0.05, 
                 max_symbol_exposure_pct: float = 0.20,
                 max_correlated_exposure_pct: float = 0.30):
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_symbol_exposure_pct = max_symbol_exposure_pct
        self.max_correlated_exposure_pct = max_correlated_exposure_pct
        
        self.daily_high_equity = 0.0
        self.last_day_reset = time.time()

    def evaluate_risk(self, 
                      current_equity: float, 
                      proposed_action_qty: float, 
                      symbol: str, 
                      current_price: float, 
                      portfolio_state: Dict[str, Any]) -> float:
        """
        Evaluates an AI's proposed action and clamps the quantity if it violates risk limits.
        Returns the safe quantity to execute.
        """
        # 1. Daily Drawdown Limit
        now = time.time()
        if now - self.last_day_reset > 86400:
            self.daily_high_equity = current_equity
            self.last_day_reset = now
            
        if current_equity > self.daily_high_equity:
            self.daily_high_equity = current_equity
            
        daily_drawdown = (self.daily_high_equity - current_equity) / self.daily_high_equity if self.daily_high_equity > 0 else 0.0
        
        if daily_drawdown >= self.max_daily_drawdown_pct:
            logger.error(f"RISK HALT: Daily drawdown limit breached ({daily_drawdown*100:.2f}%). Halting trading.")
            return 0.0 # Force hold
            
        # 2. Maximum Symbol Exposure
        current_symbol_notional = portfolio_state.get('positions', {}).get(symbol, 0.0)
        proposed_notional = abs(proposed_action_qty) * current_price
        
        total_symbol_exposure_pct = (current_symbol_notional + proposed_notional) / current_equity if current_equity > 0 else 0.0
        
        if total_symbol_exposure_pct > self.max_symbol_exposure_pct:
            logger.warning(f"RISK CLAMP: Proposed exposure ({total_symbol_exposure_pct*100:.2f}%) exceeds limit for {symbol}.")
            # Clamp the size
            max_allowed_notional = (self.max_symbol_exposure_pct * current_equity) - current_symbol_notional
            safe_qty = max(0.0, max_allowed_notional / current_price)
            # Match the sign of the proposed quantity
            return safe_qty if proposed_action_qty > 0 else -safe_qty
            
        # 3. Correlated Asset Exposure (e.g., BTC and ETH)
        if symbol in ["BTCUSDT", "ETHUSDT"]:
            btc_exp = portfolio_state.get('positions', {}).get("BTCUSDT", 0.0)
            eth_exp = portfolio_state.get('positions', {}).get("ETHUSDT", 0.0)
            correlated_exp_pct = (btc_exp + eth_exp + proposed_notional) / current_equity if current_equity > 0 else 0.0
            
            if correlated_exp_pct > self.max_correlated_exposure_pct:
                logger.warning(f"RISK CLAMP: Correlated exposure ({correlated_exp_pct*100:.2f}%) exceeds limit.")
                max_allowed_notional = (self.max_correlated_exposure_pct * current_equity) - (btc_exp + eth_exp)
                safe_qty = max(0.0, max_allowed_notional / current_price)
                return safe_qty if proposed_action_qty > 0 else -safe_qty
                
        return proposed_action_qty
