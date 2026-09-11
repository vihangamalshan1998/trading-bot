from core.logging.logger import logger
from typing import Dict, Any

class RiskManager:
    """
    Deterministic firewall to protect the portfolio from AI hallucinations.
    """
    def __init__(self, max_daily_drawdown: float = 0.05, max_leverage: int = 10, bearish_macro_threshold: float = -0.8):
        self.max_daily_drawdown = max_daily_drawdown
        self.max_leverage = max_leverage
        self.bearish_macro_threshold = bearish_macro_threshold
        
    def validate_action(self, symbol: str, ai_action: int, portfolio_state: Dict[str, Any], macro_state: Dict[str, Any]) -> int:
        """
        Validates the AI's intended action.
        Returns the original action if approved, or a safe override action (e.g., 0 HOLD or 3/4 CLOSE).
        """
        wallet_balance = portfolio_state.get("wallet_balance", 0.0)
        start_balance = portfolio_state.get("start_of_day_balance", wallet_balance)
        
        # 1. Max Drawdown Check
        if start_balance > 0:
            drawdown = (wallet_balance - start_balance) / start_balance
            if drawdown < -self.max_daily_drawdown:
                logger.warning(f"RISK MANAGER: Daily drawdown ({drawdown*100:.2f}%) exceeds limit. Forcing CLOSE/HOLD.")
                # Return CLOSE_LONG (3) or CLOSE_SHORT (4) based on position, or HOLD (0) if no position
                pos = portfolio_state.get("positions", {}).get(symbol, 0.0)
                if pos > 0:
                    return 3
                elif pos < 0:
                    return 4
                else:
                    return 0
                    
        # 2. Macro Override
        sentiment = macro_state.get("sentiment_score", 0.0)
        if sentiment < self.bearish_macro_threshold:
            if ai_action == 1: # OPEN_LONG
                logger.warning(f"RISK MANAGER: Macro sentiment ({sentiment}) is extremely bearish. Blocking OPEN_LONG on {symbol}.")
                return 0 # Override to HOLD
                
        # 3. Leverage Check (approximate)
        # In a real engine, we'd check if the intended notional exceeds max leverage limits.
        
        return ai_action
