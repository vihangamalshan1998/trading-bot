import time
from typing import Dict, Any, List
from core.logging.logger import logger
from core.schemas.state_schema import MarketState, PortfolioState, OrderRequest, RiskDecision

class RiskManager:
    """
    Phase 1: Deterministic firewall that intercepts AI actions to enforce hard risk limits.
    Now correctly implements the canonical State Schemas and `evaluate` interface.
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

    def evaluate(self, request: OrderRequest, portfolio: PortfolioState, market: MarketState) -> RiskDecision:
        """
        Evaluates an AI's proposed OrderRequest and clamps the quantity if it violates risk limits.
        """
        flags = []
        approved = True
        reason = "Approved"
        
        # Current equity protection
        now = time.time()
        if now - self.last_day_reset > 86400:
            self.daily_high_equity = portfolio.equity
            self.last_day_reset = now
            
        if portfolio.equity > self.daily_high_equity:
            self.daily_high_equity = portfolio.equity
            
        daily_drawdown = (self.daily_high_equity - portfolio.equity) / self.daily_high_equity if self.daily_high_equity > 0 else 0.0
        
        # 1. Daily Drawdown Limit
        if daily_drawdown >= self.max_daily_drawdown_pct:
            flags.append("DAILY_DRAWDOWN_BREACH")
            return RiskDecision(
                approved=False,
                reason=f"Daily drawdown limit breached ({daily_drawdown*100:.2f}%)",
                adjusted_quantity=0.0,
                max_allowed_quantity=0.0,
                risk_flags=flags
            )
            
        # If it's a CLOSE or HOLD order, always allow it through drawdown checks
        if "CLOSE" in request.action_type or request.action_type == "HOLD":
            return RiskDecision(
                approved=True,
                reason="Closing/Holding positions is safe",
                adjusted_quantity=request.requested_quantity,
                max_allowed_quantity=request.requested_quantity,
                risk_flags=flags
            )
            
        # 2. Maximum Symbol Exposure
        # Calculate current notional for this symbol
        current_symbol_notional = 0.0
        if request.symbol in portfolio.positions:
            pos = portfolio.positions[request.symbol]
            current_symbol_notional = abs(pos.quantity) * market.mid_price
            
        proposed_notional = request.requested_quantity * market.mid_price
        
        total_symbol_exposure_pct = (current_symbol_notional + proposed_notional) / portfolio.equity if portfolio.equity > 0 else 0.0
        
        safe_qty = request.requested_quantity
        max_allowed_qty = request.requested_quantity
        
        if total_symbol_exposure_pct > self.max_symbol_exposure_pct:
            flags.append("MAX_SYMBOL_EXPOSURE")
            max_allowed_notional = (self.max_symbol_exposure_pct * portfolio.equity) - current_symbol_notional
            safe_qty = max(0.0, max_allowed_notional / market.mid_price)
            max_allowed_qty = safe_qty
            approved = True
            reason = f"Clamped exposure to {self.max_symbol_exposure_pct*100:.1f}%"
            
        # 3. Correlated Asset Exposure (e.g., BTC and ETH)
        if request.symbol in ["BTCUSDT", "ETHUSDT"]:
            btc_exp = 0.0
            eth_exp = 0.0
            if "BTCUSDT" in portfolio.positions:
                btc_exp = abs(portfolio.positions["BTCUSDT"].quantity) * (market.mid_price if request.symbol == "BTCUSDT" else portfolio.positions["BTCUSDT"].current_price)
            if "ETHUSDT" in portfolio.positions:
                eth_exp = abs(portfolio.positions["ETHUSDT"].quantity) * (market.mid_price if request.symbol == "ETHUSDT" else portfolio.positions["ETHUSDT"].current_price)
                
            correlated_exp_pct = (btc_exp + eth_exp + (safe_qty * market.mid_price)) / portfolio.equity if portfolio.equity > 0 else 0.0
            
            if correlated_exp_pct > self.max_correlated_exposure_pct:
                flags.append("MAX_CORRELATED_EXPOSURE")
                max_allowed_notional = (self.max_correlated_exposure_pct * portfolio.equity) - (btc_exp + eth_exp)
                correlated_safe_qty = max(0.0, max_allowed_notional / market.mid_price)
                safe_qty = min(safe_qty, correlated_safe_qty)
                max_allowed_qty = safe_qty
                reason = f"Clamped due to correlated exposure limit ({self.max_correlated_exposure_pct*100:.1f}%)"

        if safe_qty <= 0:
            approved = False
            reason = "Safe quantity <= 0 after constraints."

        return RiskDecision(
            approved=approved,
            reason=reason,
            adjusted_quantity=safe_qty,
            max_allowed_quantity=max_allowed_qty,
            risk_flags=flags
        )
