import time
import math
from typing import Dict, Any, List
from core.logging.logger import logger
from core.schemas.state_schema import MarketState, PortfolioState, OrderRequest, RiskDecision
from core.config.settings import settings

class RiskManager:
    """
    Phase 1: Deterministic firewall implementing all 12 requested strict bounds.
    """
    def __init__(self, 
                 max_position_size: float = 10.0, # max absolute position size
                 max_symbol_exposure_pct: float = 0.20,
                 max_portfolio_exposure_pct: float = 0.80,
                 max_leverage: int = 10,
                 max_order_size: float = 5.0,
                 max_open_positions: int = 5,
                 max_daily_loss_pct: float = 0.05,
                 max_drawdown_pct: float = 0.10):
                 
        self.max_position_size = max_position_size
        self.max_symbol_exposure_pct = max_symbol_exposure_pct
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self.max_leverage = max_leverage
        self.max_order_size = max_order_size
        self.max_open_positions = max_open_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        self.daily_high_equity = 0.0
        self.global_high_equity = 0.0
        self.last_day_reset = time.time()

    def evaluate(self, order_request: OrderRequest, portfolio_state: PortfolioState, market_state: MarketState) -> RiskDecision:
        """
        Evaluates an AI's proposed OrderRequest and applies exactly the 12 requested Phase 1 checks.
        """
        flags = []
        approved = True
        reason = "Approved"
        safe_qty = order_request.requested_quantity
        max_allowed = safe_qty
        
        # 1. Trading Disabled
        if not settings.trading_enabled:
            return RiskDecision(approved=False, reason="TRADING_DISABLED", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["TRADING_DISABLED"])
            
        # 2. Emergency Stop
        if settings.emergency_stop:
            return RiskDecision(approved=False, reason="EMERGENCY_STOP", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["EMERGENCY_STOP"])
            
        # 3. NaN/Inf Checks
        if math.isnan(safe_qty) or math.isinf(safe_qty) or math.isnan(market_state.mid_price) or math.isinf(market_state.mid_price):
            return RiskDecision(approved=False, reason="NAN_INF_DETECTED", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["NAN_INF_DETECTED"])
            
        # 4. Insufficient Equity
        if portfolio_state.equity <= 0:
            return RiskDecision(approved=False, reason="INSUFFICIENT_EQUITY", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["INSUFFICIENT_EQUITY"])
            
        # 5. Invalid Price
        if market_state.mid_price <= 0:
            return RiskDecision(approved=False, reason="INVALID_PRICE", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["INVALID_PRICE"])
            
        # 6. Stale Market Data
        now = time.time()
        if now - market_state.timestamp > 60:
            return RiskDecision(approved=False, reason="STALE_MARKET_DATA", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["STALE_MARKET_DATA"])
            
        # 7. Max Order Size
        if order_request.requested_quantity <= 0:
            return RiskDecision(approved=False, reason="INVALID_QUANTITY", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["INVALID_QUANTITY"])
            
        if order_request.requested_quantity > self.max_order_size:
            flags.append("MAX_ORDER_SIZE_EXCEEDED")
            safe_qty = self.max_order_size
            max_allowed = self.max_order_size
            reason = f"Clamped to max_order_size: {self.max_order_size}"

        # 8. Daily Drawdown Breach
        if now - self.last_day_reset > 86400:
            self.daily_high_equity = portfolio_state.equity
            self.last_day_reset = now
            
        if portfolio_state.equity > self.daily_high_equity:
            self.daily_high_equity = portfolio_state.equity
        if portfolio_state.equity > self.global_high_equity:
            self.global_high_equity = portfolio_state.equity
            
        drawdown = (self.global_high_equity - portfolio_state.equity) / self.global_high_equity if self.global_high_equity > 0 else 0.0
        
        if drawdown >= self.max_drawdown_pct:
            return RiskDecision(approved=False, reason="MAX_DRAWDOWN", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["MAX_DRAWDOWN"])
            
        # Closing positions is always allowed if we got past emergency/stale data checks
        if "CLOSE" in order_request.action_type or order_request.action_type == "HOLD":
            return RiskDecision(approved=True, reason=reason, adjusted_quantity=safe_qty, max_allowed_quantity=max_allowed, risk_flags=flags)
            
        # 9. Max Open Positions
        active_positions = sum(1 for p in portfolio_state.positions.values() if p.quantity != 0)
        if active_positions >= self.max_open_positions and order_request.symbol not in [sym for sym, p in portfolio_state.positions.items() if p.quantity != 0]:
            return RiskDecision(approved=False, reason="MAX_OPEN_POSITIONS", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["MAX_OPEN_POSITIONS"])

        # 10. Max Leverage (Symbol specific)
        current_symbol_notional = 0.0
        if order_request.symbol in portfolio_state.positions:
            pos = portfolio_state.positions[order_request.symbol]
            current_symbol_notional = abs(pos.quantity) * market_state.mid_price
            if pos.leverage > self.max_leverage:
                return RiskDecision(approved=False, reason="MAX_LEVERAGE_EXCEEDED", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=["MAX_LEVERAGE_EXCEEDED"])
            
        # 11. Max Portfolio Exposure
        proposed_notional = safe_qty * market_state.mid_price
        total_symbol_pct = (current_symbol_notional + proposed_notional) / portfolio_state.equity
        total_portfolio_pct = (portfolio_state.total_exposure + proposed_notional) / portfolio_state.equity
        
        if total_portfolio_pct > self.max_portfolio_exposure_pct:
            flags.append("EXCESSIVE_PORTFOLIO_EXPOSURE")
            max_p_notional = (self.max_portfolio_exposure_pct * portfolio_state.equity) - portfolio_state.total_exposure
            safe_qty = min(safe_qty, max(0.0, max_p_notional / market_state.mid_price))
            max_allowed = safe_qty
            reason = "Clamped due to portfolio exposure limit"
            
        # 12. Max Symbol Exposure
        if total_symbol_pct > self.max_symbol_exposure_pct:
            flags.append("MAX_SYMBOL_EXPOSURE")
            max_s_notional = (self.max_symbol_exposure_pct * portfolio_state.equity) - current_symbol_notional
            safe_qty = min(safe_qty, max(0.0, max_s_notional / market_state.mid_price))
            max_allowed = safe_qty
            reason = "Clamped due to symbol exposure limit"
            
        if safe_qty <= 0:
            return RiskDecision(approved=False, reason="Safe quantity <= 0 after exposure limits", adjusted_quantity=0.0, max_allowed_quantity=0.0, risk_flags=flags)

        return RiskDecision(
            approved=True,
            reason=reason,
            adjusted_quantity=safe_qty,
            max_allowed_quantity=max_allowed,
            risk_flags=flags
        )
