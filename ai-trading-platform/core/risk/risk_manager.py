import math
import time

from core.config.settings import settings
from core.logging.logger import logger
from core.schemas.state_schema import (
    MarketState,
    OrderRequest,
    PortfolioState,
    MacroState,
    RiskDecision,
)


class RiskManager:
    """
    Phase 1 deterministic risk firewall.

    IMPORTANT:
    - AI proposes an action.
    - RiskManager decides whether that action is allowed.
    - RiskManager can reduce quantity.
    - RiskManager can never increase quantity.
    - RiskManager cannot be bypassed.
    """

    def __init__(self):
        # Centralized configuration
        self.max_position_size = settings.max_position_size
        self.max_symbol_exposure_pct = settings.max_symbol_exposure_pct
        self.max_portfolio_exposure_pct = settings.max_portfolio_exposure_pct
        self.max_leverage = settings.max_leverage
        self.max_order_size = settings.max_order_size
        self.max_open_positions = settings.max_open_positions
        self.max_daily_loss_pct = settings.max_daily_loss_pct
        self.max_drawdown_pct = settings.max_drawdown_pct
        self.max_market_data_age_seconds = (
            settings.max_market_data_age_seconds
        )
        self.correlated_exposure_limit_pct = (
            settings.correlated_exposure_limit_pct
        )
        
        # Macro thresholds
        self.bearish_macro_threshold = -0.8
        self.bullish_macro_threshold = 0.8

        self.daily_high_equity = 0.0
        self.global_high_equity = 0.0
        self.last_day_reset = time.time()

    @staticmethod
    def _reject(reason: str) -> RiskDecision:
        return RiskDecision(
            approved=False,
            reason=reason,
            adjusted_quantity=0.0,
            max_allowed_quantity=0.0,
            risk_flags=[reason],
        )

    def _update_equity_highs(self, equity: float, now: float) -> None:
        # Reset daily high once per 24h.
        if now - self.last_day_reset >= 86400:
            self.daily_high_equity = equity
            self.last_day_reset = now

        if self.daily_high_equity <= 0:
            self.daily_high_equity = equity

        if self.global_high_equity <= 0:
            self.global_high_equity = equity

        self.daily_high_equity = max(
            self.daily_high_equity,
            equity,
        )

        self.global_high_equity = max(
            self.global_high_equity,
            equity,
        )

    def _correlated_exposure(
        self,
        portfolio: PortfolioState,
        market: MarketState,
    ) -> float:
        """
        Phase-1 conservative BTC/ETH correlated exposure.

        Uses current position prices when available.
        Uses current market price for the requested symbol.

        This is NOT a statistical correlation model.
        It is a conservative exposure bucket.
        """

        correlated_symbols = {"BTCUSDT", "ETHUSDT"}

        exposure = 0.0

        for symbol, position in portfolio.positions.items():
            if symbol not in correlated_symbols:
                continue

            if position.quantity == 0:
                continue

            if symbol == market.symbol:
                price = market.mid_price
            elif position.current_price > 0:
                price = position.current_price
            else:
                # Cannot safely value unknown exposure.
                return float("inf")

            exposure += abs(position.quantity) * price

        return exposure

    def evaluate(
        self,
        order_request: OrderRequest,
        portfolio_state: PortfolioState,
        market_state: MarketState,
        macro_state: MacroState = None,
    ) -> RiskDecision:

        now = time.time()

        # ========================================================
        # 1. Global safety gates
        # ========================================================

        if not settings.trading_enabled:
            return self._reject("TRADING_DISABLED")

        if settings.emergency_stop:
            return self._reject("EMERGENCY_STOP")

        # ========================================================
        # 2. Validate core numerical values
        # ========================================================

        qty = float(order_request.requested_quantity)
        price = float(market_state.mid_price)

        if not math.isfinite(qty):
            return self._reject("NAN_INF_DETECTED")

        if not math.isfinite(price):
            return self._reject("NAN_INF_DETECTED")

        if qty <= 0:
            return self._reject("INVALID_QUANTITY")

        if price <= 0:
            return self._reject("INVALID_PRICE")

        # ========================================================
        # 3. Market freshness
        # ========================================================

        age = now - market_state.timestamp

        if age < 0:
            return self._reject("INVALID_MARKET_TIMESTAMP")

        if age > self.max_market_data_age_seconds:
            return self._reject("STALE_MARKET_DATA")

        # ========================================================
        # 4. Portfolio health
        # ========================================================

        if not math.isfinite(portfolio_state.equity):
            return self._reject("INVALID_EQUITY")

        if portfolio_state.equity <= 0:
            return self._reject("INSUFFICIENT_EQUITY")

        if not math.isfinite(portfolio_state.free_margin):
            return self._reject("INVALID_FREE_MARGIN")

        if portfolio_state.free_margin < 0:
            return self._reject("INSUFFICIENT_FREE_MARGIN")

        # ========================================================
        # 5. Update drawdown tracking
        # ========================================================

        self._update_equity_highs(
            portfolio_state.equity,
            now,
        )

        # ========================================================
        # 6. Closing/HOLD actions
        # ========================================================

        action = order_request.action_type.upper()

        if action == "HOLD" or "CLOSE" in action:
            return RiskDecision(
                approved=True,
                reason="Approved",
                adjusted_quantity=qty,
                max_allowed_quantity=qty,
                risk_flags=[],
            )

        daily_loss = 0.0

        if self.daily_high_equity > 0:
            daily_loss = (
                self.daily_high_equity - portfolio_state.equity
            ) / self.daily_high_equity

        if daily_loss >= self.max_daily_loss_pct:
            return self._reject("MAX_DAILY_LOSS")

        drawdown = 0.0

        if self.global_high_equity > 0:
            drawdown = (
                self.global_high_equity - portfolio_state.equity
            ) / self.global_high_equity

        if drawdown >= self.max_drawdown_pct:
            return self._reject("MAX_DRAWDOWN")

        # Only OPEN actions continue.
        if action not in {"OPEN_LONG", "OPEN_SHORT"}:
            return self._reject("INVALID_ACTION")
            
        # ========================================================
        # 6.5 Macro Override
        # ========================================================
        if macro_state is not None:
            sentiment = getattr(macro_state, "sentiment_score", 0.0)
            if sentiment < self.bearish_macro_threshold and action == "OPEN_LONG":
                logger.warning(f"RISK MANAGER: Macro sentiment ({sentiment}) is extremely bearish. Blocking OPEN_LONG on {order_request.symbol}.")
                return self._reject("MACRO_BEARISH_OVERRIDE")
            elif sentiment > self.bullish_macro_threshold and action == "OPEN_SHORT":
                logger.warning(f"RISK MANAGER: Macro sentiment ({sentiment}) is extremely bullish. Blocking OPEN_SHORT on {order_request.symbol}.")
                return self._reject("MACRO_BULLISH_OVERRIDE")

        # ========================================================
        # 7. Maximum single-order quantity
        # ========================================================

        if qty > self.max_order_size:
            return self._reject("MAX_ORDER_SIZE_EXCEEDED")

        # ========================================================
        # 8. Maximum open positions
        # ========================================================

        active_symbols = {
            symbol
            for symbol, position in portfolio_state.positions.items()
            if position.quantity != 0
        }

        if (
            len(active_symbols) >= self.max_open_positions
            and order_request.symbol not in active_symbols
        ):
            return self._reject("MAX_OPEN_POSITIONS")

        # ========================================================
        # 9. Existing position
        # ========================================================

        position = portfolio_state.positions.get(
            order_request.symbol
        )

        current_quantity = 0.0
        current_leverage = 1

        if position is not None:
            current_quantity = float(position.quantity)
            current_leverage = int(position.leverage)

            if current_leverage > self.max_leverage:
                return self._reject("MAX_LEVERAGE_EXCEEDED")

        # ========================================================
        # 10. Maximum resulting position size
        # ========================================================

        if action == "OPEN_LONG":
            resulting_quantity = current_quantity + qty
        else:
            resulting_quantity = current_quantity - qty

        if abs(resulting_quantity) > self.max_position_size:
            return self._reject("MAX_POSITION_SIZE_EXCEEDED")

        # ========================================================
        # 11. Proposed notional
        # ========================================================

        proposed_notional = qty * price

        if notional_is_invalid := (
            not math.isfinite(proposed_notional)
            or proposed_notional <= 0
        ):
            return self._reject("INVALID_NOTIONAL")

        # ========================================================
        # 12. Conservative margin check
        # ========================================================

        # Phase 1 does not yet have authoritative exchange leverage
        # for the requested order.
        #
        # Therefore we use max_leverage conservatively.
        required_margin = proposed_notional / self.max_leverage

        if portfolio_state.free_margin < required_margin:
            return self._reject("INSUFFICIENT_FREE_MARGIN")

        # ========================================================
        # 13. Portfolio exposure
        # ========================================================

        proposed_portfolio_exposure = (
            portfolio_state.total_exposure
            + proposed_notional
        )

        portfolio_exposure_pct = (
            proposed_portfolio_exposure
            / portfolio_state.equity
        )

        if portfolio_exposure_pct > self.max_portfolio_exposure_pct:
            return self._reject(
                "EXCESSIVE_PORTFOLIO_EXPOSURE"
            )

        # ========================================================
        # 14. Symbol exposure
        # ========================================================

        current_symbol_notional = 0.0

        if position is not None and current_quantity != 0:
            current_symbol_price = (
                position.current_price
                if position.current_price > 0
                else price
            )

            current_symbol_notional = (
                abs(current_quantity)
                * current_symbol_price
            )

        proposed_symbol_exposure = (
            current_symbol_notional
            + proposed_notional
        )

        symbol_exposure_pct = (
            proposed_symbol_exposure
            / portfolio_state.equity
        )

        if symbol_exposure_pct > self.max_symbol_exposure_pct:
            return self._reject(
                "MAX_SYMBOL_EXPOSURE"
            )

        # ========================================================
        # 15. Correlated BTC/ETH exposure
        # ========================================================

        correlated_exposure = self._correlated_exposure(
            portfolio_state,
            market_state,
        )

        if not math.isfinite(correlated_exposure):
            return self._reject(
                "INVALID_CORRELATED_EXPOSURE"
            )

        if order_request.symbol in {"BTCUSDT", "ETHUSDT"}:
            correlated_exposure += proposed_notional

        correlated_pct = (
            correlated_exposure
            / portfolio_state.equity
        )

        if (
            correlated_pct
            > self.correlated_exposure_limit_pct
        ):
            return self._reject(
                "CORRELATED_EXPOSURE_LIMIT"
            )

        # ========================================================
        # 16. Final quantity validation
        # ========================================================

        if qty <= 0 or not math.isfinite(qty):
            return self._reject("INVALID_QUANTITY")

        logger.debug(
            "Risk approved: %s %s qty=%s",
            order_request.symbol,
            action,
            qty,
        )

        return RiskDecision(
            approved=True,
            reason="Approved",
            adjusted_quantity=qty,
            max_allowed_quantity=qty,
            risk_flags=[],
        )