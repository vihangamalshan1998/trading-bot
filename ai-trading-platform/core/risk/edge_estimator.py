from dataclasses import dataclass
from typing import Optional
from core.config.settings import settings
from core.schemas.state_schema import MarketState

@dataclass
class EdgeResult:
    symbol: str
    side: str
    gross_expected_return_pct: float
    estimated_costs_pct: float
    net_expected_return_pct: float
    entry_fee_pct: float
    exit_fee_pct: float
    half_spread_pct: float
    slippage_pct: float
    funding_pct: float
    uncertainty: float
    horizon_seconds: float
    approved: bool
    reason_code: str

class EdgeEstimator:
    """
    Phase 2: Pre-Trade Net-Edge Estimator.
    Evaluates whether candidate trading opportunities possess positive mathematical expectancy
    strictly after transaction commissions, bid-ask spread, estimated slippage, and funding costs.
    """
    def __init__(
        self,
        min_net_edge_pct: float = 0.08,  # Default minimum net edge after all friction
        default_horizon_seconds: float = 300.0,
    ):
        self.min_net_edge_pct = min_net_edge_pct
        self.default_horizon_seconds = default_horizon_seconds

    def estimate(
        self,
        symbol: str,
        side: str,
        predicted_move_pct: float,
        confidence: float,
        market: MarketState,
        is_maker: bool = True,
        order_size_notional: float = 400.0,
        regime: str = "NEUTRAL",
    ) -> EdgeResult:
        """
        Computes the complete cost structure and net edge for an order.
        """
        # 1. Expected Gross Return
        gross_return = float(predicted_move_pct)
        uncertainty = max(0.0, 1.0 - confidence)

        if gross_return <= 0.0:
            return EdgeResult(
                symbol=symbol,
                side=side,
                gross_expected_return_pct=gross_return,
                estimated_costs_pct=0.0,
                net_expected_return_pct=gross_return,
                entry_fee_pct=0.0,
                exit_fee_pct=0.0,
                half_spread_pct=0.0,
                slippage_pct=0.0,
                funding_pct=0.0,
                uncertainty=uncertainty,
                horizon_seconds=self.default_horizon_seconds,
                approved=False,
                reason_code="NEGATIVE_PREDICTED_RETURN",
            )

        # 2. Commissions (Maker 0.02% vs Taker 0.05%)
        entry_fee = (settings.maker_fee_pct if is_maker else settings.taker_fee_pct) * 100.0
        # Assume conservative taker exit on stop/target
        exit_fee = settings.taker_fee_pct * 100.0

        # 3. Spread Cost (Half-spread for crossing, 0 for pure resting maker)
        mid_price = market.mid_price if market.mid_price > 0 else 1.0
        b_ask = getattr(market, "best_ask", getattr(market, "ask", mid_price))
        b_bid = getattr(market, "best_bid", getattr(market, "bid", mid_price))
        spread = max(0.0, b_ask - b_bid)
        half_spread_pct = (spread / (2.0 * mid_price)) * 100.0 if not is_maker else 0.0

        # 4. Estimated Slippage & Market Impact
        # Proportional to spread and order size vs top of book depth
        bid_q = getattr(market, "bid_qty", 50.0)
        ask_q = getattr(market, "ask_qty", 50.0)
        top_depth_qty = max(1.0, bid_q + ask_q)
        top_depth_notional = top_depth_qty * mid_price
        depth_ratio = min(1.0, order_size_notional / max(top_depth_notional, 1.0))
        slippage_pct = (half_spread_pct * 0.5) + (depth_ratio * 0.02)

        # 5. Funding Attribution (if held over 8h window)
        funding_rate = getattr(market, "funding_rate", 0.0)
        # Prorate funding cost over estimated holding horizon
        horizon_hours = self.default_horizon_seconds / 3600.0
        funding_pct = abs(funding_rate * (horizon_hours / 8.0) * 100.0)

        # 6. Total Friction Costs
        total_costs = entry_fee + exit_fee + half_spread_pct + slippage_pct + funding_pct

        # 7. Net Expected Return
        net_return = gross_return - total_costs

        # 8. Regime-calibrated required hurdle
        required_hurdle = self.min_net_edge_pct
        if regime == "REGIME_CHOP":
            required_hurdle += 0.05  # Demand higher edge in chop
        elif regime == "REGIME_SHOCK":
            required_hurdle += 0.15  # Extreme caution in volatility shocks

        approved = net_return >= required_hurdle

        if not approved:
            if gross_return < total_costs:
                reason = f"NEGATIVE_NET_EDGE: Gross ({gross_return:.3f}%) < Costs ({total_costs:.3f}%)"
            else:
                reason = f"SUB_HURDLE_EDGE: Net ({net_return:.3f}%) < Required ({required_hurdle:.3f}%)"
        else:
            reason = "EDGE_VERIFIED"

        return EdgeResult(
            symbol=symbol,
            side=side,
            gross_expected_return_pct=gross_return,
            estimated_costs_pct=total_costs,
            net_expected_return_pct=net_return,
            entry_fee_pct=entry_fee,
            exit_fee_pct=exit_fee,
            half_spread_pct=half_spread_pct,
            slippage_pct=slippage_pct,
            funding_pct=funding_pct,
            uncertainty=uncertainty,
            horizon_seconds=self.default_horizon_seconds,
            approved=approved,
            reason_code=reason,
        )
