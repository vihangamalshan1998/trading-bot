import math
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class PerformanceSummary:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    gross_pnl: float
    total_fees: float
    net_pnl: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    net_expectancy: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    skip_reasons: Dict[str, int]

class PerformanceReporter:
    """
    Phase 10: Institutional Performance Metrics & Expectancy Analytics.
    Separates Gross vs. Net outcomes, calculates real Profit Factor, Net Expectancy,
    Sharpe/Sortino ratios, and aggregates reason codes for skipped trades.
    """
    def __init__(self):
        self.skip_reasons: Dict[str, int] = {}

    def record_skip(self, reason: str):
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def calculate_summary(self, trades: List[Dict[str, Any]]) -> PerformanceSummary:
        if not trades:
            return PerformanceSummary(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate_pct=0.0,
                gross_pnl=0.0, total_fees=0.0, net_pnl=0.0, profit_factor=0.0,
                avg_win=0.0, avg_loss=0.0, net_expectancy=0.0, max_drawdown_pct=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, skip_reasons=dict(self.skip_reasons)
            )

        net_pnls = []
        gross_pnls = []
        fees = []

        wins = []
        losses = []

        cumulative_net = 0.0
        peak_net = 0.0
        max_dd = 0.0

        for t in trades:
            r_pnl = float(t.get("realized_pnl", 0.0))
            fee = abs(float(t.get("fees_paid", t.get("commission", 0.0))))
            net = r_pnl - fee
            
            gross_pnls.append(r_pnl)
            fees.append(fee)
            net_pnls.append(net)

            if net > 0:
                wins.append(net)
            elif net < 0:
                losses.append(abs(net))

            cumulative_net += net
            if cumulative_net > peak_net:
                peak_net = cumulative_net
            dd = peak_net - cumulative_net
            if dd > max_dd:
                max_dd = dd

        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        total_gross = sum(gross_pnls)
        total_fee = sum(fees)
        total_net = sum(net_pnls)

        total_win_val = sum(wins)
        total_loss_val = sum(losses)
        profit_factor = (total_win_val / total_loss_val) if total_loss_val > 0 else (99.0 if total_win_val > 0 else 0.0)

        avg_win = (total_win_val / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (total_loss_val / losing_trades) if losing_trades > 0 else 0.0

        p_win = winning_trades / total_trades if total_trades > 0 else 0.0
        p_loss = losing_trades / total_trades if total_trades > 0 else 0.0
        net_expectancy = (p_win * avg_win) - (p_loss * avg_loss)

        # Sharpe and Sortino
        if len(net_pnls) > 1:
            mean_ret = np.mean(net_pnls)
            std_ret = np.std(net_pnls)
            sharpe = (mean_ret / (std_ret + 1e-8)) * math.sqrt(len(net_pnls))
            downside_pnls = [x for x in net_pnls if x < 0]
            downside_std = np.std(downside_pnls) if downside_pnls else 1e-8
            sortino = (mean_ret / (downside_std + 1e-8)) * math.sqrt(len(net_pnls))
        else:
            sharpe = 0.0
            sortino = 0.0

        return PerformanceSummary(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            gross_pnl=total_gross,
            total_fees=total_fee,
            net_pnl=total_net,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            net_expectancy=net_expectancy,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            skip_reasons=dict(self.skip_reasons)
        )
