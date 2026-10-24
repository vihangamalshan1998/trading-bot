"""
Shared exit policy, expressed in PRICE moves (fractions), independent of leverage.

Used by the live bot (apps/trading_bot/main.py) and by the trainer (apps/trainer/ppo.py), so the
AI is scored on the same exits it will trade with.

Why it changed: the previous rules (at 20x) stopped out at -0.75% price, banked half the position
at +0.25% and closed the rest at a "breakeven" of +0.0125% that did not even cover fees. An average
win was ~+0.03% vs an average loss of ~-0.85%, which needs a ~96% win rate just to break even.

New rules:
  - Stop-loss: 1x recent volatility (1h ATR), 0.35% to 2%. Until 24 Oct 2026 it was capped at 0.75%,
    inside the normal 1h swing of most coins (history and live: ~47% of trades hit the full stop).
    The bot sizes each position so a stop loses no more dollars than a 0.75% stop did (risk_size_mult).
  - Breakeven guard: once a trade has been +BREAKEVEN_TRIGGER in profit, it never closes below a
    breakeven that covers costs (before TP1 too).
  - First take-profit (50% of the position): at least 1.5x the stop AND at least stop + 2x costs.
  - Remaining 50%: stop moves to a breakeven that really covers round-trip costs, then trails one
    stop-distance behind the best price reached. Hard take-profit at 3x the stop.
  - Stale positions (no progress for 4h) are closed to free margin; every position closes at 4h (MAX HOLD).
"""
from dataclasses import dataclass

# Round-trip cost in price terms: maker entry 0.02% + taker exit 0.05% + slippage allowance
ROUND_TRIP_COST = 0.0010

STOP_VOL_MULT = 1.0
MIN_STOP = 0.0035
MAX_STOP = 0.020           # 1h ATR up to 2% (was 0.75% until 24 Oct 2026)
RISK_REF_STOP = 0.0075     # positions are sized so a stop loses at most what a 0.75% stop did
BREAKEVEN_TRIGGER = 0.010  # once +1.0% in profit, the stop moves to breakeven (entry + costs)
TP1_RR = 1.5               # first take-profit at >= 1.5x the stop
HARD_TP_RR = 3.0
DEFAULT_VOLATILITY = 0.0075  # used when the collector has no volatility reading yet
STALE_SECONDS = 4 * 3600
STALE_BAND = ROUND_TRIP_COST  # "no progress" = within +/- costs
# Hard time limit: the critic forecasts at most 4h ahead, and the trainer grades every decision by
# replaying it for at most 4h, so live positions close at 4h too (training == live).
MAX_HOLD_SECONDS = 4 * 3600

# Weights of the critic's 5m / 1h / 4h forecasts that define the expected move of a trade under
# these exits (1-4h swing). Shared by the trainer's reward and the bot's entry hurdle.
HORIZON_WEIGHTS = (0.1, 0.6, 0.3)


def expected_move_pct(critic_values) -> float:
    """Blended expected price move in % from the critic's [5m, 1h, 4h] outputs."""
    vals = [float(v) for v in critic_values][:3] if critic_values is not None else []
    if not vals:
        return 0.0
    while len(vals) < 3:
        vals.append(vals[-1])
    return sum(w * v for w, v in zip(HORIZON_WEIGHTS, vals))


@dataclass(frozen=True)
class ExitLevels:
    stop: float          # close everything at -stop
    tp1: float           # bank 50% at +tp1
    hard_tp: float       # close everything at +hard_tp
    breakeven: float     # after TP1, never let the runner close below +breakeven
    trail_gap: float     # after TP1, runner trails trail_gap behind the peak


def exit_levels(volatility: float) -> ExitLevels:
    """volatility: recent 1h ATR as a fraction of price (e.g. 0.006 = 0.6%)."""
    vol = volatility if volatility and volatility > 0 else DEFAULT_VOLATILITY
    stop = min(MAX_STOP, max(MIN_STOP, vol * STOP_VOL_MULT))
    tp1 = max(stop * TP1_RR, stop + 2.0 * ROUND_TRIP_COST)
    return ExitLevels(
        stop=stop,
        tp1=tp1,
        hard_tp=stop * HARD_TP_RR,
        breakeven=ROUND_TRIP_COST,
        trail_gap=stop,
    )


def risk_size_mult(levels: ExitLevels) -> float:
    """Position-size factor so a stop loses at most what a RISK_REF_STOP stop did (1.0 = today's size).
    Never above 1.0: calm coins with a tight stop keep today's size rather than growing."""
    return min(1.0, RISK_REF_STOP / max(levels.stop, 1e-9))


def max_notional_for_risk(equity: float, levels: ExitLevels, risk_pct: float) -> float:
    """Largest position (notional) whose loss at the stop, costs included, is at most risk_pct of equity."""
    return max(0.0, equity) * risk_pct / (levels.stop + ROUND_TRIP_COST)


def ai_profit_close_min(levels: ExitLevels) -> float:
    """Smallest favourable move at which the AI may close a winner: one stop-distance, but never more than
    RISK_REF_STOP (0.75%). Without the cap the wider stops of 24 Oct 2026 would have blocked AI profit closes
    below ~1.2% (the +0.96% LTC close of that morning, +$146, needed only 0.75%)."""
    return min(levels.stop, RISK_REF_STOP)


def backup_stop_price(entry: float, is_long: bool, levels: ExitLevels, mult: float, peak_move: float, tp1_done: bool) -> float:
    """
    Price of the exchange-side backup stop (fires only when the bot is down). Normally mult x the trade's stop
    away from entry; once the safe line is drawn (peak >= BREAKEVEN_TRIGGER, or TP1 banked) it moves to the
    entry price itself, so a winner cannot turn into a loss on the exchange either. It sits just below the
    bot's own breakeven exit (+costs), so with the bot running the bot always closes first.
    """
    if tp1_done or peak_move >= BREAKEVEN_TRIGGER:
        return entry
    dist = levels.stop * mult
    return entry * (1 - dist) if is_long else entry * (1 + dist)


@dataclass(frozen=True)
class ExitDecision:
    close_all: bool = False
    close_half: bool = False
    reason: str = ""


def evaluate_exit(move: float, peak_move: float, tp1_done: bool, held_seconds: float, levels: ExitLevels) -> ExitDecision:
    """
    move:       current favourable price move of the position (fraction, + = in profit)
    peak_move:  best favourable move reached since entry
    tp1_done:   the 50% first take-profit has already been banked
    """
    if move <= -levels.stop:
        return ExitDecision(close_all=True, reason="HARD STOP")
    if not tp1_done and peak_move >= BREAKEVEN_TRIGGER and move <= levels.breakeven:
        return ExitDecision(close_all=True, reason="BREAKEVEN GUARD")
    if move >= levels.hard_tp:
        return ExitDecision(close_all=True, reason="HARD TAKE-PROFIT")
    if not tp1_done:
        if move >= levels.tp1:
            return ExitDecision(close_half=True, reason="PARTIAL TP 50%")
    else:
        floor = max(levels.breakeven, peak_move - levels.trail_gap)
        if move <= floor:
            reason = "FREE-ROLL BREAKEVEN" if floor <= levels.breakeven else "TRAILING STOP"
            return ExitDecision(close_all=True, reason=reason)
    if held_seconds >= STALE_SECONDS and abs(move) < STALE_BAND:
        return ExitDecision(close_all=True, reason="STALE PRUNE")
    if held_seconds >= MAX_HOLD_SECONDS:
        return ExitDecision(close_all=True, reason="MAX HOLD")
    return ExitDecision()


def volatility_1h_from_ohlc(ohlc) -> float:
    """
    1h ATR(14) as a fraction of price from 1-minute bars (rows: open, high, low, close), oldest first.
    Same definition as FeatureEngine.volatility_1h() so the grader replays trades with the exact
    stop distance the live bot would have used. 0.0 until 3 hourly bars exist.
    """
    import numpy as np
    bars = np.asarray(ohlc, dtype=np.float64)
    usable = (len(bars) // 60) * 60
    if usable < 180:
        return 0.0
    b = bars[len(bars) - usable:].reshape(-1, 60, 4)
    highs, lows, closes = b[:, :, 1].max(axis=1), b[:, :, 2].min(axis=1), b[:, -1, 3]
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = float(np.mean(tr[-14:])) if len(tr) else 0.0
    return atr / max(float(closes[-1]), 1e-12)


def simulate_trade(direction: int, entry: float, bars, levels: ExitLevels, bar_seconds: int = 60) -> float:
    """
    Replays one position opened at `entry` through 1-minute bars (rows: high, low, close) with the
    live exit rules and returns the realized favourable price move (fraction, GROSS of costs),
    averaged over the position (the 50% TP1 slice and the runner).

    Inside a bar the adverse extreme is assumed to come first (conservative: a bar that touches
    both the stop and a target counts as a stop). Positions still open at MAX_HOLD_SECONDS, or when
    the data ends, close at that bar's close.
    """
    if entry <= 0 or direction == 0:
        return 0.0
    banked = 0.0          # move realized by the TP1 half
    tp1_done = False
    peak = 0.0
    move_close = 0.0
    held = 0
    for high, low, close in bars:
        held += bar_seconds
        if direction > 0:
            best, worst, move_close = (high - entry) / entry, (low - entry) / entry, (close - entry) / entry
        else:
            best, worst, move_close = (entry - low) / entry, (entry - high) / entry, (entry - close) / entry
        # 1. adverse side first
        if not tp1_done:
            floor = levels.breakeven if peak >= BREAKEVEN_TRIGGER else -levels.stop
            if worst <= floor:
                return floor
        else:
            floor = max(levels.breakeven, peak - levels.trail_gap)
            if worst <= floor:
                return 0.5 * banked + 0.5 * floor
        # 2. favourable side
        if best >= levels.hard_tp:
            return 0.5 * banked + 0.5 * levels.hard_tp if tp1_done else levels.hard_tp
        if not tp1_done and best >= levels.tp1:
            tp1_done, banked = True, levels.tp1
        peak = max(peak, best)
        # 3. time limit
        if held >= MAX_HOLD_SECONDS:
            break
    if tp1_done:
        return 0.5 * banked + 0.5 * move_close
    return move_close
