"""
Shared exit policy, expressed in PRICE moves (fractions), independent of leverage.

Used by the live bot (apps/trading_bot/main.py) and by the trainer (apps/trainer/ppo.py), so the
AI is scored on the same exits it will trade with.

Why it changed: the previous rules (at 20x) stopped out at -0.75% price, banked half the position
at +0.25% and closed the rest at a "breakeven" of +0.0125% that did not even cover fees. An average
win was ~+0.03% vs an average loss of ~-0.85%, which needs a ~96% win rate just to break even.

New rules:
  - Stop-loss: 1x recent volatility (1h ATR), never wider than the old 0.75% (loss per trade
    cannot grow while leverage / margin sizing stay unchanged).
  - First take-profit (50% of the position): at least 1.5x the stop AND at least stop + 2x costs.
  - Remaining 50%: stop moves to a breakeven that really covers round-trip costs, then trails one
    stop-distance behind the best price reached. Hard take-profit at 3x the stop.
  - Stale positions (no progress for 4h) are closed to free margin.
"""
from dataclasses import dataclass

# Round-trip cost in price terms: maker entry 0.02% + taker exit 0.05% + slippage allowance
ROUND_TRIP_COST = 0.0010

STOP_VOL_MULT = 1.0
MIN_STOP = 0.0035
MAX_STOP = 0.0075          # the previous fixed stop (-15% leveraged at 20x): never exceed it
TP1_RR = 1.5               # first take-profit at >= 1.5x the stop
HARD_TP_RR = 3.0
DEFAULT_VOLATILITY = 0.0075  # used when the collector has no volatility reading yet
STALE_SECONDS = 4 * 3600
STALE_BAND = ROUND_TRIP_COST  # "no progress" = within +/- costs

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
    return ExitDecision()
