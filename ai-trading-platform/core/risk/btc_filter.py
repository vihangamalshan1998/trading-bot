"""
Entry filters against hard 1h moves (whole-day history replay, apps/research/studies/rules_study.py, 24 Oct 2026):
- BTC filter: no new long while Bitcoin fell more than BTC_MOVE_LIMIT over the last hour, no new short while it
  rose more than that. Altcoins follow BTC. With the 30 min loss lock it lifted the live rules on both the
  15 exam days and the 77 training days (C2).
- Falling-knife filter: the same on the coin's own last hour (COIN_MOVE_LIMIT). On top of C2: +11.4 / +13.1 total %
  (exam / training days), profit per trade about doubled, about 20% fewer trades.
Open positions are not touched. Missing or stale data never blocks (fail open).
"""
from typing import Optional

BTC_MOVE_LIMIT = 0.005      # 0.5% BTC move over the last hour
COIN_MOVE_LIMIT = 0.005     # 0.5% move of the coin itself over the last hour
BTC_MOVE_MAX_AGE = 300.0    # seconds: an older reading counts as missing


def move_from_klines(klines) -> Optional[float]:
    """Move over the last hour from 61+ 1m klines (newest last): latest close vs the open 60 bars earlier."""
    if not klines or len(klines) < 61:
        return None
    try:
        start, last = float(klines[-61][1]), float(klines[-1][4])
    except (TypeError, ValueError, IndexError):
        return None
    return last / start - 1.0 if start > 0 else None


def blocks_entry(direction: int, move: Optional[float], limit: float = BTC_MOVE_LIMIT) -> bool:
    """True when a new position in `direction` (+1 long / -1 short) goes against a 1h move beyond `limit`."""
    if move is None or direction == 0:
        return False
    return move * direction <= -limit
