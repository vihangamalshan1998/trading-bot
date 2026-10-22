"""
Pass 2 of the history replay: all coins' pass-1 frames -> full 200-slot frames (built by the live bot's
own builders, core/schemas/state_builder.py) -> training records exactly like the bot's market snapshots.

The cross-section block ranks each coin against all coins at the same frame, so every coin of a day is
assembled together on the shared frame grid.
"""
import collections
import json
import os
import time
import types
import urllib.request
import uuid
import zlib
from datetime import date, datetime, timezone
from typing import Dict, Iterator, List, Optional

import numpy as np

from core.schemas import state_builder
from apps.replay import archive
from apps.replay.symbol_replay import load_day

SEQ_LEN = 300
RECORD_EVERY_SECONDS = 1800   # one record per coin every 30 min (records overlap 26 min of frames anyway)
MODEL_VERSION = "replay_v1"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"

# Flat account (slots 0-19 are never fed to the brain; kept like a live flat-account frame)
FLAT_ACCOUNT = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 10


def fear_greed_by_day() -> Dict[str, float]:
    """{'YYYY-MM-DD': value 0-100} from alternative.me (daily, published at 00:00 UTC). Cached for 12h."""
    path = os.path.join(archive.CACHE_DIR, "fear_greed.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 12 * 3600:
        with open(path) as f:
            return json.load(f)
    with urllib.request.urlopen(urllib.request.Request(FEAR_GREED_URL, headers={"User-Agent": "Mozilla/5.0"}),
                                timeout=30) as resp:
        rows = json.load(resp)["data"]
    out = {datetime.fromtimestamp(int(r["timestamp"]), timezone.utc).date().isoformat(): float(r["value"]) for r in rows}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def macro_at(day: date, fear_greed: Dict[str, float]):
    """MacroState-like values for replayed frames. Only Fear & Greed has history; the rest are
    placeholders (REPLAY_MISSING_SLOTS) at the live feed's neutral defaults."""
    return types.SimpleNamespace(
        sentiment_score=0.0, volatility_expectation=0.5, regime=0.0, sp500_momentum=0.0, dxy_momentum=0.0,
        vix_momentum=0.0, gold_momentum=0.0, treasury_yield_momentum=0.0, ndx_momentum=0.0,
        defi_tvl_momentum=0.0, fear_greed_index=fear_greed.get(day.isoformat(), 50.0) / 100.0)


def day_frames(symbols: List[str], day: date, fear_greed: Dict[str, float]) -> Dict[str, Dict[str, np.ndarray]]:
    """{symbol: {'t': (n,), 'state': (n, 200) float32, 'valid': (n,) bool}} for one UTC day."""
    p1 = {s: load_day(s, day) for s in symbols}
    p1 = {s: d for s, d in p1.items() if d is not None}
    if not p1:
        return {}
    grid = next(iter(p1.values()))["t"]
    macro = macro_at(day, fear_greed)
    coins = list(p1)
    feats = np.stack([p1[s]["features"] for s in coins])            # (c, n, 90)
    valid = np.stack([p1[s]["valid"] for s in coins])               # (c, n)
    cross = state_builder.cross_section_block_batch(feats, valid)   # (c, n, 10)
    btc = p1["BTCUSDT"]["features"] if "BTCUSDT" in p1 else None
    if btc is not None:
        # BTC momentum (and the beta to BTC) is 0 in frames where BTC has no data, as in the per-frame build
        btc = np.where(p1["BTCUSDT"]["valid"][:, None], btc, 0.0)
    n = len(grid)
    flat = np.tile(np.asarray(FLAT_ACCOUNT, dtype=np.float64), (n, 1))
    glob = np.tile(np.asarray(state_builder.global_block({}), dtype=np.float64), (n, 1))
    out = {}
    for ci, s in enumerate(coins):
        obs = np.concatenate([
            flat,
            state_builder.market_block_batch(feats[ci]),
            state_builder.macro_block_batch(grid, feats[ci], btc, macro),
            state_builder.whale_statarb_block_batch(p1[s]["whale"][:, 0], p1[s]["whale"][:, 1]),
            cross[ci],
            glob,
            np.zeros((n, 30)),
        ], axis=1)
        out[s] = {"t": grid, "state": state_builder.finalize_batch(obs), "valid": valid[ci].copy()}
    return out


def _record_offset(symbol: str) -> int:
    """Per-coin phase of the record schedule (live coins also emit at unrelated moments)."""
    return zlib.crc32(symbol.encode()) % RECORD_EVERY_SECONDS


def records(symbols: List[str], days: List[date], log=print,
            record_symbols: Optional[List[str]] = None) -> Iterator[dict]:
    """
    Yields experience rows (dicts for the experiences schema), oldest first, across consecutive days.
    `symbols` are all coins the live bot holds market states for (they all count in the cross-section
    ranks); records are written only for `record_symbols` (default: all), like the live bot skips
    excluded_trading_symbols but still ranks against them.
    """
    record_set = set(record_symbols or symbols)
    fear_greed = fear_greed_by_day()
    history: Dict[str, collections.deque] = {s: collections.deque(maxlen=SEQ_LEN) for s in symbols}
    last_t: Dict[str, float] = {}
    next_due: Dict[str, float] = {}
    for day in days:
        t0 = time.time()
        frames = day_frames(symbols, day, fear_greed)
        n = 0
        for s, d in frames.items():
            if s not in record_set:
                continue
            hist = history[s]
            for k in range(len(d["t"])):
                t = float(d["t"][k])
                if not d["valid"][k] or (s in last_t and t - last_t[s] > 60):
                    hist.clear()  # a gap in the data breaks the sequence
                if not d["valid"][k]:
                    continue
                last_t[s] = t
                hist.append(d["state"][k])
                if len(hist) < SEQ_LEN:
                    continue
                if s not in next_due:
                    off = _record_offset(s)
                    next_due[s] = (t // RECORD_EVERY_SECONDS) * RECORD_EVERY_SECONDS + off
                    if next_due[s] < t:
                        next_due[s] += RECORD_EVERY_SECONDS
                if t < next_due[s]:
                    continue
                while next_due[s] <= t:
                    next_due[s] += RECORD_EVERY_SECONDS
                n += 1
                yield {
                    "experience_id": f"replay-{s}-{int(t)}-{uuid.uuid4().hex[:8]}",
                    "timestamp": int(t),
                    "symbol": s,
                    "market_state": [state_builder.round_frame(f) for f in hist],
                    "macro_state": None,
                    "portfolio_state": {"source": "history_replay"},
                    "derivatives_state": None,
                    "position_before": 0.0, "position_after": 0.0, "entry_price": 0.0, "exit_price": None,
                    "leverage": 0, "margin": 0.0, "fees": 0.0, "holding_duration": 0,
                    "action": "HOLD", "action_probability": 0.5, "confidence": 0.5,
                    "model_version": MODEL_VERSION,
                    "reward": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                }
        if log:
            log(f"assemble: {day} -> {n} records from {len(frames)} coins ({time.time() - t0:.0f}s)")
