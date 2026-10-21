"""
Historical backtester for the live exit policy and simple baseline strategies.

Answers the first question before any AI work: on months of real 1-minute data, does ANY simple
entry rule make money after fees when managed by the live exits (core/risk/exit_policy.py)?
A random-entry benchmark shows what fees + exits cost when there is no edge at all.

    python -m scripts.evaluation.backtest --days 180
    python -m scripts.evaluation.backtest --days 365 --symbols BTCUSDT,ETHUSDT,SOLUSDT --notional 8

Data: Binance USD-M mainnet 1m klines (public), cached under data/historical/klines_1m/.
Exits are path-dependent on each bar's high/low; when one bar touches both the stop and a target,
the stop is assumed to hit first (conservative). Costs: exit_policy.ROUND_TRIP_COST per round trip.
--notional is position notional as a multiple of equity (current live sizing: 20x leverage with
25-40% margin = 5-8x). Results are reported per trade (% of notional) and on equity.

The ~35 live entry gates are NOT simulated: they depend on order-book / whale / statarb streams
that have no history. Each gate should earn its place once those inputs are recorded.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from core.config.settings import settings
from core.risk import exit_policy as xp

CACHE_DIR = os.path.join("data", "historical", "klines_1m")
BASE_URL = "https://fapi.binance.com"


# ---------------------------------------------------------------- data
def load_klines(symbol: str, days: int) -> np.ndarray:
    """Returns rows [open_time_ms, open, high, low, close, volume, taker_buy_volume] for the last `days`."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}.npy")
    end_ms = (int(time.time()) // 60) * 60_000
    start_ms = end_ms - days * 86_400_000
    data = np.load(path) if os.path.exists(path) else np.empty((0, 7))
    have_from = int(data[0, 0]) if len(data) else end_ms
    have_to = int(data[-1, 0]) + 60_000 if len(data) else end_ms
    chunks = [data]
    if start_ms < have_from:
        chunks.insert(0, _download(symbol, start_ms, have_from))
    if have_to < end_ms and len(data):
        chunks.append(_download(symbol, have_to, end_ms))
    data = np.concatenate([c for c in chunks if len(c)]) if any(len(c) for c in chunks) else np.empty((0, 7))
    if len(data):
        _, idx = np.unique(data[:, 0], return_index=True)
        data = data[idx]
        np.save(path, data)
    return data[data[:, 0] >= start_ms]


def _download(symbol: str, start_ms: int, end_ms: int) -> np.ndarray:
    rows, cursor = [], start_ms
    while cursor < end_ms:
        q = urllib.parse.urlencode({"symbol": symbol, "interval": "1m", "startTime": cursor, "endTime": end_ms - 1, "limit": 1500})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"{BASE_URL}/fapi/v1/klines?{q}", timeout=30) as r:
                    batch = json.load(r)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"{symbol}: download retry {attempt + 1} after {e!r}")
                time.sleep(2 ** attempt)
        if not batch:
            break
        rows.extend([[float(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9])] for k in batch])
        cursor = int(batch[-1][0]) + 60_000
        time.sleep(0.12)
    return np.array(rows) if rows else np.empty((0, 7))


def resample(bars: np.ndarray, n: int) -> np.ndarray:
    """1m rows -> n-minute OHLC rows [t, o, h, l, c] aligned to n-minute boundaries; last row is the closing index."""
    t = bars[:, 0] // (n * 60_000)
    starts = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    ends = np.r_[starts[1:], len(bars)] - 1
    o = bars[starts, 1]
    h = np.maximum.reduceat(bars[:, 2], starts)
    lo = np.minimum.reduceat(bars[:, 3], starts)
    c = bars[ends, 4]
    return np.column_stack([ends, o, h, lo, c])


def ema(x: np.ndarray, p: int) -> np.ndarray:
    a, out = 2.0 / (p + 1), np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(x: np.ndarray, p: int = 14) -> np.ndarray:
    d = np.diff(x, prepend=x[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = ema(up, 2 * p - 1), ema(dn, 2 * p - 1)  # Wilder smoothing
    return 100 - 100 / (1 + au / np.maximum(ad, 1e-12))


def atr_frac(hb: np.ndarray, p: int = 14) -> np.ndarray:
    """1h ATR as a fraction of price, per hourly bar (same definition the live FeatureEngine publishes)."""
    h, lo, c = hb[:, 2], hb[:, 3], hb[:, 4]
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - lo, np.maximum(abs(h - prev), abs(lo - prev)))
    out = np.convolve(tr, np.ones(p) / p, mode="full")[: len(tr)]
    out[:p] = np.cumsum(tr[:p]) / np.arange(1, p + 1)
    return out / c


# ---------------------------------------------------------------- strategies
# A strategy maps hourly bars -> desired direction per hourly bar (+1 long, -1 short, 0 none).
def s_trend(hb):
    c = hb[:, 4]; f, s = ema(c, 20), ema(c, 50)
    return np.where((f > s) & (c > f), 1, np.where((f < s) & (c < f), -1, 0))


def s_breakout(hb, n=20):
    c, h, lo = hb[:, 4], hb[:, 2], hb[:, 3]
    sig = np.zeros(len(c), dtype=int)
    for i in range(n, len(c)):
        if c[i] > h[i - n:i].max():
            sig[i] = 1
        elif c[i] < lo[i - n:i].min():
            sig[i] = -1
    return sig


def s_mean_reversion(hb):
    r = rsi(hb[:, 4], 14)
    return np.where(r < 25, 1, np.where(r > 75, -1, 0))


def s_random(hb, seed=7, rate=0.15):
    rng = np.random.default_rng(seed)
    x = rng.random(len(hb))
    return np.where(x < rate / 2, 1, np.where(x < rate, -1, 0))


STRATEGIES: Dict[str, Callable] = {
    "trend_ema20_50": s_trend,
    "breakout_20h": s_breakout,
    "mean_reversion_rsi": s_mean_reversion,
    "random_benchmark": s_random,
}


# ---------------------------------------------------------------- simulation
@dataclass
class Result:
    trades: List[float] = field(default_factory=list)   # net % of notional per trade
    holds: List[float] = field(default_factory=list)    # minutes
    reasons: Dict[str, int] = field(default_factory=dict)


def simulate(bars: np.ndarray, signal_fn: Callable) -> Result:
    hb = resample(bars, 60)
    sig = signal_fn(hb)
    vol = atr_frac(hb)
    res = Result()
    i_bar = 0
    for k in range(60, len(hb) - 1):  # warm-up: 60 hours of indicators
        entry_idx = int(hb[k, 0]) + 1  # first 1m bar after the hourly close
        if sig[k] == 0 or entry_idx <= i_bar or entry_idx >= len(bars):
            continue
        d = int(sig[k])
        entry = bars[entry_idx, 1]
        lv = xp.exit_levels(float(vol[k]))
        tp1_done, peak, banked = False, 0.0, 0.0
        j = entry_idx
        exit_reason, exit_move = "END OF DATA", 0.0
        while j < len(bars):
            hi = (bars[j, 2] - entry) / entry * d if d > 0 else (entry - bars[j, 3]) / entry
            lo = (bars[j, 3] - entry) / entry * d if d > 0 else (entry - bars[j, 2]) / entry
            close_move = (bars[j, 4] - entry) / entry * d
            held = (j - entry_idx) * 60.0
            # adverse extreme first (conservative), then favourable extreme, then the close
            done = False
            for move in (lo, hi, close_move):
                peak = max(peak, move)
                dec = xp.evaluate_exit(move, peak, tp1_done, held, lv)
                if dec.close_half and not tp1_done:
                    banked, tp1_done = lv.tp1, True
                    continue
                if dec.close_all:
                    exit_reason = dec.reason
                    if dec.reason == "HARD STOP":
                        exit_move = -lv.stop
                    elif dec.reason == "HARD TAKE-PROFIT":
                        exit_move = lv.hard_tp
                    elif dec.reason in ("TRAILING STOP", "FREE-ROLL BREAKEVEN"):
                        exit_move = max(lv.breakeven, peak - lv.trail_gap)
                    else:
                        exit_move = move
                    done = True
                    break
            if done:
                break
            j += 1
        if exit_reason == "END OF DATA":
            exit_move = (bars[-1, 4] - entry) / entry * d
        gross = 0.5 * banked + 0.5 * exit_move if tp1_done else exit_move
        res.trades.append((gross - xp.ROUND_TRIP_COST) * 100.0)
        res.holds.append((j - entry_idx))
        res.reasons[exit_reason] = res.reasons.get(exit_reason, 0) + 1
        i_bar = j
    return res


def summarize(name: str, trades: List[float], holds: List[float], notional: float, days: int) -> Dict:
    t = np.array(trades) if trades else np.zeros(0)
    if len(t) == 0:
        return {"strategy": name, "trades": 0}
    equity = np.cumprod(1 + t / 100.0 * notional)
    peak = np.maximum.accumulate(equity)
    wins, losses = t[t > 0], t[t <= 0]
    return {
        "strategy": name,
        "trades": int(len(t)),
        "trades_per_day": round(len(t) / days, 2),
        "win_rate": round(float((t > 0).mean()), 3),
        "avg_win_pct": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss_pct": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "net_pct_per_trade": round(float(t.mean()), 4),
        "equity_return_pct": round(float((equity[-1] - 1) * 100), 1),
        "max_drawdown_pct": round(float(((peak - equity) / peak).max() * 100), 1),
        "avg_hold_min": round(float(np.mean(holds)), 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--symbols", default=",".join(s for s in settings.symbol_universe if s not in settings.excluded_trading_symbols))
    ap.add_argument("--notional", type=float, default=settings.max_leverage * settings.major_max_margin_pct,
                    help="position notional as a multiple of equity (live default: leverage x max margin)")
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    per_strategy = {name: Result() for name in STRATEGIES}
    for sym in symbols:
        bars = load_klines(sym, args.days)
        if len(bars) < 60 * 24 * 7:
            print(f"{sym}: not enough history ({len(bars)} bars), skipped")
            continue
        for name, fn in STRATEGIES.items():
            r = simulate(bars, fn)
            per_strategy[name].trades += r.trades
            per_strategy[name].holds += r.holds
            for k, v in r.reasons.items():
                per_strategy[name].reasons[k] = per_strategy[name].reasons.get(k, 0) + v
        print(f"{sym}: {len(bars)} bars simulated")

    print(f"\n=== {args.days} days, {len(symbols)} symbols, round-trip cost {xp.ROUND_TRIP_COST*100:.2f}%, "
          f"notional {args.notional:.1f}x equity (equity figures assume trades in sequence) ===")
    rows = [summarize(n, r.trades, r.holds, args.notional, args.days) for n, r in per_strategy.items()]
    for row in rows:
        print(json.dumps(row))
    for n, r in per_strategy.items():
        print(f"{n} exits: {r.reasons}")
    edge = [row for row in rows if row.get("trades", 0) >= 50 and row["net_pct_per_trade"] > 0 and row["strategy"] != "random_benchmark"]
    print("\nVERDICT:", "strategies with a positive net edge after costs: " + ", ".join(r["strategy"] for r in edge)
          if edge else "no baseline strategy beats costs with these exits - the AI needs better inputs or a different horizon.")


if __name__ == "__main__":
    main()
