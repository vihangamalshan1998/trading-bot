"""
Replay-vs-live check: rebuilds the moments of real live records with the history replay and compares
every input slot.

  On the VPS (exports the newest frame of each live record since a time):
    venv/bin/python -m apps.replay.validate export --since 1792648800 --out /tmp/live_frames.json
  On the training machine (after copying the file):
    python -m apps.replay.validate compare --live live_frames.json

Per slot it reports the correlation between live and replayed values over all records and the median
absolute gap in units of the live spread (MAD / live std). Slots the replay cannot rebuild
(apps/replay/layout.py REPLAY_MISSING_SLOTS) are listed separately.
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

LAGS = (0, 60, 150)  # frames back from the record time compared (0 = the newest frame)


def export(since: int, out: str, limit: int):
    from sqlalchemy import create_engine, text
    from core.config.settings import settings
    eng = create_engine(settings.database_url)
    rows = []
    with eng.connect() as c:
        res = c.execute(text("SELECT symbol, timestamp, market_state FROM experiences WHERE timestamp >= :s "
                             "AND action = 'HOLD' ORDER BY id LIMIT :n"), {"s": since, "n": limit})
        for sym, ts, ms in res:
            frames = json.loads(ms) if isinstance(ms, str) else ms
            if not frames or len(frames) < 300:
                continue
            rows.append({"symbol": sym, "timestamp": int(ts),
                         "frames": {str(l): frames[-1 - l] for l in LAGS}})
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"exported {len(rows)} live records since {since} -> {out}")


def compare(live_path: str, workers: int):
    from apps.replay.symbol_replay import build_day, FRAME_SECONDS
    from apps.replay.assemble import day_frames, fear_greed_by_day
    from apps.replay.layout import REPLAY_MISSING_SLOTS, REPLAY_APPROX_SLOTS
    from core.schemas.state_layout import SLOTS, ACTIVE

    with open(live_path) as f:
        live = json.load(f)
    if not live:
        print("no live records")
        return
    symbols = sorted({r["symbol"] for r in live})
    days = sorted({datetime.fromtimestamp(r["timestamp"], timezone.utc).date() for r in live})
    print(f"{len(live)} live records, {len(symbols)} coins, days {days}")

    from concurrent.futures import ProcessPoolExecutor
    from apps.replay import archive
    from apps.replay.symbol_replay import SymbolDayReplay
    from core.config.settings import settings
    # every coin the bot ranks against (excluded coins too), not only the coins with records
    symbols = sorted(set(symbols) | set(settings.symbol_universe))
    jobs = [(s, d) for d in days for s in symbols]
    # Fetch once, one request at a time (Binance REST weight limit), before the parallel replay
    for s, d in jobs:
        SymbolDayReplay(s, d).load()
        print(f"  data ready {s} {d}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (s, d), path in zip(jobs, pool.map(_build, jobs)):
            print(f"  pass 1 {s} {d}: {'ok' if path else 'NO DATA'}")

    fg = fear_greed_by_day()
    replay = {d: day_frames(symbols, d, fg) for d in days}

    pairs = defaultdict(list)  # lag -> list of (live_frame, replay_frame)
    for r in live:
        d = datetime.fromtimestamp(r["timestamp"], timezone.utc).date()
        rs = replay.get(d, {}).get(r["symbol"])
        if rs is None:
            continue
        k = int(np.searchsorted(rs["t"], r["timestamp"], side="right")) - 1
        for lag_s, frame in r["frames"].items():
            lag = int(lag_s)
            # live frames are ~FRAME_SECONDS apart, so `lag` frames back is roughly the same time in the replay grid
            kk = k - lag
            if 0 <= kk < len(rs["t"]) and rs["valid"][kk]:
                pairs[lag].append((np.asarray(frame, dtype=np.float64), rs["state"][kk].astype(np.float64)))

    for lag in sorted(pairs):
        if not pairs[lag]:
            continue
        L = np.stack([p[0] for p in pairs[lag]])
        R = np.stack([p[1] for p in pairs[lag]])
        print(f"\n=== {len(L)} record frames, {lag} frames before the record time ===")
        print(f"{'slot':>4} {'name':<26} {'kind':<7} {'corr':>6} {'gap/std':>8} {'live mean':>11} {'replay mean':>11} {'live std':>10}")
        bad = []
        for s in SLOTS:
            if s.status != ACTIVE or s.index < 20:
                continue
            kind = "MISSING" if s.index in REPLAY_MISSING_SLOTS else ("approx" if s.index in REPLAY_APPROX_SLOTS else "exact")
            l, rr = L[:, s.index], R[:, s.index]
            sd = float(np.std(l))
            corr = float(np.corrcoef(l, rr)[0, 1]) if sd > 0 and np.std(rr) > 0 else float("nan")
            gap = float(np.median(np.abs(l - rr)) / sd) if sd > 0 else float(np.median(np.abs(l - rr)))
            print(f"{s.index:>4} {s.name:<26} {kind:<7} {corr:>6.3f} {gap:>8.3f} {np.mean(l):>11.4g} {np.mean(rr):>11.4g} {sd:>10.4g}")
            if kind != "MISSING" and (not np.isfinite(corr) or corr < 0.9):
                bad.append((s.index, s.name, kind, corr))
        print(f"\nslots below 0.9 correlation (excluding MISSING): {len(bad)}")
        for b in bad:
            print("   ", b)


def _build(job):
    from apps.replay.symbol_replay import build_day
    s, d = job
    try:
        return build_day(s, d)
    except Exception as e:
        print(f"  pass 1 {s} {d} failed: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--since", type=int, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=5000)
    c = sub.add_parser("compare")
    c.add_argument("--live", required=True)
    c.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.since, a.out, a.limit)
    else:
        compare(a.live, a.workers)
