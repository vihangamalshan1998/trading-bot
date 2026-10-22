"""
History replay runner (training machine):

    python -m apps.replay.run --start 2026-08-20 --end 2026-10-20 --workers 6

1. downloads the archive files (cached),
2. pass 1: per coin-day market features (parallel, cached),
3. pass 2: assembles 200-slot frames for all coins and writes records to the `replay_experiences` table
   (same schema as experiences; days already written are skipped),
4. grades them with the live reward calculator (true market returns + path-aware trade outcomes,
   src 'replay'), using the archive's own 1m klines as the price source.

Live records stay in `experiences`; the trainer reads replayed rows as a separate pool.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from typing import List

import numpy as np
from sqlalchemy import MetaData, Table, create_engine, func, insert, select, text

from core.config.settings import settings
from apps.replay import archive
from apps.replay.assemble import records
from apps.replay.symbol_replay import build_day

REPLAY_TABLE = "replay_experiences"
INSERT_BATCH = 50


def replay_table(engine) -> Table:
    with engine.begin() as c:
        c.execute(text(f"CREATE TABLE IF NOT EXISTS {REPLAY_TABLE} LIKE experiences"))
    return Table(REPLAY_TABLE, MetaData(), autoload_with=engine)


def _build(job):
    s, d = job
    t0 = time.time()
    try:
        return s, d, build_day(s, d), time.time() - t0, None
    except Exception as e:
        return s, d, None, time.time() - t0, repr(e)


def run_pass1(symbols: List[str], days: List[date], workers: int):
    jobs = [(s, d) for d in days for s in symbols]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for s, d, path, secs, err in pool.map(_build, jobs):
            done += 1
            status = "ok" if path else ("FAILED " + err if err else "no data")
            print(f"pass 1 [{done}/{len(jobs)}] {s} {d}: {status} ({secs:.0f}s)", flush=True)


def write_records(engine, table: Table, symbols: List[str], days: List[date], record_symbols: List[str]):
    """Assembles and inserts records day by day. A day already present in the table is skipped
    (its frames are still assembled so the next day's sequences continue across midnight)."""
    with engine.connect() as c:
        have = {int(t) // 86400 for (t,) in c.execute(select(table.c.timestamp))}
    batch, total = [], 0
    for rec in records(symbols, days, log=lambda m: print(m, flush=True), record_symbols=record_symbols):
        if rec["timestamp"] // 86400 in have:
            continue
        batch.append(rec)
        if len(batch) >= INSERT_BATCH:
            with engine.begin() as c:
                c.execute(insert(table), batch)
            total += len(batch)
            batch = []
    if batch:
        with engine.begin() as c:
            c.execute(insert(table), batch)
        total += len(batch)
    print(f"wrote {total} replay records", flush=True)


def grade(engine, table: Table, symbols: List[str], days: List[date]):
    """True market returns (5m / 1h / 4h) and LONG / SHORT trade outcomes, as the live grader does."""
    from apps.experience.reward_calculator import RewardCalculator, KLINE_MS
    calc = RewardCalculator()
    # Archive klines as the price oracle: no REST calls, identical candles
    kdays = [days[0] - timedelta(days=1)] + days + [days[-1] + timedelta(days=1)]
    for s in symbols:
        rows = [archive.load_klines("klines", s, d) for d in kdays]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        k = np.concatenate(rows)
        cache = calc.prices.cache[s]
        for r in k:
            cache[int(r[0])] = (float(r[1]), float(r[2]), float(r[3]), float(r[4]))
        calc.prices.fetched_ranges[s].append((int(k[0, 0]), int(k[-1, 0]) + KLINE_MS))
    calc.prices.prune = lambda keep_after_ms: None  # keep the preloaded history
    n = calc.grade_all(table=table)
    print(f"graded {n} replay records", flush=True)
    n = calc.outcomes_all(table=table, src="replay")
    print(f"trade outcomes for {n} replay records", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first UTC day (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="last UTC day (YYYY-MM-DD)")
    ap.add_argument("--symbols", default=",".join(settings.symbol_universe))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-grading", action="store_true")
    a = ap.parse_args()
    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    days = archive.day_range(date.fromisoformat(a.start), date.fromisoformat(a.end))
    print(f"replay {len(symbols)} coins x {len(days)} days ({days[0]} .. {days[-1]})", flush=True)

    # archive downloads (pass 1 also downloads on demand; doing it first keeps the workers busy computing)
    archive.download_all(symbols, [days[0] - timedelta(days=i) for i in range(4, 0, -1)] + days, kinds=("klines",))
    archive.download_all(symbols, [days[0] - timedelta(days=1)] + days)
    run_pass1(symbols, days, a.workers)

    engine = create_engine(settings.database_url)
    table = replay_table(engine)
    excluded = set(getattr(settings, "excluded_trading_symbols", []))
    record_symbols = [s for s in symbols if s not in excluded]
    write_records(engine, table, symbols, days, record_symbols)
    if not a.skip_grading:
        grade(engine, table, record_symbols, days)


if __name__ == "__main__":
    main()
