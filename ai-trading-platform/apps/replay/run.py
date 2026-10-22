"""
History replay runner (training machine):

    python -m apps.replay.run --start 2026-08-20 --end 2026-10-20 --workers 6

Run it in batches (e.g. one week per run) if you like; every step resumes after a stop, crash or power cut:
downloaded files, replayed coin-days and finished days are kept; a day cut short is cleared and redone.

1. downloads the archive files (cached),
2. pass 1: per coin-day market features (parallel, cached),
3. pass 2: assembles 200-slot frames for all coins, saves them per coin-day (apps/replay/states.py) and
   writes one small index row per training record to the `replay_records` table,
4. grades them with the live reward calculator (true market returns + path-aware trade outcomes,
   src REPLAY_OUTCOME_SRC), using the archive's own 1m klines as the price source.

Live records stay in `experiences`; the trainer reads replayed records as a separate pool, cutting each
record's 300 frames out of the coin-day files.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from typing import List

import numpy as np
from sqlalchemy import (BigInteger, Column, Date, Float, Integer, MetaData, String, Table, create_engine,
                        insert, text)

from core.config.settings import settings
from apps.replay import archive
from apps.replay.assemble import record_index
from apps.replay.states import REPLAY_OUTCOME_SRC, REPLAY_TABLE
from apps.replay.symbol_replay import build_day

DAYS_TABLE = "replay_record_days"
INSERT_BATCH = 2000


def replay_engine():
    """
    Database engine for the replay writer. Its connections skip the MySQL binary log: that log only serves
    replication, which the training laptop does not use, and it doubled every ~0.7 MB record write (and kept
    a second copy on disk). Other clients of the database are not affected.
    """
    from sqlalchemy import event
    engine = create_engine(settings.database_url)

    @event.listens_for(engine, "connect")
    def _no_binlog(dbapi_conn, _record):
        try:
            with dbapi_conn.cursor() as cur:
                cur.execute("SET SESSION sql_log_bin = 0")
        except Exception:
            pass  # no privilege: keep the default

    return engine


def replay_table(engine) -> Table:
    """One row per replayed training record: where its frames are, and the grades (columns named as in
    experiences, so the live reward calculator grades it unchanged)."""
    table = Table(
        REPLAY_TABLE, MetaData(),
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("experience_id", String(64), unique=True),
        Column("timestamp", BigInteger, index=True),
        Column("symbol", String(20), index=True),
        Column("day", Date, index=True),
        Column("frame_idx", Integer),
        Column("action", String(20)),
        Column("model_version", String(50)),
        Column("reward_5m", Float), Column("reward_1h", Float), Column("reward_4h", Float),
        Column("data_quality_flags", String(200), index=True),
    )
    table.create(engine, checkfirst=True)
    return table


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


def _days_table(engine):
    with engine.begin() as c:
        c.execute(text(f"CREATE TABLE IF NOT EXISTS {DAYS_TABLE} (day DATE PRIMARY KEY, records INT NOT NULL, "
                       f"finished_at BIGINT NOT NULL)"))


def write_records(engine, table: Table, symbols: List[str], days: List[date], record_symbols: List[str]):
    """
    Assembles each day, saves its frames and inserts its index rows. A day counts as written only once ALL
    its rows are in (DAYS_TABLE). A day cut short by a crash or power cut is cleared and written again on the
    next run; finished days are skipped (their frames are still assembled so sequences continue across midnight).
    """
    _days_table(engine)
    with engine.connect() as c:
        done = {d for (d,) in c.execute(text(f"SELECT day FROM {DAYS_TABLE}"))}
    for d in days:
        if d in done:
            continue
        with engine.begin() as c:
            n = c.execute(table.delete().where(table.c.day == d)).rowcount
        if n:
            print(f"{d}: removed {n} records of an unfinished earlier run", flush=True)

    batch, total, day_count = [], 0, 0
    current = None

    def flush():
        nonlocal batch, total
        if batch:
            with engine.begin() as c:
                c.execute(insert(table), batch)
            total += len(batch)
            batch = []

    def finish(d, count):
        flush()
        with engine.begin() as c:
            c.execute(text(f"INSERT INTO {DAYS_TABLE} (day, records, finished_at) VALUES (:d, :n, :t) "
                           f"ON DUPLICATE KEY UPDATE records = :n, finished_at = :t"),
                      {"d": d, "n": count, "t": int(time.time())})
        print(f"{d}: {count} records saved (day complete)", flush=True)

    for rec in record_index(symbols, days, log=lambda m: print(m, flush=True), record_symbols=record_symbols):
        d = rec["day"]
        if d != current:
            if current is not None and current not in done:
                finish(current, day_count)
            current, day_count = d, 0
        if d in done:
            continue
        batch.append(rec)
        day_count += 1
        if len(batch) >= INSERT_BATCH:
            flush()
    if current is not None and current not in done:
        finish(current, day_count)
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
    n = calc.outcomes_all(table=table, src=REPLAY_OUTCOME_SRC)
    print(f"trade outcomes for {n} replay records", flush=True)


def run_range(symbols: List[str], days: List[date], workers: int, grading: bool = True):
    """The whole pipeline for consecutive days (each step resumes after a stop)."""
    print(f"replay {len(symbols)} coins x {len(days)} days ({days[0]} .. {days[-1]})", flush=True)
    # archive downloads (pass 1 also downloads on demand; doing it first keeps the workers busy computing)
    archive.download_all(symbols, [days[0] - timedelta(days=i) for i in range(4, 0, -1)] + days, kinds=("klines",))
    archive.download_all(symbols, [days[0] - timedelta(days=1)] + days)
    run_pass1(symbols, days, workers)

    engine = replay_engine()
    table = replay_table(engine)
    excluded = set(getattr(settings, "excluded_trading_symbols", []))
    record_symbols = [s for s in symbols if s not in excluded]
    write_records(engine, table, symbols, days, record_symbols)
    if grading:
        grade(engine, table, record_symbols, days)


def finished_days(engine) -> set:
    _days_table(engine)
    with engine.connect() as c:
        return {d for (d,) in c.execute(text(f"SELECT day FROM {DAYS_TABLE}"))}


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
    run_range(symbols, days, a.workers, grading=not a.skip_grading)


if __name__ == "__main__":
    main()
