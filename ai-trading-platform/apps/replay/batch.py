"""
Builds the replay history week by week, newest week first (REPLAY_HISTORY.bat).

    python -m apps.replay.batch --days 60 --workers 4

Stop it any time (close the window); start it again and it skips finished weeks and finishes the one it
was on. Newest first, so stopping early still leaves the most recent (most relevant) history.
While it runs, Windows is kept awake and the work runs at low priority so the laptop stays usable.
"""
import argparse
import sys
import time
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import create_engine

from core.config.settings import settings
from apps.replay import archive
from apps.replay.run import finished_days, run_range

# The archive publishes a day after it ends (sometimes a day later): stop two days back
NEWEST_DAY_LAG = 2
BATCH_DAYS = 7


def _keep_awake_low_priority():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000  # inherited by the worker processes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        print("Windows kept awake while this runs; low priority so the laptop stays usable.", flush=True)
    except Exception as e:
        print(f"(could not set keep-awake / priority: {e})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="how many days of history (ending 2 days ago)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    _keep_awake_low_priority()

    newest = datetime.now(timezone.utc).date() - timedelta(days=NEWEST_DAY_LAG)
    oldest = newest - timedelta(days=a.days - 1)
    weeks = []
    end = newest
    while end >= oldest:
        start = max(oldest, end - timedelta(days=BATCH_DAYS - 1))
        weeks.append(archive.day_range(start, end))
        end = start - timedelta(days=1)

    symbols = list(settings.symbol_universe)
    engine = create_engine(settings.database_url)
    t_all = time.time()
    print(f"History replay: {oldest} .. {newest} ({a.days} days) in {len(weeks)} weekly batches, newest first.", flush=True)
    for i, days in enumerate(weeks, 1):
        done = finished_days(engine)
        if all(d in done for d in days):
            print(f"[batch {i}/{len(weeks)}] {days[0]} .. {days[-1]}: already done, skipped.", flush=True)
            continue
        t0 = time.time()
        print(f"\n[batch {i}/{len(weeks)}] {days[0]} .. {days[-1]}: starting...", flush=True)
        run_range(symbols, days, a.workers)
        print(f"[batch {i}/{len(weeks)}] finished in {(time.time() - t0) / 60:.0f} min.", flush=True)
    print(f"\nALL DONE: {len(weeks)} batches, {(time.time() - t_all) / 60:.0f} min this run. "
          f"Finished days in the database: {len(finished_days(engine))}.", flush=True)


if __name__ == "__main__":
    main()
