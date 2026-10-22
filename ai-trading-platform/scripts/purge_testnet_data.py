"""
Removes training data recorded while the bot read the Binance TESTNET market (before 2026-10-22 06:00 UTC,
state_layout.FULL_INPUTS_SINCE). Those records describe a fake market and are never trained on again.

    python scripts/purge_testnet_data.py          # dry run: only counts what would be removed
    python scripts/purge_testnet_data.py --yes    # deletes (cannot be undone)

Run it on the laptop and on the VPS (venv/bin/python, PYTHONPATH=.). It touches only:
  experiences rows older than the cutoff (in small batches, so the live bot keeps writing),
  trade_outcomes of deleted live / golden rows, all golden_experiences rows (all older than the cutoff),
  and the old backup / archive tables listed in OLD_TABLES.
Records from the cutoff on, and the replay_experiences table, are kept.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, inspect, text

from core.config.settings import settings
from core.schemas.state_layout import FULL_INPUTS_SINCE

OLD_TABLES = ("phantom_experiences_archive", "experiences_reward_backup_20261021",
              "golden_experiences_reward_backup_20261021")
BATCH = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="really delete (default: dry run)")
    a = ap.parse_args()
    eng = create_engine(settings.database_url)
    tables = set(inspect(eng).get_table_names())
    cutoff = int(FULL_INPUTS_SINCE)

    with eng.connect() as c:
        old = c.execute(text("SELECT COUNT(*) FROM experiences WHERE timestamp < :t"), {"t": cutoff}).scalar()
        new = c.execute(text("SELECT COUNT(*) FROM experiences WHERE timestamp >= :t"), {"t": cutoff}).scalar()
        gold = c.execute(text("SELECT COUNT(*) FROM golden_experiences")).scalar() if "golden_experiences" in tables else 0
        gold_new = (c.execute(text("SELECT COUNT(*) FROM golden_experiences WHERE timestamp >= :t"), {"t": cutoff}).scalar()
                    if "golden_experiences" in tables else 0)
    drop = [t for t in OLD_TABLES if t in tables]
    print(f"cutoff {cutoff} (2026-10-22 06:00 UTC)")
    print(f"experiences: {old} old rows to delete, {new} new rows kept")
    print(f"golden_experiences: {gold} rows to delete ({gold_new} of them newer than the cutoff)")
    print(f"old tables to drop: {drop or 'none'}")
    if not a.yes:
        print("dry run: nothing deleted (add --yes to delete)")
        return

    total = 0
    while True:
        with eng.begin() as c:
            n = c.execute(text(f"DELETE FROM experiences WHERE timestamp < :t LIMIT {BATCH}"), {"t": cutoff}).rowcount
        total += n
        if n == 0:
            break
        if total % 3000 < BATCH:
            print(f"  deleted {total} / {old} ...", flush=True)
        time.sleep(0.1)
    print(f"deleted {total} old experiences")

    with eng.begin() as c:
        if "trade_outcomes" in tables:
            n = c.execute(text("DELETE o FROM trade_outcomes o LEFT JOIN experiences e ON e.id = o.exp_id "
                               "WHERE o.src = 'exp' AND e.id IS NULL")).rowcount
            n += c.execute(text("DELETE FROM trade_outcomes WHERE src = 'gold'")).rowcount
            print(f"deleted {n} trade outcomes of removed rows")
        if "golden_experiences" in tables:
            print(f"deleted {c.execute(text('DELETE FROM golden_experiences')).rowcount} golden rows")
        for t in drop:
            c.execute(text(f"DROP TABLE {t}"))
            print(f"dropped {t}")

    print("reclaiming disk space (OPTIMIZE TABLE)...", flush=True)
    with eng.connect() as c:
        for t in ("experiences", "golden_experiences", "trade_outcomes"):
            if t in tables:
                c.execute(text(f"OPTIMIZE TABLE {t}")).fetchall()
    print("done")


if __name__ == "__main__":
    main()
