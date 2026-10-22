"""
Removes training data recorded while the bot read the Binance TESTNET market (before 2026-10-22 06:00 UTC,
state_layout.FULL_INPUTS_SINCE). Those records describe a fake market and are never trained on again.

    python scripts/purge_testnet_data.py          # dry run: only counts what would be removed
    python scripts/purge_testnet_data.py --yes    # deletes (cannot be undone)

Run it on the laptop and on the VPS (venv/bin/python, PYTHONPATH=.). It touches only:
  experiences rows older than the cutoff (the rows kept are copied to a fresh table that replaces the old one),
  trade_outcomes of deleted live / golden rows, all golden_experiences rows (all older than the cutoff),
  and the old backup / archive tables listed in OLD_TABLES.
Records from the cutoff on, and the replay_experiences table, are kept.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, inspect, text

from core.config.settings import settings
from core.schemas.state_layout import FULL_INPUTS_SINCE

OLD_TABLES = ("phantom_experiences_archive", "experiences_reward_backup_20261021",
              "golden_experiences_reward_backup_20261021")


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

    # Fast path: copy the rows we KEEP into a fresh table and swap it in, then drop the old table in one
    # step (deleting ~0.5 MB rows one by one takes hours). Rows the live bot writes during the swap are
    # copied over right after it.
    for table in ("experiences", "golden_experiences"):
        if table not in tables:
            continue
        keep, old_t = f"{table}_keep", f"{table}_purged"
        with eng.begin() as c:
            c.execute(text(f"DROP TABLE IF EXISTS {keep}"))
            c.execute(text(f"CREATE TABLE {keep} LIKE {table}"))
            if table == "experiences":
                c.execute(text(f"INSERT INTO {keep} SELECT * FROM {table} WHERE timestamp >= :t"), {"t": cutoff})
            c.execute(text(f"RENAME TABLE {table} TO {old_t}, {keep} TO {table}"))
        with eng.begin() as c:
            if table == "experiences":
                late = c.execute(text(f"INSERT IGNORE INTO {table} SELECT * FROM {old_t} WHERE timestamp >= :t"),
                                 {"t": cutoff}).rowcount
                if late:
                    print(f"  copied {late} rows written during the swap")
            c.execute(text(f"DROP TABLE {old_t}"))
        with eng.connect() as c:
            left = c.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"{table}: old rows removed, {left} rows kept")

    with eng.begin() as c:
        if "trade_outcomes" in tables:
            n = c.execute(text("DELETE o FROM trade_outcomes o LEFT JOIN experiences e ON e.id = o.exp_id "
                               "WHERE o.src = 'exp' AND e.id IS NULL")).rowcount
            n += c.execute(text("DELETE FROM trade_outcomes WHERE src = 'gold'")).rowcount
            print(f"deleted {n} trade outcomes of removed rows")
        for t in drop:
            c.execute(text(f"DROP TABLE {t}"))
            print(f"dropped {t}")
    print("done")


if __name__ == "__main__":
    main()
