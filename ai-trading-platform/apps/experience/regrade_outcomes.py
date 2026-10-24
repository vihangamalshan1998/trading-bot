"""
Re-grades every path-aware trade outcome (trade_outcomes: live 'exp', 'gold' and replayed 'rpl' rows) with
the CURRENT exit policy (core/risk/exit_policy.py), in place: each row keeps its old label until its new one
is written, so the trainer can keep running. Needed after an exit-rule change (24 Oct 2026: stop = 1h swing up
to 2% + breakeven guard after +1%), so the trainer and the tournament grade trades the way the bot now trades.

Prices: the Binance archive's 1m klines (identical candles, no REST), with the REST source only for the last
days the archive does not have yet. Resumable: finished coins are listed in data/regrade_outcomes_progress.json.

Run on the training machine from ai-trading-platform (restart MASTER_TRAINER first, so new labels it writes
already use the new rules):
    venv\\Scripts\\python -m apps.experience.regrade_outcomes
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import bindparam, func, update

from apps.experience.reward_calculator import KLINE_MS, VOL_HISTORY_MIN, RewardCalculator
from apps.replay import archive
from core.database.models.ai import TradeOutcome
from core.risk import exit_policy

PROGRESS_PATH = os.path.join("data", "regrade_outcomes_progress.json")
POLICY_TAG = f"stop<= {exit_policy.MAX_STOP} breakeven@{exit_policy.BREAKEVEN_TRIGGER}"
UPDATE_CHUNK = 2000


def _load_progress() -> set:
    try:
        with open(PROGRESS_PATH) as f:
            d = json.load(f)
        return set(d.get("done", [])) if d.get("policy") == POLICY_TAG else set()
    except (OSError, ValueError):
        return set()


def _save_progress(done: set):
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"policy": POLICY_TAG, "done": sorted(done)}, f)


def _preload_archive(calc: RewardCalculator, symbol: str, t_min: int, t_max: int):
    first = datetime.fromtimestamp(t_min - VOL_HISTORY_MIN * 60, timezone.utc).date() - timedelta(days=1)
    last = datetime.fromtimestamp(t_max + exit_policy.MAX_HOLD_SECONDS, timezone.utc).date()
    cache = calc.prices.cache[symbol]
    d = first
    while d <= last:
        rows = archive.load_klines("klines", symbol, d)
        if rows is not None and len(rows):
            for r in rows:
                cache[int(r[0])] = (float(r[1]), float(r[2]), float(r[3]), float(r[4]))
            calc.prices.fetched_ranges[symbol].append((int(rows[0][0]), int(rows[-1][0]) + KLINE_MS))
        d += timedelta(days=1)


def regrade_symbol(calc: RewardCalculator, session, symbol: str, dry: bool = False) -> int:
    o = TradeOutcome.__table__
    rows = session.query(o.c.id, o.c.timestamp, o.c.long_pct, o.c.short_pct).filter(
        o.c.symbol == symbol).order_by(o.c.timestamp).all()
    if not rows:
        return 0
    calc.prices.cache[symbol] = {}
    calc.prices.fetched_ranges[symbol] = []
    _preload_archive(calc, symbol, rows[0].timestamp, rows[-1].timestamp)
    # REST only for the newest days the archive has not published yet
    need_end = (rows[-1].timestamp + exit_policy.MAX_HOLD_SECONDS) * 1000 + 3 * KLINE_MS
    have_end = max((e for _, e in calc.prices.fetched_ranges[symbol]), default=0)
    if have_end < need_end:
        start = have_end or (rows[0].timestamp - VOL_HISTORY_MIN * 60) * 1000 - KLINE_MS
        try:
            calc.prices.ensure(symbol, start, need_end)
        except Exception as e:
            print(f"  {symbol}: REST top-up failed ({e}); rows without a full price path keep their old label", flush=True)
    values = []
    for r in rows:
        long_pct, short_pct, stop_pct = calc.outcome_for(symbol, int(r.timestamp))
        if long_pct is None:
            continue
        values.append({"row_id": r.id, "lp": long_pct, "sp": short_pct, "st": stop_pct})
    if dry:
        old = {r.id: (r.long_pct, r.short_pct) for r in rows}
        ch = [v for v in values if old[v["row_id"]][0] is not None]
        mean = lambda xs: sum(xs) / max(1, len(xs))
        print(f"  {symbol} DRY RUN, {len(ch)} rows: long {mean([old[v['row_id']][0] for v in ch]):+.4f}% -> {mean([v['lp'] for v in ch]):+.4f}%, "
              f"short {mean([old[v['row_id']][1] for v in ch]):+.4f}% -> {mean([v['sp'] for v in ch]):+.4f}%, "
              f"median stop {sorted(v['st'] for v in ch)[len(ch) // 2] if ch else 0:.2f}%", flush=True)
        return len(values)
    stmt = update(o).where(o.c.id == bindparam("row_id")).values(
        long_pct=bindparam("lp"), short_pct=bindparam("sp"), stop_pct=bindparam("st"))
    for i in range(0, len(values), UPDATE_CHUNK):
        session.execute(stmt, values[i:i + UPDATE_CHUNK])
        session.commit()
    calc.prices.cache[symbol] = {}
    return len(values)


def main():
    calc = RewardCalculator()
    calc.prices.prune = lambda keep_after_ms: None
    done = _load_progress()
    o = TradeOutcome.__table__
    dry = "--dry" in sys.argv
    with calc.SessionLocal() as session:
        counts = dict(session.query(o.c.symbol, func.count(o.c.id)).group_by(o.c.symbol).all())
        if dry:
            sym = sys.argv[sys.argv.index("--dry") + 1] if len(sys.argv) > sys.argv.index("--dry") + 1 else sorted(counts)[0]
            regrade_symbol(calc, session, sym, dry=True)
            return 0
        todo = sorted(s for s in counts if s not in done)
        total = sum(counts.values())
        print(f"Re-grading {total} trade outcomes ({len(counts)} coins, {len(done)} already done) with: {POLICY_TAG}", flush=True)
        t0 = time.time()
        for i, sym in enumerate(todo, 1):
            n = regrade_symbol(calc, session, sym)
            done.add(sym)
            _save_progress(done)
            print(f"  [{i}/{len(todo)}] {sym}: {n} of {counts[sym]} re-graded ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
