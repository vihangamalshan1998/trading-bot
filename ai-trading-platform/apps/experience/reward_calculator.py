import sys
import time
import json
import asyncio
import urllib.request
import urllib.parse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sqlalchemy import create_engine, update, or_, func, insert, Table, MetaData
from sqlalchemy.orm import sessionmaker
from core.database.models.ai import Experience, TradeOutcome, GRADED_FLAG, UNGRADEABLE_FLAG
from core.risk import exit_policy
from core.config.settings import settings
from core.logging.logger import logger, set_log_file

HORIZONS_SEC = (300, 3600, 14400)  # 5m, 1h, 4h
KLINE_MS = 60_000
KLINES_PER_REQUEST = 1500
VOL_HISTORY_MIN = 1000  # 1m bars before entry used for the 1h ATR stop distance (same window as the live FeatureEngine)
OUTCOME_BARS = exit_policy.MAX_HOLD_SECONDS // 60


class KlinePriceSource:
    """
    Price oracle backed by Binance 1m klines (public endpoint, no API key).
    Uses the market-data venue (mainnet by default, settings.market_data_mainnet) so graded
    returns describe the same market the brain's inputs come from.

    price_at(t) = open of the first 1m candle whose open time is >= t (at most 60s late,
    identical rule for the start and every horizon, so the bias cancels out).
    """
    def __init__(self):
        from apps.market_collector.context_poller import market_data_rest_url
        self.base_url = market_data_rest_url()
        # {symbol: {minute_open_ms: (open, high, low, close)}}
        self.cache: Dict[str, Dict[int, Tuple[float, float, float, float]]] = defaultdict(dict)
        self.fetched_ranges: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def _fetch(self, symbol: str, start_ms: int, end_ms: int):
        cursor = start_ms
        while cursor < end_ms:
            params = urllib.parse.urlencode({
                "symbol": symbol, "interval": "1m", "startTime": cursor, "limit": KLINES_PER_REQUEST
            })
            with urllib.request.urlopen(f"{self.base_url}/fapi/v1/klines?{params}", timeout=20) as resp:
                rows = json.load(resp)
            if not rows:
                break
            for k in rows:
                self.cache[symbol][int(k[0])] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]))
            last_open = int(rows[-1][0])
            if last_open + KLINE_MS <= cursor:
                break
            cursor = last_open + KLINE_MS
            time.sleep(0.1)  # stay far below public rate limits
        self.fetched_ranges[symbol].append((start_ms, end_ms))

    def _covered(self, symbol: str, start_ms: int, end_ms: int) -> bool:
        return any(s <= start_ms and end_ms <= e for s, e in self.fetched_ranges[symbol])

    def prune(self, keep_after_ms: int):
        """Drops cached candles older than keep_after_ms (the service runs for weeks; the cache only grew)."""
        for symbol, candles in self.cache.items():
            for k in [k for k in candles if k < keep_after_ms]:
                del candles[k]
        for symbol in list(self.fetched_ranges):
            self.fetched_ranges[symbol] = [(s, e) for s, e in self.fetched_ranges[symbol] if s >= keep_after_ms]

    def ensure(self, symbol: str, start_ms: int, end_ms: int):
        if not self._covered(symbol, start_ms, end_ms):
            self._fetch(symbol, start_ms, end_ms)

    def price_at(self, symbol: str, t_sec: int) -> Optional[float]:
        t_ms = int(t_sec) * 1000
        minute = ((t_ms + KLINE_MS - 1) // KLINE_MS) * KLINE_MS
        candles = self.cache.get(symbol, {})
        # Tolerate up to 2 missing candles (exchange maintenance gaps)
        for k in range(3):
            bar = candles.get(minute + k * KLINE_MS)
            if bar and bar[0] > 0:
                return bar[0]
        return None

    def entry_minute(self, t_sec: int) -> int:
        """Open time (ms) of the candle whose open is the entry price used by price_at()."""
        t_ms = int(t_sec) * 1000
        return ((t_ms + KLINE_MS - 1) // KLINE_MS) * KLINE_MS

    def bars(self, symbol: str, start_ms: int, n: int) -> List[Tuple[float, float, float, float]]:
        """Up to n consecutive 1m candles from start_ms (missing minutes are skipped)."""
        candles = self.cache.get(symbol, {})
        out = []
        for i in range(n):
            bar = candles.get(start_ms + i * KLINE_MS)
            if bar:
                out.append(bar)
        return out


class RewardCalculator:
    """
    The 'Teacher's Assistant'.
    Grades every experience (HOLD included) with the TRUE forward market return of its symbol
    over 5m / 1h / 4h, in percent of price (signed, direction-free):

        reward_h = (price(t + h) - price(t)) / price(t) * 100

    These are exactly the units the live bot assumes when it reads the critic head
    ("critic_return 5m = +0.15%"), and they let the trainer score ANY action counterfactually.
    """
    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.prices = KlinePriceSource()

    def process_batch(self, batch_size: int = 2000, table: Optional[Table] = None) -> int:
        t = Experience.__table__ if table is None else table
        try:
            # Grade only once the full 4h horizon (+ 30 min margin) is in the past
            safe_cutoff = int(time.time()) - (HORIZONS_SEC[-1] + 1800)
            if time.time() - getattr(self, "_last_prune", 0.0) > 3600:
                self._last_prune = time.time()
                self.prices.prune(int((time.time() - 2 * 86400) * 1000))

            with self.SessionLocal() as session:
                # Light-weight column query: never load the 300x200 market_state JSON here.
                rows = session.query(t.c.id, t.c.symbol, t.c.timestamp).filter(
                    t.c.timestamp < safe_cutoff,
                    or_(t.c.data_quality_flags.is_(None),
                        t.c.data_quality_flags.notin_([GRADED_FLAG, UNGRADEABLE_FLAG]))
                ).order_by(t.c.id.asc()).limit(batch_size).all()

                if not rows:
                    return 0

                by_symbol = defaultdict(list)
                for r in rows:
                    by_symbol[r.symbol].append(r)

                graded = failed = 0
                for symbol, sym_rows in by_symbol.items():
                    t_min = min(r.timestamp for r in sym_rows)
                    t_max = max(r.timestamp for r in sym_rows)
                    try:
                        self.prices.ensure(symbol, t_min * 1000 - KLINE_MS, (t_max + HORIZONS_SEC[-1]) * 1000 + 3 * KLINE_MS)
                    except Exception as e:
                        if t_max < time.time() - 3 * 86400:
                            # Prices for these rows are not coming back. Left ungraded, they would fill every
                            # id-ordered batch and stop grading for all other coins.
                            for r in sym_rows:
                                session.execute(update(t).where(t.c.id == r.id).values(data_quality_flags=UNGRADEABLE_FLAG))
                            failed += len(sym_rows)
                            logger.warning(f"Reward Calculator: no klines for {symbol} rows older than 3 days ({e}); flagged {UNGRADEABLE_FLAG}.")
                        else:
                            logger.warning(f"Reward Calculator: kline fetch failed for {symbol}: {e}. Will retry later.")
                        continue

                    for r in sym_rows:
                        p0 = self.prices.price_at(symbol, r.timestamp)
                        futures = [self.prices.price_at(symbol, r.timestamp + h) for h in HORIZONS_SEC]
                        if not p0 or any(p is None for p in futures):
                            session.execute(update(t).where(t.c.id == r.id).values(data_quality_flags=UNGRADEABLE_FLAG))
                            failed += 1
                            continue
                        r5m, r1h, r4h = [float((p - p0) / p0 * 100.0) for p in futures]
                        session.execute(update(t).where(t.c.id == r.id).values(
                            reward_5m=r5m, reward_1h=r1h, reward_4h=r4h, data_quality_flags=GRADED_FLAG
                        ))
                        graded += 1

                session.commit()
                if failed:
                    logger.warning(f"Reward Calculator: {failed} rows had no price data and were flagged {UNGRADEABLE_FLAG}.")
                return graded + failed

        except Exception as e:
            logger.error(f"Reward Calculator Error: {e}")
            return 0

    def outcome_for(self, symbol: str, t_sec: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        (long_pct, short_pct, stop_pct): realized gross price move of a LONG / SHORT opened at t_sec under
        the live exit policy, replayed on 1m klines. Klines for [t - 1000m, t + 4h] must be loaded.
        """
        start = self.prices.entry_minute(t_sec)
        fwd = self.prices.bars(symbol, start, OUTCOME_BARS)
        if len(fwd) < int(OUTCOME_BARS * 0.9) or fwd[0][0] <= 0:
            return None, None, None
        hist = self.prices.bars(symbol, start - VOL_HISTORY_MIN * KLINE_MS, VOL_HISTORY_MIN)
        levels = exit_policy.exit_levels(exit_policy.volatility_1h_from_ohlc(hist) if hist else 0.0)
        entry = fwd[0][0]
        path = [(h, l, c) for _, h, l, c in fwd]
        long_pct = exit_policy.simulate_trade(1, entry, path, levels) * 100.0
        short_pct = exit_policy.simulate_trade(-1, entry, path, levels) * 100.0
        return float(long_pct), float(short_pct), float(levels.stop * 100.0)

    def process_outcomes(self, table: Optional[Table] = None, src: str = "exp", batch_size: int = 2000) -> int:
        """
        Adds path-aware trade outcomes (trade_outcomes table) for graded rows that do not have one yet.
        Rows are processed in id order after the last labelled id, so each call is a cheap range scan.
        """
        t = Experience.__table__ if table is None else table
        o = TradeOutcome.__table__
        try:
            o.create(self.engine, checkfirst=True)
            with self.SessionLocal() as session:
                last_id = session.query(func.max(o.c.exp_id)).filter(o.c.src == src).scalar() or 0
                rows = session.query(t.c.id, t.c.symbol, t.c.timestamp).filter(
                    t.c.id > last_id, t.c.data_quality_flags == GRADED_FLAG
                ).order_by(t.c.id.asc()).limit(batch_size).all()
                if not rows:
                    return 0
                by_symbol = defaultdict(list)
                for r in rows:
                    by_symbol[r.symbol].append(r)
                values, missing_klines = [], False
                for symbol, sym_rows in by_symbol.items():
                    t_min = min(r.timestamp for r in sym_rows)
                    t_max = max(r.timestamp for r in sym_rows)
                    try:
                        self.prices.ensure(symbol, (t_min - VOL_HISTORY_MIN * 60) * 1000 - KLINE_MS,
                                           (t_max + exit_policy.MAX_HOLD_SECONDS) * 1000 + 3 * KLINE_MS)
                    except Exception as e:
                        logger.warning(f"Trade outcomes: kline fetch failed for {symbol}: {e}. Will retry later.")
                        missing_klines = True
                        continue
                    for r in sym_rows:
                        long_pct, short_pct, stop_pct = self.outcome_for(symbol, r.timestamp)
                        values.append({"src": src, "exp_id": r.id, "symbol": symbol, "timestamp": r.timestamp,
                                       "long_pct": long_pct, "short_pct": short_pct, "stop_pct": stop_pct})
                if missing_klines:
                    # Keep id order intact: never skip past rows whose klines could not be fetched
                    done = {v["exp_id"] for v in values}
                    first_missing = min(r.id for r in rows if r.id not in done)
                    values = [v for v in values if v["exp_id"] < first_missing]
                if values:
                    session.execute(insert(o), values)
                    session.commit()
                return len(values)
        except Exception as e:
            logger.error(f"Trade outcome labelling error: {e}")
            return 0

    def outcomes_all(self, table: Optional[Table] = None, src: str = "exp") -> int:
        total = 0
        while True:
            n = self.process_outcomes(table=table, src=src)
            if n == 0:
                return total
            total += n
            logger.info(f"Trade outcomes: labelled {total} {src} rows so far...")

    def grade_all(self, table: Optional[Table] = None) -> int:
        """Grades the whole backlog (used for the one-off re-grade of legacy rows)."""
        total = 0
        while True:
            n = self.process_batch(table=table)
            if n == 0:
                return total
            total += n
            logger.info(f"Reward Calculator: graded {total} rows in {(table if table is not None else Experience.__table__).name} so far...")


def golden_table() -> Table:
    """Golden experiences share the Experience schema in a separate table."""
    return Table("golden_experiences", MetaData(), autoload_with=create_engine(settings.database_url))


async def main():
    set_log_file("logs/reward_calculator.log")
    logger.info("Starting Teacher's Assistant (Reward Calculator, true market returns)...")
    calculator = RewardCalculator()

    if "--outcomes" in sys.argv:
        n = calculator.outcomes_all()
        logger.info(f"Trade outcomes complete: {n} experiences.")
        try:
            n = calculator.outcomes_all(table=golden_table(), src="gold")
            logger.info(f"Trade outcomes complete: {n} golden experiences.")
        except Exception as e:
            logger.warning(f"Golden table outcomes skipped: {e}")
        return

    if "--regrade" in sys.argv:
        n = calculator.grade_all()
        logger.info(f"Re-grade complete: {n} experiences.")
        try:
            gt = golden_table()
            n = calculator.grade_all(table=gt)
            logger.info(f"Re-grade complete: {n} golden experiences.")
        except Exception as e:
            logger.warning(f"Golden table re-grade skipped: {e}")
        return

    while True:
        processed = calculator.process_batch()
        if processed > 0:
            logger.info(f"Graded {processed} historical experiences with Multi-Horizon market returns.")
            await asyncio.sleep(2.0)
        else:
            await asyncio.sleep(60.0)

if __name__ == "__main__":
    asyncio.run(main())
