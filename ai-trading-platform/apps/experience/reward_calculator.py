import sys
import time
import json
import asyncio
import urllib.request
import urllib.parse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sqlalchemy import create_engine, update, or_, Table, MetaData
from sqlalchemy.orm import sessionmaker
from core.database.models.ai import Experience, GRADED_FLAG, UNGRADEABLE_FLAG
from core.config.settings import settings
from core.logging.logger import logger, set_log_file

HORIZONS_SEC = (300, 3600, 14400)  # 5m, 1h, 4h
KLINE_MS = 60_000
KLINES_PER_REQUEST = 1500


class KlinePriceSource:
    """
    Price oracle backed by Binance 1m klines (public endpoint, no API key).
    Uses the same testnet/mainnet base URL as the execution adapter so graded
    returns describe the exact market the bot trades.

    price_at(t) = open of the first 1m candle whose open time is >= t (at most 60s late,
    identical rule for the start and every horizon, so the bias cancels out).
    """
    def __init__(self):
        self.base_url = "https://testnet.binancefuture.com" if settings.binance_testnet else "https://fapi.binance.com"
        # {symbol: {minute_open_ms: open_price}}
        self.cache: Dict[str, Dict[int, float]] = defaultdict(dict)
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
                self.cache[symbol][int(k[0])] = float(k[1])
            last_open = int(rows[-1][0])
            if last_open + KLINE_MS <= cursor:
                break
            cursor = last_open + KLINE_MS
            time.sleep(0.1)  # stay far below public rate limits
        self.fetched_ranges[symbol].append((start_ms, end_ms))

    def _covered(self, symbol: str, start_ms: int, end_ms: int) -> bool:
        return any(s <= start_ms and end_ms <= e for s, e in self.fetched_ranges[symbol])

    def ensure(self, symbol: str, start_ms: int, end_ms: int):
        if not self._covered(symbol, start_ms, end_ms):
            self._fetch(symbol, start_ms, end_ms)

    def price_at(self, symbol: str, t_sec: int) -> Optional[float]:
        t_ms = int(t_sec) * 1000
        minute = ((t_ms + KLINE_MS - 1) // KLINE_MS) * KLINE_MS
        candles = self.cache.get(symbol, {})
        # Tolerate up to 2 missing candles (exchange maintenance gaps)
        for k in range(3):
            p = candles.get(minute + k * KLINE_MS)
            if p and p > 0:
                return p
        return None


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
