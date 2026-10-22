"""
Pass 1 of the history replay: one coin, one UTC day -> the live FeatureEngine's 90 market features
and the whale tracker's 60s flow on the bot's frame grid (one frame every FRAME_SECONDS).

It re-creates the market collector's tick stream (apps/market_collector/market_data.py) from archive
files and feeds it into the unchanged FeatureEngine:
  every second  : last trade price, traded volume / taker-buy volume / trade count of that second,
                  estimated funding rate, mark premium, next funding time
  every 60 s    : open interest, depth bands, cross-exchange context   (context_poller fast cycle)
  every 300 s   : 24h change / quote volume, long/short ratio, 4h trend (context_poller slow cycle)

Differences from live, by necessity (no archive): best bid / ask (spread, top-of-book sizes, L2 skew) and
liquidations are absent; open interest is 5-minute data (interpolated with a 5-minute lag, never ahead);
the 0.1% depth band uses the archive's 0.2% band; Bybit / OKX are absent from the cross-exchange inputs.
Those slots are listed in REPLAY_APPROX_SLOTS / REPLAY_MISSING_SLOTS (apps/replay/layout.py).
"""
import collections
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np

from apps.feature_engine.engine import FeatureEngine
from apps.market_collector.context_poller import cross_exchange_features
from apps.replay import archive
from apps.whale_tracker.service import WHALE_THRESHOLDS, DEFAULT_WHALE_THRESHOLD

FRAME_SECONDS = 5.3          # measured live inference cycle (5 s sleep + work), 2026-10-22
WARMUP_SECONDS = 35 * 60     # fills the engine's 30-minute tick buffers before the first frame
SEED_BARS = 1000             # the collector pre-loads 1000 closed 1m bars at start-up
FAST_POLL, SLOW_POLL = 60, 300
WHALE_WINDOW = 60
WHALE_TRADE_CAP = 2_000_000.0
FRAMES_DIR = os.path.join(archive.CACHE_DIR, "frames")
FRAME_FILE_VERSION = 1

# Binance funding: interest 0.03% per day, premium clamp +-0.05% (per funding interval)
DAILY_INTEREST = 0.0003
FUNDING_CLAMP = 0.0005


def day_bounds(day: date) -> tuple:
    start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    return start, start + 86400


def frame_grid(day: date) -> np.ndarray:
    start, end = day_bounds(day)
    return start + np.arange(int((end - start) / FRAME_SECONDS) + 1) * FRAME_SECONDS


def _concat(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return np.concatenate(parts)


class _MinuteSeries:
    """1-minute kline rows keyed by minute index (epoch // 60)."""
    def __init__(self, rows: Optional[np.ndarray]):
        if rows is None or len(rows) == 0:
            self.first, self.rows = 0, np.zeros((0, 8))
            return
        self.first = int(rows[0, 0] // 60000)
        last = int(rows[-1, 0] // 60000)
        full = np.full((last - self.first + 1, rows.shape[1]), np.nan)
        full[(rows[:, 0] // 60000).astype(np.int64) - self.first] = rows
        # carry forward through missing minutes (exchange maintenance)
        for i in range(1, len(full)):
            if np.isnan(full[i, 1]):
                full[i, 1:] = full[i - 1, 1:]
        self.rows = full

    def get(self, minute: int, col: int) -> float:
        i = minute - self.first
        if i < 0 or i >= len(self.rows):
            return float("nan")
        return float(self.rows[i, col])

    def window(self, minute_from: int, minute_to: int) -> np.ndarray:
        """Rows for minutes [minute_from, minute_to)."""
        a, b = max(0, minute_from - self.first), max(0, minute_to - self.first)
        return self.rows[a:b]


class SymbolDayReplay:
    def __init__(self, symbol: str, day: date):
        self.symbol, self.day = symbol, day
        self.day_start, self.day_end = day_bounds(day)
        self.tick_start = self.day_start - WARMUP_SECONDS

    # ------------------------------------------------------------------ data
    def load(self) -> bool:
        sym, d = self.symbol, self.day
        prev = d - timedelta(days=1)
        trades = _concat([archive.load_agg_trades(sym, prev), archive.load_agg_trades(sym, d)])
        if trades is None or len(trades["t"]) == 0:
            return False
        lo = (self.tick_start - 1) * 1000
        keep = (trades["t"] > lo) & (trades["t"] <= self.day_end * 1000)
        self.trades = {k: v[keep] for k, v in trades.items()}
        if len(self.trades["t"]) == 0:
            return False

        # 1m klines: 1000 seed bars + 24h stats + 20 4h candles need ~4 days of history
        kdays = [d - timedelta(days=i) for i in range(4, -1, -1)]
        self.k1m = _MinuteSeries(_concat([archive.load_klines("klines", sym, x) for x in kdays]))
        self.mark = _MinuteSeries(_concat([archive.load_klines("markPriceKlines", sym, x) for x in (prev, d)]))
        self.index = _MinuteSeries(_concat([archive.load_klines("indexPriceKlines", sym, x) for x in (prev, d)]))
        self.premium = _MinuteSeries(_concat([archive.load_klines("premiumIndexKlines", sym, x) for x in (prev, d)]))
        self.metrics = _concat([archive.load_metrics(sym, prev), archive.load_metrics(sym, d)])
        self.depth = _concat([archive.load_book_depth(sym, prev), archive.load_book_depth(sym, d)])
        self.funding_times = archive.funding_times(sym, (self.day_start - 2 * 86400) * 1000, (self.day_end + 86400) * 1000)
        return len(self.k1m.rows) > 0

    # ------------------------------------------------------------- context
    def _funding_period(self, t: int) -> tuple:
        """(start, end) epoch s of the funding interval containing t."""
        ft = self.funding_times // 1000
        i = int(np.searchsorted(ft, t, side="right"))
        if 0 < i < len(ft):
            return int(ft[i - 1]), int(ft[i])
        step = int(ft[-1] - ft[-2]) if len(ft) >= 2 else 8 * 3600
        if i == 0:
            return int(ft[0]) - step, int(ft[0])
        n = (t - int(ft[-1])) // step
        start = int(ft[-1]) + n * step
        return start, start + step

    def _estimated_funding(self, t: int) -> float:
        """Binance's running funding estimate: linearly weighted average premium + clamped interest."""
        p_start, p_end = self._funding_period(t)
        minute = t // 60
        prem = self.premium.window(p_start // 60, minute)[:, 4] if minute > p_start // 60 else np.zeros(0)
        prem = prem[~np.isnan(prem)]
        interest = DAILY_INTEREST * (p_end - p_start) / 86400.0
        if len(prem) == 0:
            return interest
        w = np.arange(1, len(prem) + 1, dtype=np.float64)
        avg = float(np.dot(prem, w) / w.sum())
        return float(np.clip(avg + np.clip(interest - avg, -FUNDING_CLAMP, FUNDING_CLAMP), -0.02, 0.02))

    def _open_interest(self, t: int) -> float:
        """5-minute OI, linearly interpolated with a 5-minute lag (never uses a value from the future)."""
        m = self.metrics
        if m is None or len(m["t"]) == 0:
            return 0.0
        i = int(np.searchsorted(m["t"], t, side="right")) - 1
        if i < 0:
            return 0.0
        if i == 0:
            return float(m["oi"][0])
        span = float(m["t"][i] - m["t"][i - 1]) or 300.0
        frac = min(1.0, (t - m["t"][i]) / span)
        return float(m["oi"][i - 1] + (m["oi"][i] - m["oi"][i - 1]) * frac)

    def _ls_ratio(self, t: int) -> Optional[float]:
        m = self.metrics
        if m is None:
            return None
        i = int(np.searchsorted(m["t"], t, side="right")) - 1
        return float(m["ls"][i]) if i >= 0 and np.isfinite(m["ls"][i]) else None

    def _depth(self, t: int) -> Dict[str, float]:
        d = self.depth
        if d is None or len(d["t"]) == 0:
            return {}
        i = int(np.searchsorted(d["t"], t, side="right"))
        if i == 0:
            return {}
        ts = d["t"][i - 1]
        if t - ts > 300:
            return {}
        j = int(np.searchsorted(d["t"], ts, side="left"))
        snap = dict(zip(d["pct"][j:i], d["notional"][j:i]))
        out = {}
        for name, band in (("0p1pct", 0.2), ("1pct", 1.0)):
            bn, an = snap.get(-band, 0.0), snap.get(band, 0.0)
            out[f"depth_imb_{name}"] = (bn - an) / (bn + an) if (bn + an) > 0 else 0.0
            if name == "1pct":
                out["depth_bid_usd_1pct"], out["depth_ask_usd_1pct"] = bn, an
        return out

    def _slow_context(self, t: int, last_price: float) -> Dict[str, float]:
        out = {}
        minute = t // 60
        open_24h = self.k1m.get(minute - 1440, 1)
        if last_price > 0 and open_24h and np.isfinite(open_24h):
            out["price_change_24h"] = (last_price - open_24h) / open_24h
        qv = self.k1m.window(minute - 1440, minute)[:, 6]
        if len(qv):
            out["volume_24h"] = float(np.nansum(qv))
        ls = self._ls_ratio(t)
        if ls is not None:
            out["ls_ratio"] = ls
        # 20 4h candles, the last one still open (its close is the current price)
        cur_4h = (t // 14400) * 14400
        closes = []
        for j in range(19, 0, -1):
            c = self.k1m.get((cur_4h - (j - 1) * 14400) // 60 - 1, 4)
            if np.isfinite(c):
                closes.append(c)
        if last_price > 0:
            closes.append(last_price)
        if closes:
            out["ema_4h"] = sum(closes) / len(closes)
        return out

    # ----------------------------------------------------------------- run
    def run(self) -> Optional[Dict[str, np.ndarray]]:
        if not self.load():
            return None
        sym = self.symbol
        engine = FeatureEngine(sym)
        seed_from = self.tick_start // 60 - SEED_BARS
        seed = self.k1m.window(seed_from, self.tick_start // 60)
        engine.seed_bars([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in seed if np.isfinite(r[1])], now=self.tick_start)

        tr = self.trades
        seconds = np.arange(self.tick_start, self.day_end + 1, dtype=np.int64)
        idx = np.searchsorted(tr["t"], seconds * 1000, side="right")  # trades with t <= s*1000
        cum_q = np.concatenate([[0.0], np.cumsum(tr["q"])])
        cum_buy = np.concatenate([[0.0], np.cumsum(np.where(tr["buyer_maker"], 0.0, tr["q"]))])

        # whale trades (USD >= threshold, capped) for the 60s flow window
        thresh = WHALE_THRESHOLDS.get(sym, DEFAULT_WHALE_THRESHOLD)["trade"]
        usd = np.minimum(tr["p"] * tr["q"], WHALE_TRADE_CAP)
        wmask = usd >= thresh
        w_t = tr["t"][wmask]
        w_buy = np.concatenate([[0.0], np.cumsum(np.where(tr["buyer_maker"][wmask], 0.0, usd[wmask]))])
        w_sell = np.concatenate([[0.0], np.cumsum(np.where(tr["buyer_maker"][wmask], usd[wmask], 0.0))])

        grid = frame_grid(self.day)
        frames = np.zeros((len(grid), 90), dtype=np.float32)
        whale = np.zeros((len(grid), 2), dtype=np.float64)
        price_out = np.zeros(len(grid), dtype=np.float64)
        frame_sec = np.floor(grid).astype(np.int64)
        fi = 0

        raw = {"mid_price": 0.0, "volume": 0.0, "buy_volume": 0.0, "trade_count": 0.0, "best_bid": 0.0,
               "best_ask": 0.0, "bid_qty": 0.0, "ask_qty": 0.0, "funding_rate": 0.0, "liquidation_volume": 0.0,
               "open_interest": 0.0, "ls_ratio": 1.0, "ema_4h": 0.0, "ob_skew_l2": 0.0,
               "price_change_24h": 0.0, "volume_24h": 0.0, "mark_price_premium": 0.0}
        oi_history = collections.deque()
        last_price = 0.0
        data_end = int(tr["t"][-1] // 1000)  # a partial (today's) day ends here: no frames past the data
        for n, s in enumerate(seconds):
            s = int(s)
            if s > data_end:
                break
            hi = idx[n]
            lo = idx[n - 1] if n > 0 else int(np.searchsorted(tr["t"], (s - 1) * 1000, side="right"))
            if hi > 0:
                last_price = float(tr["p"][hi - 1])
            if last_price <= 0:
                continue
            raw["mid_price"] = last_price
            raw["best_bid"] = raw["best_ask"] = last_price  # no book archive: zero spread (flagged slot)
            raw["volume"] = float(cum_q[hi] - cum_q[lo])
            raw["buy_volume"] = float(cum_buy[hi] - cum_buy[lo])
            raw["trade_count"] = float(hi - lo)

            minute = s // 60
            mark, index = self.mark.get(minute - 1, 4), self.index.get(minute - 1, 4)
            if np.isfinite(mark) and np.isfinite(index) and index > 0:
                raw["mark_price_premium"] = (mark - index) / index
            raw["funding_rate"] = self._estimated_funding(s)
            raw["next_funding_ms"] = float(self._funding_period(s)[1] * 1000)

            since = s - self.tick_start
            if since % FAST_POLL == 0:
                raw["open_interest"] = self._open_interest(s)
                raw.update(self._depth(s))
                oi_now = {"binance": raw["open_interest"] * mark} if raw["open_interest"] > 0 and np.isfinite(mark) else {}
                raw.update(cross_exchange_features(raw["funding_rate"], [None, None], oi_history, oi_now, float(s)))
            if since % SLOW_POLL == 0:
                raw.update(self._slow_context(s, last_price))

            engine.add_tick(raw, ts=float(s))

            while fi < len(grid) and frame_sec[fi] == s:
                frames[fi] = engine.compute_features()
                a = int(np.searchsorted(w_t, (s - WHALE_WINDOW) * 1000, side="right"))
                b = int(np.searchsorted(w_t, s * 1000, side="right"))
                whale[fi] = (min(1e7, w_buy[b] - w_buy[a]), min(1e7, w_sell[b] - w_sell[a]))
                price_out[fi] = last_price
                fi += 1
        valid = np.zeros(len(grid), dtype=bool)
        valid[:fi] = price_out[:fi] > 0
        return {"t": grid, "features": frames, "whale": whale, "price": price_out, "valid": valid}


def frame_path(symbol: str, day: date) -> str:
    return os.path.join(FRAMES_DIR, symbol, f"{symbol}-{day.isoformat()}-v{FRAME_FILE_VERSION}.npz")


def build_day(symbol: str, day: date, force: bool = False) -> Optional[str]:
    """Runs pass 1 for one coin-day and caches the result. Returns the file path (None = no data)."""
    path = frame_path(symbol, day)
    partial_day = day_bounds(day)[1] > time.time() - 3600  # today's file grows: always rebuild it
    if os.path.exists(path) and not force and not partial_day:
        return path
    out = SymbolDayReplay(symbol, day).run()
    if out is None:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return path


def load_day(symbol: str, day: date) -> Optional[Dict[str, np.ndarray]]:
    path = frame_path(symbol, day)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
