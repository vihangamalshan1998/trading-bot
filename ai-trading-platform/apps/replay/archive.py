"""
Downloads and loads Binance USD-M futures history from the public archive (data.binance.vision).

Daily files used by the replay (all mainnet, UTC days):
  aggTrades            every trade (price, qty, taker side)           -> prices, volume, flow, whale trades
  klines 1m            OHLCV + quote / taker-buy volume                -> bar seeding, 24h stats, 4h trend, grading
  markPriceKlines 1m   mark price                                      -> mark premium
  indexPriceKlines 1m  index price                                     -> mark premium
  premiumIndexKlines 1m premium index                                  -> estimated funding rate
  metrics              open interest + long/short ratios every 5 min
  bookDepth            cumulative depth at +-0.2% / 1-5% every ~30 s

Not in the archive: best bid / ask (bookTicker stopped 2024-03-30), liquidations.
Files are cached under data/replay_cache/ and only downloaded once.
"""
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://data.binance.vision/data/futures/um/daily"
REST_URL = "https://fapi.binance.com"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "replay_cache")

KLINE_KINDS = ("klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines")
DAILY_KINDS = ("aggTrades", "metrics", "bookDepth") + KLINE_KINDS


def day_range(start: date, end: date) -> List[date]:
    """Inclusive list of UTC days."""
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _file_name(kind: str, symbol: str, day: date) -> Tuple[str, str]:
    """(url, local path) of one daily archive file."""
    d = day.isoformat()
    if kind in KLINE_KINDS:
        name = f"{symbol}-1m-{d}.zip"
        url = f"{ARCHIVE_URL}/{kind}/{symbol}/1m/{name}"
    else:
        name = f"{symbol}-{kind}-{d}.zip"
        url = f"{ARCHIVE_URL}/{kind}/{symbol}/{name}"
    return url, os.path.join(CACHE_DIR, kind, symbol, name)


def download(kind: str, symbol: str, day: date, retries: int = 3) -> Optional[str]:
    """Local path of the file (downloaded if missing), or None when the archive has no such file."""
    url, path = _file_name(kind, symbol, day)
    if os.path.exists(path):
        return path
    missing_marker = path + ".missing"
    if os.path.exists(missing_marker):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Not published (coin not listed yet, or the archive is a day behind). Only remember
                # misses for days that are surely final, so a late upload is fetched next time.
                if day < (datetime.now(timezone.utc).date() - timedelta(days=3)):
                    open(missing_marker, "w").close()
                return None
            if attempt == retries - 1:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
        time.sleep(2.0 * (attempt + 1))
    return None


def download_all(symbols: Iterable[str], days: Iterable[date], kinds: Iterable[str] = DAILY_KINDS,
                 workers: int = 8, log=print) -> int:
    """Downloads every (kind, symbol, day) file in parallel. Returns the number of files available."""
    jobs = [(k, s, d) for s in symbols for d in days for k in kinds]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, path in enumerate(pool.map(lambda j: download(*j), jobs), 1):
            done += path is not None
            if log and (i % 200 == 0 or i == len(jobs)):
                log(f"archive: {i}/{len(jobs)} files checked, {done} available")
    return done


def _read_csv(path: str, columns: List[str]) -> pd.DataFrame:
    """CSV inside a daily zip. Older files have no header row; newer ones do."""
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:200].split(b"\n", 1)[0]
    has_header = not first[:1].isdigit() and not first[:1] == b"-"
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    df = df.iloc[:, :len(columns)]
    df.columns = columns
    return df


def _ms(values: np.ndarray) -> np.ndarray:
    """Epoch milliseconds (some newer archives use microseconds)."""
    v = values.astype(np.int64)
    return np.where(v > 10**14, v // 1000, v)


# ----------------------------------------------------------------------------------------------
# REST fallback for the last few days (the archive publishes a day only after it ends, sometimes
# a day later). Same output structures as the archive loaders. bookDepth has no REST history.
# ----------------------------------------------------------------------------------------------
REST_FALLBACK_DAYS = 3
REST_CACHE_DIR = os.path.join(CACHE_DIR, "rest")
_last_rest_call = [0.0]


def _rest_fallback_allowed(day: date) -> bool:
    return day >= datetime.now(timezone.utc).date() - timedelta(days=REST_FALLBACK_DAYS)


def _rest_get(path: str, params: Dict, min_gap: float = 0.25):
    query = "&".join(f"{k}={v}" for k, v in params.items())
    for attempt in range(5):
        wait = _last_rest_call[0] + min_gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_rest_call[0] = time.time()
        try:
            with urllib.request.urlopen(f"{REST_URL}{path}?{query}", timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                time.sleep(30 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"REST {path} failed after retries")


def _day_ms(day: date) -> Tuple[int, int]:
    start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
    return start, min(start + 86400_000, int(time.time() * 1000))


PARTIAL_CACHE_SECONDS = 3 * 3600


def _rest_cached(name: str, day: date, fetch):
    """
    Caches REST results. A complete day is kept for good; a partial (still running) day is reused for
    PARTIAL_CACHE_SECONDS, so one validation run fetches it once and later runs see the newer data.
    """
    complete = time.time() * 1000 > _day_ms(day)[0] + 86400_000 + 600_000
    path = os.path.join(REST_CACHE_DIR, f"{name}-{day.isoformat()}{'' if complete else '-partial'}.npz")
    if os.path.exists(path) and (complete or time.time() - os.path.getmtime(path) < PARTIAL_CACHE_SECONDS):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    out = fetch()
    if out is not None:
        os.makedirs(REST_CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
    return out


def _rest_agg_trades(symbol: str, day: date, since_ms: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
    start, end = _day_ms(day)
    if since_ms is not None:
        start = max(start, int(since_ms))
    rows: List[dict] = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + 3600_000 - 1, end)
        batch = _rest_get("/fapi/v1/aggTrades", {"symbol": symbol, "startTime": window_start,
                                                  "endTime": window_end, "limit": 1000}, min_gap=0.5)
        while batch:
            rows.extend(batch)
            if len(batch) < 1000 or int(batch[-1]["T"]) >= window_end:
                break
            batch = _rest_get("/fapi/v1/aggTrades", {"symbol": symbol, "fromId": int(batch[-1]["a"]) + 1,
                                                      "limit": 1000}, min_gap=0.5)
            batch = [r for r in batch if int(r["T"]) <= window_end]
        window_start = window_end + 1
    if not rows:
        return None
    seen, uniq = set(), []
    for r in rows:
        if r["a"] not in seen:
            seen.add(r["a"])
            uniq.append(r)
    uniq.sort(key=lambda r: (int(r["T"]), int(r["a"])))
    return {"t": np.array([int(r["T"]) for r in uniq], dtype=np.int64),
            "p": np.array([float(r["p"]) for r in uniq]), "q": np.array([float(r["q"]) for r in uniq]),
            "buyer_maker": np.array([bool(r["m"]) for r in uniq])}


_KLINE_ENDPOINTS = {"klines": ("/fapi/v1/klines", "symbol"), "markPriceKlines": ("/fapi/v1/markPriceKlines", "symbol"),
                    "indexPriceKlines": ("/fapi/v1/indexPriceKlines", "pair"),
                    "premiumIndexKlines": ("/fapi/v1/premiumIndexKlines", "symbol")}


def _rest_klines(kind: str, symbol: str, day: date) -> Optional[np.ndarray]:
    path, key = _KLINE_ENDPOINTS[kind]
    start, end = _day_ms(day)
    rows = []
    cursor = start
    while cursor < end:
        batch = _rest_get(path, {key: symbol, "interval": "1m", "startTime": cursor, "endTime": end - 1, "limit": 1500})
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 60_000
        if len(batch) < 1500:
            break
    now_ms = time.time() * 1000
    rows = [r for r in rows if int(r[6]) < now_ms]  # closed candles only
    if not rows:
        return None
    arr = np.array([[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[7]), float(r[9])]
                    for r in rows])
    return arr[np.argsort(arr[:, 0])]


def _rest_metrics(symbol: str, day: date) -> Optional[Dict[str, np.ndarray]]:
    start, end = _day_ms(day)
    oi = _rest_get("/futures/data/openInterestHist", {"symbol": symbol, "period": "5m", "startTime": start,
                                                      "endTime": end, "limit": 500})
    ls = _rest_get("/futures/data/globalLongShortAccountRatio", {"symbol": symbol, "period": "5m", "startTime": start,
                                                                 "endTime": end, "limit": 500})
    if not oi:
        return None
    ls_by_t = {int(r["timestamp"]): float(r["longShortRatio"]) for r in (ls or [])}
    t = np.array([int(r["timestamp"]) // 1000 for r in oi], dtype=np.int64)
    out = {"t": t, "oi": np.array([float(r["sumOpenInterest"]) for r in oi]),
           "ls": np.array([ls_by_t.get(int(r["timestamp"]), np.nan) for r in oi])}
    order = np.argsort(out["t"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def load_agg_trades(symbol: str, day: date, since_ms: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    {'t': epoch ms, 'p': price, 'q': qty, 'buyer_maker': bool} sorted by time. `since_ms` only limits the
    REST fallback (the replay needs just the last minutes of the previous day); archive files are read whole.
    """
    path = download("aggTrades", symbol, day)
    if not path:
        if _rest_fallback_allowed(day):
            name = f"aggTrades-{symbol}" + (f"-from{int(since_ms)}" if since_ms is not None else "")
            return _rest_cached(name, day, lambda: _rest_agg_trades(symbol, day, since_ms))
        return None
    df = _read_csv(path, ["id", "p", "q", "first", "last", "t", "m"])
    m = df["m"]
    buyer_maker = (m.astype(str).str.lower() == "true").to_numpy() if m.dtype == object else m.to_numpy(dtype=bool)
    out = {"t": _ms(df["t"].to_numpy()), "p": df["p"].to_numpy(np.float64), "q": df["q"].to_numpy(np.float64),
           "buyer_maker": buyer_maker}
    order = np.argsort(out["t"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def load_klines(kind: str, symbol: str, day: date) -> Optional[np.ndarray]:
    """Rows [open_ms, open, high, low, close, volume, quote_volume, taker_buy_base_volume]."""
    path = download(kind, symbol, day)
    if not path:
        if _rest_fallback_allowed(day):
            out = _rest_cached(f"{kind}-{symbol}", day, lambda: {"rows": r} if (r := _rest_klines(kind, symbol, day)) is not None else None)
            return out["rows"] if out else None
        return None
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
            "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    df = _read_csv(path, cols)
    arr = np.stack([_ms(df["open_time"].to_numpy()).astype(np.float64)] +
                   [df[c].to_numpy(np.float64) for c in ("open", "high", "low", "close", "volume", "quote_volume",
                                                         "taker_buy_volume")], axis=1)
    return arr[np.argsort(arr[:, 0])]


def _epoch_s(text: pd.Series) -> np.ndarray:
    # Unit-agnostic (pandas 3 may store seconds / microseconds instead of nanoseconds)
    dt = pd.to_datetime(text, utc=True)
    return ((dt - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)


def load_metrics(symbol: str, day: date) -> Optional[Dict[str, np.ndarray]]:
    """{'t': epoch s, 'oi': open interest (coins), 'ls': global long/short ACCOUNT ratio}."""
    path = download("metrics", symbol, day)
    if not path:
        if _rest_fallback_allowed(day):
            return _rest_cached(f"metrics-{symbol}", day, lambda: _rest_metrics(symbol, day))
        return None
    df = _read_csv(path, ["create_time", "symbol", "oi", "oi_value", "top_acc_ls", "top_pos_ls", "ls", "taker_ls"])
    out = {"t": _epoch_s(df["create_time"]), "oi": df["oi"].to_numpy(np.float64), "ls": df["ls"].to_numpy(np.float64)}
    order = np.argsort(out["t"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def load_book_depth(symbol: str, day: date) -> Optional[Dict[str, np.ndarray]]:
    """{'t': epoch s, 'pct': signed band in percent (-5..5), 'notional': cumulative USD within the band}."""
    path = download("bookDepth", symbol, day)
    if not path:
        return None
    df = _read_csv(path, ["timestamp", "pct", "depth", "notional"])
    out = {"t": _epoch_s(df["timestamp"]), "pct": df["pct"].to_numpy(np.float64),
           "notional": df["notional"].to_numpy(np.float64)}
    order = np.argsort(out["t"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def funding_times(symbol: str, start_ms: int, end_ms: int) -> np.ndarray:
    """Funding settlement times (epoch ms) from the REST history (cached per symbol)."""
    path = os.path.join(CACHE_DIR, "fundingRate", f"{symbol}.json")
    rows: List[dict] = []
    if os.path.exists(path):
        with open(path) as f:
            rows = json.load(f)
    have_to = max((int(r["fundingTime"]) for r in rows), default=0)
    have_from = min((int(r["fundingTime"]) for r in rows), default=0)
    if not rows or have_from > start_ms or have_to < end_ms:
        cursor = start_ms - 2 * 86400_000
        fetched: Dict[int, dict] = {int(r["fundingTime"]): r for r in rows}
        while cursor < end_ms + 86400_000:
            url = f"{REST_URL}/fapi/v1/fundingRate?symbol={symbol}&startTime={cursor}&limit=1000"
            with urllib.request.urlopen(url, timeout=30) as resp:
                batch = json.load(resp)
            if not batch:
                break
            for r in batch:
                fetched[int(r["fundingTime"])] = r
            last = int(batch[-1]["fundingTime"])
            if last <= cursor or len(batch) < 1000:
                break
            cursor = last + 1
            time.sleep(0.2)
        rows = [fetched[k] for k in sorted(fetched)]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(rows, f)
    # Settlement times are on the hour; the API reports a few ms late
    return np.array(sorted({(int(r["fundingTime"]) + 30_000) // 60_000 * 60_000 for r in rows}), dtype=np.int64)
