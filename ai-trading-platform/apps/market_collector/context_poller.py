"""
REST context shared by both market collectors (service.py and market_data.py).

The websocket streams give price, trades and the order book. Everything else the FeatureEngine
needs is polled here and written into each collector's per-symbol `raw` dict:
  every 60s : open_interest, mark_price_premium, funding_rate, next funding time,
              order-book depth bands (Binance mainnet, 1000 levels),
              cross-exchange funding / open interest (Bybit + OKX)
  every 5m  : price_change_24h, volume_24h, ls_ratio (global long/short account ratio), ema_4h,
              OKX funding rates
and once at start-up, 1-minute klines to pre-load the FeatureEngine bar history (so multi-timeframe
indicators and the 1h volatility used by the exit policy are available immediately, not after hours).
"""
import asyncio
import collections
import time
from typing import Dict, List, Callable

import aiohttp

from core.config.settings import settings
from core.logging.logger import logger

FAST_POLL_SECONDS = 60
SLOW_POLL_EVERY = 5  # slow endpoints every 5 fast cycles

BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"
OKX_OI_URL = "https://www.okx.com/api/v5/public/open-interest"
OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate"
OI_HISTORY_SECONDS = 3600


def market_data_is_mainnet() -> bool:
    """Market data comes from the venue the bot trades on (consistent PnL / stops) unless overridden."""
    return bool(settings.market_data_mainnet or not settings.binance_testnet)


def market_data_rest_url() -> str:
    return "https://fapi.binance.com" if market_data_is_mainnet() else "https://testnet.binancefuture.com"


# Long/short ratio, 24h stats and real order-book depth are only meaningful on the mainnet data API
MAINNET_DATA_URL = "https://fapi.binance.com"


def okx_inst_id(symbol: str) -> str:
    """BTCUSDT -> BTC-USDT-SWAP, 1000PEPEUSDT -> PEPE-USDT-SWAP (OKX lists the un-scaled coin)."""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    if base.startswith("1000"):
        base = base[4:]
    return f"{base}-USDT-SWAP"


def depth_features(bids: List[list], asks: List[list]) -> Dict[str, float]:
    """
    Order-book depth bands from a REST snapshot ([[price, qty], ...], best first).
      depth_imb_0p1pct : (bid - ask) / (bid + ask) notional within 0.1% of mid (every coin's book reaches it)
      depth_imb_1pct   : same within 1% (truncated to the snapshot range on very deep books like BTC)
      depth_bid_usd_1pct / depth_ask_usd_1pct : resting notional within 1% (log-scaled by the bot)
    """
    try:
        b = [(float(p), float(q)) for p, q in bids]
        a = [(float(p), float(q)) for p, q in asks]
        if not b or not a:
            return {}
        mid = (b[0][0] + a[0][0]) / 2.0
        if mid <= 0:
            return {}

        def band(levels, frac, side):
            lim = mid * (1 - frac) if side == "bid" else mid * (1 + frac)
            return sum(p * q for p, q in levels if (p >= lim if side == "bid" else p <= lim))

        out = {}
        for name, frac in (("0p1pct", 0.001), ("1pct", 0.01)):
            bn, an = band(b, frac, "bid"), band(a, frac, "ask")
            out[f"depth_imb_{name}"] = (bn - an) / (bn + an) if (bn + an) > 0 else 0.0
            if name == "1pct":
                out["depth_bid_usd_1pct"], out["depth_ask_usd_1pct"] = bn, an
        return out
    except Exception:
        return {}


def cross_exchange_features(binance_funding: float, other_funding: List[float],
                            oi_history: "collections.deque", oi_now: Dict[str, float], now: float) -> Dict[str, float]:
    """
    xex_funding_mean   : average funding across Binance / Bybit / OKX (crowding of the whole market)
    xex_funding_spread : other exchanges' average minus Binance (where the crowd is leaning harder)
    xex_oi_change_1h   : % change of total USD open interest over ~1h, using only exchanges present
                         at both points (an exchange dropping out never looks like an OI collapse)
    """
    out = {}
    rates = [binance_funding] + [f for f in other_funding if f is not None]
    out["xex_funding_mean"] = sum(rates) / len(rates)
    others = [f for f in other_funding if f is not None]
    out["xex_funding_spread"] = (sum(others) / len(others) - binance_funding) if others else 0.0

    if oi_now:
        oi_history.append((now, dict(oi_now)))
    while oi_history and now - oi_history[0][0] > OI_HISTORY_SECONDS * 1.5:
        oi_history.popleft()
    change = 0.0
    past = next((snap for ts, snap in oi_history if now - ts >= OI_HISTORY_SECONDS * 0.9), None)
    if past and oi_now:
        common = [k for k in oi_now if k in past and past[k] > 0]
        prev = sum(past[k] for k in common)
        if prev > 0:
            change = sum(oi_now[k] for k in common) / prev - 1.0
    out["xex_oi_change_1h"] = max(-1.0, min(1.0, change))
    return out


async def _get_json(session: aiohttp.ClientSession, url: str, params: Dict = None):
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        if resp.status != 200:
            return None
        return await resp.json()


async def fetch_klines(symbol: str, interval: str, limit: int) -> List[list]:
    """Klines from the market-data venue (no API key: the testnet trading key is not valid on mainnet)."""
    try:
        async with aiohttp.ClientSession() as session:
            data = await _get_json(session, f"{market_data_rest_url()}/fapi/v1/klines",
                                   {"symbol": symbol, "interval": interval, "limit": limit})
            return data or []
    except Exception as e:
        logger.warning(f"[{symbol}] Could not fetch {interval} klines: {e}")
        return []


async def fetch_1m_klines(symbol: str, limit: int = 1000) -> List[list]:
    return await fetch_klines(symbol, "1m", limit)


async def fetch_24hr_tickers() -> List[dict]:
    async with aiohttp.ClientSession() as session:
        return await _get_json(session, f"{market_data_rest_url()}/fapi/v1/ticker/24hr") or []


async def seed_engines(engines: Dict[str, object]):
    """Pre-loads every FeatureEngine with ~16h of closed 1m bars."""
    for sym, engine in engines.items():
        klines = await fetch_1m_klines(sym)
        if klines:
            engine.seed_bars(klines)
        await asyncio.sleep(0.2)
    logger.info(f"Pre-loaded 1m bar history for {len(engines)} symbols.")


async def _fetch_other_exchanges(session: aiohttp.ClientSession, symbols: List[str], okx_funding: Dict[str, float],
                                 refresh_okx_funding: bool) -> tuple:
    """(bybit {symbol: (funding, oi_usd)}, okx_oi {instId: oi_usd}); refreshes okx_funding in place."""
    bybit, okx_oi = {}, {}
    try:
        data = await _get_json(session, BYBIT_TICKERS_URL, {"category": "linear"})
        for t in ((data or {}).get("result") or {}).get("list") or []:
            if t.get("symbol") in symbols:
                bybit[t["symbol"]] = (float(t.get("fundingRate") or 0.0), float(t.get("openInterestValue") or 0.0))
    except Exception as e:
        logger.debug(f"Bybit context poll failed: {e}")
    try:
        data = await _get_json(session, OKX_OI_URL, {"instType": "SWAP"})
        for t in (data or {}).get("data") or []:
            if t.get("oiUsd"):
                okx_oi[t["instId"]] = float(t["oiUsd"])
    except Exception as e:
        logger.debug(f"OKX open-interest poll failed: {e}")
    if refresh_okx_funding:
        for sym in symbols:
            inst = okx_inst_id(sym)
            if inst not in okx_oi:
                continue
            try:
                data = await _get_json(session, OKX_FUNDING_URL, {"instId": inst})
                rows = (data or {}).get("data") or []
                if rows and rows[0].get("fundingRate") not in (None, ""):
                    okx_funding[sym] = float(rows[0]["fundingRate"])
            except Exception:
                pass
            await asyncio.sleep(0.12)  # OKX public limit: 20 requests / 2s
    return bybit, okx_oi


async def poll_context(symbols: List[str], raw: Dict[str, Dict], is_running: Callable[[], bool]):
    base = market_data_rest_url()
    cycle = 0
    okx_funding: Dict[str, float] = {}
    oi_history: Dict[str, collections.deque] = {sym: collections.deque() for sym in symbols}
    async with aiohttp.ClientSession() as session:
        while is_running():
            slow = cycle % SLOW_POLL_EVERY == 0
            bybit, okx_oi = await _fetch_other_exchanges(session, symbols, okx_funding, refresh_okx_funding=slow)
            for sym in symbols:
                r = raw.get(sym)
                if r is None:
                    continue
                try:
                    oi = await _get_json(session, f"{base}/fapi/v1/openInterest", {"symbol": sym})
                    if oi and "openInterest" in oi:
                        r["open_interest"] = float(oi["openInterest"])
                    pi = await _get_json(session, f"{base}/fapi/v1/premiumIndex", {"symbol": sym})
                    if pi:
                        mark, index = float(pi.get("markPrice", 0) or 0), float(pi.get("indexPrice", 0) or 0)
                        if index > 0:
                            r["mark_price_premium"] = (mark - index) / index
                        if mark > 0:
                            r["mark_price"] = mark
                        if "lastFundingRate" in pi:
                            r["funding_rate"] = float(pi["lastFundingRate"])
                        if pi.get("nextFundingTime"):
                            r["next_funding_ms"] = float(pi["nextFundingTime"])

                    depth = await _get_json(session, f"{MAINNET_DATA_URL}/fapi/v1/depth", {"symbol": sym, "limit": 1000})
                    if depth:
                        r.update(depth_features(depth.get("bids") or [], depth.get("asks") or []))

                    # Cross-exchange crowding (Binance + Bybit + OKX)
                    by = bybit.get(sym)
                    oi_now = {}
                    mark = float(r.get("mark_price", 0.0) or 0.0)
                    binance_oi = float(r.get("open_interest", 0.0) or 0.0)
                    if base != MAINNET_DATA_URL:
                        # testnet open interest is synthetic; cross-exchange totals need the real one
                        m_oi = await _get_json(session, f"{MAINNET_DATA_URL}/fapi/v1/openInterest", {"symbol": sym})
                        binance_oi = float(m_oi["openInterest"]) if m_oi and "openInterest" in m_oi else 0.0
                    if binance_oi > 0 and mark > 0:
                        oi_now["binance"] = binance_oi * mark
                    if by and by[1] > 0:
                        oi_now["bybit"] = by[1]
                    if okx_oi.get(okx_inst_id(sym)):
                        oi_now["okx"] = okx_oi[okx_inst_id(sym)]
                    r.update(cross_exchange_features(
                        float(r.get("funding_rate", 0.0) or 0.0),
                        [by[0] if by else None, okx_funding.get(sym)],
                        oi_history[sym], oi_now, time.time()))

                    if slow:
                        t24 = await _get_json(session, f"{base}/fapi/v1/ticker/24hr", {"symbol": sym})
                        if t24:
                            r["price_change_24h"] = float(t24.get("priceChangePercent", 0.0)) / 100.0
                            r["volume_24h"] = float(t24.get("quoteVolume", 0.0))
                        ls = await _get_json(session, f"{MAINNET_DATA_URL}/futures/data/globalLongShortAccountRatio",
                                             {"symbol": sym, "period": "5m", "limit": 1})
                        if ls:
                            r["ls_ratio"] = float(ls[0]["longShortRatio"])
                        k4 = await _get_json(session, f"{base}/fapi/v1/klines", {"symbol": sym, "interval": "4h", "limit": 20})
                        if k4:
                            closes = [float(k[4]) for k in k4]
                            r["ema_4h"] = sum(closes) / len(closes)  # SMA as a stable 4h trend anchor
                except Exception as e:
                    logger.warning(f"[{sym}] Context poll failed: {e}")
                await asyncio.sleep(0.1)
            cycle += 1
            await asyncio.sleep(FAST_POLL_SECONDS)
