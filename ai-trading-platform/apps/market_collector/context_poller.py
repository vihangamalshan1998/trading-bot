"""
REST context shared by both market collectors (service.py and market_data.py).

The websocket streams give price, trades and the order book. Everything else the FeatureEngine
needs is polled here and written into each collector's per-symbol `raw` dict:
  every 60s : open_interest, mark_price_premium, funding_rate
  every 5m  : price_change_24h, volume_24h, ls_ratio (global long/short account ratio), ema_4h
and once at start-up, 1-minute klines to pre-load the FeatureEngine bar history (so multi-timeframe
indicators and the 1h volatility used by the exit policy are available immediately, not after hours).
"""
import asyncio
from typing import Dict, List, Callable

import aiohttp

from core.config.settings import settings
from core.logging.logger import logger

FAST_POLL_SECONDS = 60
SLOW_POLL_EVERY = 5  # slow endpoints every 5 fast cycles


def market_data_is_mainnet() -> bool:
    """Market data comes from the venue the bot trades on (consistent PnL / stops) unless overridden."""
    return bool(settings.market_data_mainnet or not settings.binance_testnet)


def market_data_rest_url() -> str:
    return "https://fapi.binance.com" if market_data_is_mainnet() else "https://testnet.binancefuture.com"


# Long/short ratio and 24h stats are only published by the mainnet data API
MAINNET_DATA_URL = "https://fapi.binance.com"


async def _get_json(session: aiohttp.ClientSession, url: str, params: Dict = None):
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        if resp.status != 200:
            return None
        return await resp.json()


async def fetch_1m_klines(symbol: str, limit: int = 1000) -> List[list]:
    try:
        async with aiohttp.ClientSession() as session:
            data = await _get_json(session, f"{market_data_rest_url()}/fapi/v1/klines",
                                   {"symbol": symbol, "interval": "1m", "limit": limit})
            return data or []
    except Exception as e:
        logger.warning(f"[{symbol}] Could not pre-load 1m klines: {e}")
        return []


async def seed_engines(engines: Dict[str, object]):
    """Pre-loads every FeatureEngine with ~16h of closed 1m bars."""
    for sym, engine in engines.items():
        klines = await fetch_1m_klines(sym)
        if klines:
            engine.seed_bars(klines)
        await asyncio.sleep(0.2)
    logger.info(f"Pre-loaded 1m bar history for {len(engines)} symbols.")


async def poll_context(symbols: List[str], raw: Dict[str, Dict], is_running: Callable[[], bool]):
    base = market_data_rest_url()
    cycle = 0
    async with aiohttp.ClientSession() as session:
        while is_running():
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
                        if "lastFundingRate" in pi:
                            r["funding_rate"] = float(pi["lastFundingRate"])

                    if cycle % SLOW_POLL_EVERY == 0:
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
