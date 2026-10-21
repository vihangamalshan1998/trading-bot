"""
Market-wide signals that are the same for every coin, published once a minute to the Redis key
`market:global_context` (read by the trading bot into state slots 160-167, core/schemas/state_layout.py):

  coinbase_premium_btc / _eth : (Coinbase USD price - Binance USDT spot price) / Binance price.
                                US institutional demand; it often leads Binance moves.
  dvol_btc / dvol_eth         : Deribit implied-volatility index / 100 (options market's expected move).
  dvol_btc_change_1h          : 1h relative change of BTC DVOL (a "storm warning" when it jumps).
  econ_time_to_next           : hours until the next high-impact US economic release / 24 (1.0 = none within 24h)
  econ_time_since_last        : hours since the last one / 24 (1.0 = more than 24h ago)
  econ_next_importance        : weight of the biggest release in the next 24h (Fed 1.0, CPI / jobs 0.8,
                                other high-impact 0.5, medium 0.2, none 0). Crypto often jumps 1-3% in minutes
                                around these, which hits stops; the model can learn to stand aside.

All sources are free public endpoints; a failed source keeps its last value (or 0 before the first read).
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp

from core.logging.logger import logger

GLOBAL_CONTEXT_KEY = "market:global_context"
POLL_SECONDS = 60
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CALENDAR_REFRESH_SECONDS = 3600  # the feed is rate limited; it changes rarely
CALENDAR_HORIZON_H = 24.0


def event_importance(title: str, impact: str) -> float:
    t = (title or "").lower()
    if impact == "High":
        if "fomc" in t or "federal funds" in t or "fed chair" in t:
            return 1.0
        if "cpi" in t or "non-farm" in t or "nonfarm" in t or "pce" in t:
            return 0.8
        return 0.5
    if impact == "Medium":
        return 0.2
    return 0.0


def parse_calendar(rows: List[dict]) -> List[tuple]:
    """US releases as (epoch seconds, importance), medium impact and above."""
    out = []
    for e in rows or []:
        if e.get("country") != "USD":
            continue
        w = event_importance(e.get("title", ""), e.get("impact", ""))
        if w <= 0:
            continue
        try:
            out.append((datetime.fromisoformat(e["date"]).timestamp(), w))
        except Exception:
            continue
    return sorted(out)


def econ_calendar_features(events: List[tuple], now: float) -> Dict[str, float]:
    """Time to / since high-impact releases (importance >= 0.5) and the biggest release ahead in 24h."""
    horizon = CALENDAR_HORIZON_H * 3600.0
    major = [(t, w) for t, w in events if w >= 0.5]
    nxt = [t for t, _ in major if t >= now]
    prev = [t for t, _ in major if t < now]
    ahead = [w for t, w in events if now <= t <= now + horizon]
    return {
        "econ_time_to_next": min(1.0, (min(nxt) - now) / horizon) if nxt else 1.0,
        "econ_time_since_last": min(1.0, (now - max(prev)) / horizon) if prev else 1.0,
        "econ_next_importance": max(ahead) if ahead else 0.0,
    }


async def _get_json(session: aiohttp.ClientSession, url: str, params: Dict = None):
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15),
                           headers={"User-Agent": "Mozilla/5.0"}) as resp:
        if resp.status != 200:
            return None
        return await resp.json(content_type=None)


async def _coinbase_premium(session: aiohttp.ClientSession, coin: str) -> Optional[float]:
    cb = await _get_json(session, f"https://api.exchange.coinbase.com/products/{coin}-USD/ticker")
    bn = await _get_json(session, "https://api.binance.com/api/v3/ticker/price", {"symbol": f"{coin}USDT"})
    if not cb or not bn:
        return None
    cb_px, bn_px = float(cb.get("price") or 0), float(bn.get("price") or 0)
    return (cb_px - bn_px) / bn_px if cb_px > 0 and bn_px > 0 else None


async def _dvol(session: aiohttp.ClientSession, coin: str) -> Optional[tuple]:
    """(latest DVOL, DVOL ~1h ago) from 1-minute candles."""
    now_ms = int(time.time() * 1000)
    data = await _get_json(session, "https://www.deribit.com/api/v2/public/get_volatility_index_data",
                           {"currency": coin, "start_timestamp": now_ms - 2 * 3600 * 1000,
                            "end_timestamp": now_ms, "resolution": 60})
    rows = ((data or {}).get("result") or {}).get("data") or []
    if not rows:
        return None
    last = float(rows[-1][4])
    hour_ago = float(rows[-61][4]) if len(rows) > 60 else float(rows[0][4])
    return last, hour_ago


async def poll_global_context(redis_conn, is_running=lambda: True):
    state = {"coinbase_premium_btc": 0.0, "coinbase_premium_eth": 0.0,
             "dvol_btc": 0.0, "dvol_eth": 0.0, "dvol_btc_change_1h": 0.0}
    events: Dict[float, float] = {}  # epoch -> importance; kept across weekly feed rollovers
    last_calendar = 0.0
    async with aiohttp.ClientSession() as session:
        while is_running():
            if time.time() - last_calendar > CALENDAR_REFRESH_SECONDS:
                try:
                    rows = await _get_json(session, CALENDAR_URL)
                    if rows:
                        last_calendar = time.time()
                        events.update(dict(parse_calendar(rows)))
                        cutoff = time.time() - 7 * 86400
                        events = {t: w for t, w in events.items() if t >= cutoff}
                except Exception as e:
                    logger.debug(f"Economic calendar poll failed: {e}")
            state.update(econ_calendar_features(sorted(events.items()), time.time()))
            for coin in ("BTC", "ETH"):
                try:
                    prem = await _coinbase_premium(session, coin)
                    if prem is not None:
                        state[f"coinbase_premium_{coin.lower()}"] = prem
                except Exception as e:
                    logger.debug(f"Coinbase premium poll failed for {coin}: {e}")
                try:
                    dv = await _dvol(session, coin)
                    if dv:
                        state[f"dvol_{coin.lower()}"] = dv[0] / 100.0
                        if coin == "BTC" and dv[1] > 0:
                            state["dvol_btc_change_1h"] = dv[0] / dv[1] - 1.0
                except Exception as e:
                    logger.debug(f"DVOL poll failed for {coin}: {e}")
            state["timestamp"] = time.time()
            try:
                if redis_conn is not None:
                    await redis_conn.set(GLOBAL_CONTEXT_KEY, json.dumps(state), ex=600)
            except Exception as e:
                logger.debug(f"Global context publish failed: {e}")
            await asyncio.sleep(POLL_SECONDS)
