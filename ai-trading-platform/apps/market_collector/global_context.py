"""
Market-wide signals that are the same for every coin, published once a minute to the Redis key
`market:global_context` (read by the trading bot into state slots 160-164, core/schemas/state_layout.py):

  coinbase_premium_btc / _eth : (Coinbase USD price - Binance USDT spot price) / Binance price.
                                US institutional demand; it often leads Binance moves.
  dvol_btc / dvol_eth         : Deribit implied-volatility index / 100 (options market's expected move).
  dvol_btc_change_1h          : 1h relative change of BTC DVOL (a "storm warning" when it jumps).

All sources are free public endpoints; a failed source keeps its last value (or 0 before the first read).
"""
import asyncio
import json
import time
from typing import Dict, Optional

import aiohttp

from core.logging.logger import logger

GLOBAL_CONTEXT_KEY = "market:global_context"
POLL_SECONDS = 60


async def _get_json(session: aiohttp.ClientSession, url: str, params: Dict = None):
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15),
                           headers={"User-Agent": "Mozilla/5.0"}) as resp:
        if resp.status != 200:
            return None
        return await resp.json()


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
    async with aiohttp.ClientSession() as session:
        while is_running():
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
