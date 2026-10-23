"""
Why the trading bot cannot open new trades right now, read from the state it saves in Redis
(apps/trading_bot/main.py: ENTRY_GUARD_KEY risk state, dashboard:portfolio, system:pause_until).
Shared by the dashboard API and the Telegram bot, which both used to say "trading" whenever the kill switch
was off, even while the daily loss limit blocked every entry.
"""
import json
import time
from typing import Optional

from core.config.settings import settings
from core.risk.risk_manager import DRAWDOWN_WINDOW_DAYS


async def _json(redis, key: str) -> dict:
    try:
        raw = await redis.get(key)
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


async def entry_block(redis) -> Optional[dict]:
    """{"reason", "until", "all": [...]} for the block that lasts longest, or None when entries are allowed."""
    if redis is None:
        return None
    now = time.time()
    risk = (await _json(redis, "bot:entry_guard")).get("risk") or {}
    equity = float((await _json(redis, "dashboard:portfolio")).get("equity") or 0.0)
    blocks = []
    try:
        pause = float((await redis.get("system:pause_until")) or 0.0)
    except (TypeError, ValueError):
        pause = 0.0
    if pause > now:
        blocks.append(("Paused (Telegram or loss-velocity airbag)", pause))
    cb = float(risk.get("circuit_breaker_active_until") or 0.0)
    if cb > now:
        blocks.append(("Circuit breaker (5% daily loss)", cb))
    day_high = float(risk.get("rm_daily_high") or 0.0)
    if equity > 0 and day_high > 0 and (day_high - equity) / day_high >= settings.max_daily_loss_pct:
        blocks.append((f"Daily loss limit ({settings.max_daily_loss_pct:.0%} below the day's high)",
                       float(risk.get("rm_last_day_reset") or now) + 86400.0))
    today = int(now // 86400)
    highs = {int(d): float(v) for d, v in (risk.get("rm_equity_highs_by_day") or {}).items()
             if int(d) > today - DRAWDOWN_WINDOW_DAYS}
    if equity > 0 and highs:
        peak_day = max(highs, key=highs.get)
        if (highs[peak_day] - equity) / highs[peak_day] >= settings.max_drawdown_pct:
            blocks.append((f"Drawdown limit ({settings.max_drawdown_pct:.0%} below the best of the last "
                           f"{DRAWDOWN_WINDOW_DAYS} days)", (peak_day + DRAWDOWN_WINDOW_DAYS) * 86400.0))
    if not blocks:
        return None
    reason, until = max(blocks, key=lambda b: b[1])
    return {"reason": reason, "until": until, "all": [{"reason": r, "until": u} for r, u in blocks]}
