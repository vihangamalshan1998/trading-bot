import asyncio
from types import SimpleNamespace

import msgpack
import pytest

import apps.trading_bot.main as bot_main
from core.config.settings import settings


def _bot():
    b = bot_main.ProductionTradingBot.__new__(bot_main.ProductionTradingBot)
    b.last_hold_emission_time = {}
    b.portfolio_state = SimpleNamespace(positions={}, free_margin=1000.0, total_unrealized_pnl=0.0)
    b.market_states = {"BTCUSDT": SimpleNamespace(mid_price=100000.0)}
    b.registry = SimpleNamespace(get_current_version=lambda: "v_test")
    return b


def test_one_snapshot_per_coin_per_window(monkeypatch):
    sent = []

    async def publish(channel, payload):
        sent.append((channel, msgpack.unpackb(payload, raw=False)))

    monkeypatch.setattr(bot_main.redis_manager, "publish_binary", publish)

    async def run():
        b = _bot()
        seq = [[0.123456789, 0.0, 456411264.0]] * 3
        logits = [0.5, 0.2, 0.0, -1.0]
        b._emit_market_snapshot("BTCUSDT", seq, logits)
        b._emit_market_snapshot("BTCUSDT", seq, logits)          # same window: skipped
        b._emit_market_snapshot("ETHUSDT", seq, logits)          # other coin: recorded
        b.last_hold_emission_time["BTCUSDT"] -= settings.experience_snapshot_seconds
        b._emit_market_snapshot("BTCUSDT", seq, logits)          # next window: recorded
        await asyncio.sleep(0)

    asyncio.run(run())
    assert [p["symbol"] for _, p in sent] == ["BTCUSDT", "ETHUSDT", "BTCUSDT"]
    ch, rec = sent[0]
    assert ch == "experience:completed"
    for field in ("experience_id", "timestamp", "symbol", "action", "reward"):   # storage service requires these
        assert field in rec
    assert rec["market_state"][0] == [0.123457, 0.0, 456411000.0]               # 6 significant digits
    assert rec["confidence"] == pytest.approx(0.6)
