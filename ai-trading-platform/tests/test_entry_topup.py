import asyncio
from types import SimpleNamespace

import apps.trading_bot.main as bot_main
from core.exchange.symbol_registry import SymbolConfig


def _bot(mid):
    b = bot_main.ProductionTradingBot.__new__(bot_main.ProductionTradingBot)
    b.market_states = {"APTUSDT": SimpleNamespace(mid_price=mid)}
    sent = []

    async def create_order(sym, side, qty, cid, order_type="MARKET", **kw):
        sent.append((sym, side, qty, order_type, kw.get("reduce_only", False)))
        return {}

    b.binance = SimpleNamespace(create_order=create_order)
    return b, sent


def _run(b, side="OPEN_LONG", intended=12748.2, filled=159.2, limit=0.8264, pred=0.35):
    return asyncio.run(b._top_up_partial_entry("APTUSDT", side, intended, filled, limit, pred))


def _cfg(monkeypatch):
    cfg = SymbolConfig("APTUSDT")
    cfg.quantity_precision = 1
    monkeypatch.setattr(bot_main.registry, "get_symbol", lambda s: cfg)


def test_strong_signal_small_fill_tops_up_the_rest(monkeypatch):
    _cfg(monkeypatch)
    b, sent = _bot(mid=0.8265)
    assert _run(b) is True
    assert sent == [("APTUSDT", "BUY", 12589.0, "MARKET", False)]


def test_short_tops_up_with_sell(monkeypatch):
    _cfg(monkeypatch)
    b, sent = _bot(mid=0.8263)
    assert _run(b, side="OPEN_SHORT", pred=-0.40) is True
    assert sent[0][1] == "SELL"


def test_no_top_up_for_weak_signal_big_fill_or_runaway_price(monkeypatch):
    _cfg(monkeypatch)
    b, sent = _bot(mid=0.8265)
    assert _run(b, pred=0.25) is False                       # below 0.30% predicted move
    assert _run(b, filled=7000.0) is False                   # more than half already filled
    b2, sent2 = _bot(mid=0.8274)                             # long: price already 0.12% above our limit
    assert _run(b2) is False
    assert sent == [] and sent2 == []


def test_airbag_trim_is_recorded_with_pnl():
    import json
    b = bot_main.ProductionTradingBot.__new__(bot_main.ProductionTradingBot)
    pushed = []

    class R:
        async def lpush(self, key, value):
            pushed.append((key, json.loads(value)))

        async def ltrim(self, *a):
            pass

    b.redis = SimpleNamespace(redis=R())
    b.position_entry_times = {"BNBUSDT": 0.0}
    pos = SimpleNamespace(quantity=24.63, entry_price=745.38, entry_time=0.0)
    asyncio.run(b._record_airbag_trim("BNBUSDT", pos, 6.15, {"avgPrice": "746.00", "orderId": 7}, 745.9))
    key, rec = pushed[0]
    assert key == "dashboard:trade_history"
    assert rec["side"] == "PARTIAL_CLOSE_LONG" and rec["exit_reason"] == "AIRBAG MARGIN TRIM"
    assert abs(rec["gross_pnl"] - (746.00 - 745.38) * 6.15) < 1e-6
    assert rec["realized_pnl"] < rec["gross_pnl"]
