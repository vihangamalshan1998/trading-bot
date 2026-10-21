import asyncio

import pytest

from core.exchange.binance_client import BinanceFuturesAdapter

EXCHANGE_INFO = {"symbols": [{"symbol": "TAOUSDT", "filters": [
    {"filterType": "LOT_SIZE", "maxQty": "300", "stepSize": "0.001"},
    {"filterType": "MARKET_LOT_SIZE", "maxQty": "10", "stepSize": "0.001"},
]}]}


def _adapter(fail_on_part=None):
    a = BinanceFuturesAdapter()
    sent = []

    async def request(method, endpoint, signed=False, params=None):
        if endpoint == "/fapi/v1/exchangeInfo":
            return EXCHANGE_INFO
        sent.append(dict(params))
        if fail_on_part is not None and len(sent) == fail_on_part:
            raise Exception("Binance API Error 400")
        qty = float(params["quantity"])
        return {"orderId": len(sent), "executedQty": params["quantity"], "cumQuote": str(qty * 270.0), "avgPrice": "270"}

    a._request = request
    return a, sent


def test_big_market_close_is_split_into_parts():
    a, sent = _adapter()
    res = asyncio.run(a.create_order("TAOUSDT", "SELL", 23.456, "ai_bot_x", order_type="MARKET", reduce_only=True))
    assert [p["quantity"] for p in sent] == ["10", "10", "3.456"]
    assert [p["newClientOrderId"] for p in sent] == ["ai_bot_x_1", "ai_bot_x_2", "ai_bot_x_3"]
    assert all(p["reduceOnly"] == "true" and p["type"] == "MARKET" for p in sent)
    assert float(res["executedQty"]) == pytest.approx(23.456)
    assert float(res["avgPrice"]) == pytest.approx(270.0)
    assert res["split_parts"] == 3


def test_small_market_and_limit_orders_are_unchanged():
    a, sent = _adapter()
    asyncio.run(a.create_order("TAOUSDT", "BUY", 4.5, "ai_bot_y", order_type="MARKET"))
    asyncio.run(a.create_order("TAOUSDT", "BUY", 50.0, "ai_bot_z", order_type="LIMIT", price="270", time_in_force="GTX"))
    assert [(p["quantity"], p["newClientOrderId"]) for p in sent] == [("4.5", "ai_bot_y"), ("50.0", "ai_bot_z")]


def test_unknown_symbol_sends_one_order():
    a, sent = _adapter()
    asyncio.run(a.create_order("BTCUSDT", "SELL", 500.0, "ai_bot_b", order_type="MARKET"))
    assert len(sent) == 1


def test_failed_part_raises_so_the_bot_retries():
    a, sent = _adapter(fail_on_part=2)
    with pytest.raises(Exception):
        asyncio.run(a.create_order("TAOUSDT", "SELL", 25.0, "ai_bot_f", order_type="MARKET", reduce_only=True))
    assert len(sent) == 2
