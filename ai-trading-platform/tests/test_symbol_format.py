import math

from core.exchange.symbol_registry import SymbolConfig


def _cfg(qty_prec=3, step=None, price_prec=2, tick=0.01):
    c = SymbolConfig("TESTUSDT")
    c.quantity_precision, c.step_size, c.price_precision, c.tick_size = qty_prec, step, price_prec, tick
    return c


def test_exact_quantities_keep_every_step():
    # Binary floats used to lose a step (0.29 -> 0.28): closing a 0.29 position left 0.01 open on Binance
    for prec in (0, 1, 2, 3):
        c = _cfg(qty_prec=prec)
        for n in range(1, 20001):
            q = n / 10 ** prec
            assert float(c.format_quantity(q)) == q, (prec, q)


def test_quantity_is_truncated_never_rounded_up():
    assert _cfg(qty_prec=3).format_quantity(0.123999) == "0.123"
    assert _cfg(qty_prec=0, step=1.0).format_quantity(123.9) == "123"
    assert _cfg(qty_prec=1, step=0.1).format_quantity(2.37) == "2.3"
    assert _cfg(qty_prec=3).format_quantity(0.0004) == "0.000"


def test_step_size_wins_over_precision():
    assert _cfg(qty_prec=1, step=0.5).format_quantity(2.9) == "2.5"


def test_price_is_truncated_to_the_tick():
    assert _cfg(price_prec=2, tick=0.1).format_price(100.07) == "100.00"
    assert _cfg(price_prec=2, tick=0.01).format_price(0.29) == "0.29"
    assert _cfg(price_prec=7, tick=0.0000001).format_price(0.0123456789) == "0.0123456"
    for n in range(1, 5000):
        p = n / 100
        assert float(_cfg(price_prec=2, tick=0.01).format_price(p)) == p
    assert math.isclose(float(_cfg(price_prec=4, tick=0.0001).format_price(2.34567)), 2.3456)
