import time

from core.risk import btc_filter
from apps.trading_bot import main as bot_main


def _klines(start, last, n=61):
    """n 1m klines: the oldest opens at `start`, the newest closes at `last` ([t, open, high, low, close, ...])."""
    rows = [[i, "100", "100", "100", "100"] for i in range(n)]
    rows[0][1] = str(start)
    rows[-1][4] = str(last)
    return rows


def test_move_from_klines_uses_the_open_60_bars_back():
    assert abs(btc_filter.move_from_klines(_klines(100.0, 99.0)) - (-0.01)) < 1e-12
    assert abs(btc_filter.move_from_klines(_klines(50.0, 100.0, n=70)) - 0.0) < 1e-12  # 61st from the end opens at 100


def test_move_from_klines_bad_data_is_none():
    assert btc_filter.move_from_klines([]) is None
    assert btc_filter.move_from_klines(_klines(100.0, 99.0, n=60)) is None
    assert btc_filter.move_from_klines(_klines(0.0, 99.0)) is None
    assert btc_filter.move_from_klines([[0, "x", 0, 0, "y"]] * 61) is None


def test_blocks_only_against_a_hard_btc_move():
    assert btc_filter.blocks_entry(1, -0.006)          # long while BTC fell 0.6%
    assert btc_filter.blocks_entry(1, -0.005)          # at the limit
    assert not btc_filter.blocks_entry(1, -0.004)      # small dip: allowed
    assert not btc_filter.blocks_entry(1, 0.02)        # long with BTC rising: allowed
    assert btc_filter.blocks_entry(-1, 0.006)          # short while BTC rose 0.6%
    assert not btc_filter.blocks_entry(-1, -0.02)      # short with BTC falling: allowed
    assert not btc_filter.blocks_entry(1, None)        # no data never blocks
    assert not btc_filter.blocks_entry(0, -0.05)


def test_stale_btc_reading_never_blocks():
    bot = bot_main.ProductionTradingBot.__new__(bot_main.ProductionTradingBot)
    bot.btc_1h_move, bot.btc_1h_move_time = -0.02, time.time()
    assert bot._btc_move_now() == -0.02
    bot.btc_1h_move_time = time.time() - btc_filter.BTC_MOVE_MAX_AGE - 1
    assert bot._btc_move_now() is None


def test_loss_lock_is_30_minutes():
    assert bot_main.LOSS_LOCK_SECONDS == 1800.0
