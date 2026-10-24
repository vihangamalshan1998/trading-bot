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


def test_falling_knife_uses_the_coin_limit():
    assert btc_filter.COIN_MOVE_LIMIT == 0.005
    assert btc_filter.blocks_entry(1, -0.007, btc_filter.COIN_MOVE_LIMIT)       # long into the coin's own drop
    assert btc_filter.blocks_entry(-1, 0.005, btc_filter.COIN_MOVE_LIMIT)       # short into its pump
    assert not btc_filter.blocks_entry(1, 0.03, btc_filter.COIN_MOVE_LIMIT)     # chasing is allowed
    assert not btc_filter.blocks_entry(1, -0.01, 0.02)                          # the limit is honoured


def test_stale_or_missing_reading_never_blocks():
    bot = bot_main.ProductionTradingBot.__new__(bot_main.ProductionTradingBot)
    bot.hour_moves = {"BTCUSDT": (-0.02, time.time()), "DOGEUSDT": (-0.01, time.time() - btc_filter.BTC_MOVE_MAX_AGE - 1)}
    assert bot._hour_move("BTCUSDT") == -0.02
    assert bot._hour_move("DOGEUSDT") is None   # stale
    assert bot._hour_move("ETHUSDT") is None    # never read


def test_loss_lock_is_30_minutes():
    assert bot_main.LOSS_LOCK_SECONDS == 1800.0
