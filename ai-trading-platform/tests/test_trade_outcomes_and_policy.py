import collections
import types

import torch

from core.risk import exit_policy as xp
from apps.market_collector.context_poller import cross_exchange_features, depth_features, okx_inst_id

LV = xp.ExitLevels(stop=0.005, tp1=0.0075, hard_tp=0.015, breakeven=0.001, trail_gap=0.005)
E = 100.0


def bars(*moves):
    """(high, low, close) bars from (high%, low%, close%) moves relative to entry 100."""
    return [(E * (1 + h / 100), E * (1 + l / 100), E * (1 + c / 100)) for h, l, c in moves]


def test_stop_before_recovery_is_a_loss():
    # dips through the stop, THEN rallies: fixed-horizon labels called this a winner
    path = bars((0.1, -0.6, -0.4), (1.0, -0.2, 0.9)) + bars((1.0, 0.8, 0.9)) * 300
    assert xp.simulate_trade(1, E, path, LV) == -0.005
    assert xp.simulate_trade(-1, E, path, LV) < 0  # the short also loses: both sides can be wrong


def test_tp1_then_trailing_runner():
    path = bars((0.8, 0.0, 0.7), (1.2, 0.7, 1.1), (1.1, 0.6, 0.6))
    # half at TP1 (0.75%), runner stopped at peak 1.2% - 0.5% trail = 0.7%
    assert abs(xp.simulate_trade(1, E, path, LV) - (0.5 * 0.0075 + 0.5 * 0.007)) < 1e-9


def test_same_bar_stop_and_target_counts_as_stop():
    assert xp.simulate_trade(1, E, bars((2.0, -1.0, 0.5)), LV) == -0.005


def test_max_hold_closes_at_close():
    path = bars((0.2, -0.2, 0.1)) * 300
    assert abs(xp.simulate_trade(1, E, path, LV) - 0.001) < 1e-9
    assert xp.evaluate_exit(0.004, 0.004, False, xp.MAX_HOLD_SECONDS + 1, LV).reason == "MAX HOLD"


def test_volatility_helper_matches_engine():
    from apps.feature_engine.engine import FeatureEngine
    eng = FeatureEngine("X")
    klines = [[i * 60000, 100 + (i % 7) * 0.1, 100.5 + (i % 5) * 0.1, 99.5 - (i % 3) * 0.1, 100 + (i % 11) * 0.05, 1.0]
              for i in range(600)]
    eng.seed_bars(klines)
    ohlc = [[float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in klines]
    assert eng.volatility_1h() > 0
    assert abs(eng.volatility_1h() - xp.volatility_1h_from_ohlc(ohlc)) < 1e-12


def test_trainer_scores_actions_with_real_outcomes():
    from apps.trainer.ppo import PPOTrainer
    t = PPOTrainer.__new__(PPOTrainer)
    rewards = torch.tensor([[0.0, 0.5, 0.5, -0.5, -0.3]])  # market up, but the long got stopped out
    lr, sr = t.trade_outcomes(rewards)
    pnl = t.action_pnl(torch.tensor([[0.9, -0.9, 0.0]]), lr, sr)
    cost = xp.ROUND_TRIP_COST * 100
    assert torch.allclose(pnl, torch.tensor([[-0.5 - cost, -0.3 - cost, 0.0]]))


def test_trade_policy_blocks_losing_coin_and_calibrates():
    from apps.trainer.ppo import PPOTrainer
    t = PPOTrainer.__new__(PPOTrainer)
    n = 200
    torch.manual_seed(0)
    conf_raw = torch.linspace(0.3, 0.95, n)
    acts = torch.zeros(n, 4)
    acts[:, 0] = 0.9
    acts[:, 1] = conf_raw * 2 - 1
    pos = torch.ones(n, 1)
    symbols = ["GOOD" if i % 2 else "BAD" for i in range(n)]
    pnl = torch.tensor([[0.3 if (s == "GOOD" and c > 0.5) else -0.2] for s, c in zip(symbols, conf_raw.tolist())])
    policy = t.derive_trade_policy(acts, pnl, pos, symbols)
    assert policy["blocked_symbols"] == ["BAD"]
    assert policy["min_confidence"] > 0.45
    wr = policy["calibration"]["win_rate"]
    assert all(b >= a for a, b in zip(wr, wr[1:]))  # monotone
    assert policy["filtered_pnl_per_trade"] > 0


def test_depth_and_cross_exchange_features():
    bids = [[99.95, 10], [99.5, 10], [98.0, 100]]
    asks = [[100.05, 5], [100.5, 5], [102.0, 100]]
    d = depth_features(bids, asks)
    assert d["depth_imb_0p1pct"] > 0 and d["depth_imb_1pct"] > 0
    assert d["depth_bid_usd_1pct"] > d["depth_ask_usd_1pct"]
    hist = collections.deque()
    cross_exchange_features(0.0001, [0.0003, None], hist, {"binance": 100.0, "bybit": 100.0}, now=0.0)
    out = cross_exchange_features(0.0001, [0.0003, None], hist, {"binance": 110.0, "bybit": 110.0, "okx": 50.0}, now=3600.0)
    assert abs(out["xex_funding_spread"] - 0.0002) < 1e-12
    assert abs(out["xex_oi_change_1h"] - 0.10) < 1e-9  # OKX appearing does not count as OI growth
    assert okx_inst_id("1000PEPEUSDT") == "PEPE-USDT-SWAP"


def test_cross_section_rank():
    from apps.trading_bot.main import ProductionTradingBot
    import time as _t
    mk = lambda r: types.SimpleNamespace(features=[0.0] * 3 + [r, r, r, r] + [0.0] * 83, timestamp=_t.time())
    bot = types.SimpleNamespace(market_states={"A": mk(0.01), "B": mk(0.0), "C": mk(-0.01), "D": mk(0.02)})
    f = ProductionTradingBot._cross_section_features(bot, "D")
    assert len(f) == 10
    assert abs(f[1] - (0.02 - 0.005)) < 1e-12   # relative 15m return
    assert f[4] == 1.0                           # strongest coin -> rank +1
