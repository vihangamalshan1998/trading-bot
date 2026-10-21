"""
Exit policy economics, the 200-slot layout, the feature engine's bar indicators, and
backward-compatible per-slot standardization in the model.
"""
import time
import numpy as np
import torch

from core.risk import exit_policy as xp
from core.schemas import state_layout as L
from apps.feature_engine.engine import FeatureEngine
from apps.research.model import SingleSymbolActorCritic


# ---------------------------------------------------------------- exit policy
def test_stop_never_wider_than_previous_and_reward_beats_risk():
    for vol in (0.0, 0.001, 0.004, 0.0075, 0.02, 0.08):
        lv = xp.exit_levels(vol)
        assert xp.MIN_STOP <= lv.stop <= 0.0075, "loss per trade must not grow while leverage is unchanged"
        assert lv.tp1 >= 1.5 * lv.stop
        assert lv.tp1 - xp.ROUND_TRIP_COST > lv.stop * 0 + xp.ROUND_TRIP_COST  # TP1 net of costs is a real gain
        assert lv.breakeven >= xp.ROUND_TRIP_COST, "breakeven must cover round-trip costs"


def test_exit_sequence_stop_tp1_trail_breakeven():
    lv = xp.exit_levels(0.005)  # stop 0.5%
    assert xp.evaluate_exit(-0.0051, 0.0, False, 60, lv).reason == "HARD STOP"
    assert not xp.evaluate_exit(0.004, 0.004, False, 60, lv).close_half, "no banking of tiny wins"
    assert xp.evaluate_exit(lv.tp1, lv.tp1, False, 60, lv).close_half
    # after TP1: runner trails one stop behind the peak
    assert xp.evaluate_exit(0.009, 0.0145, True, 60, lv).reason == "TRAILING STOP"
    assert xp.evaluate_exit(0.014, 0.0145, True, 60, lv).reason == ""
    # ... and never below a fee-covering breakeven
    assert xp.evaluate_exit(0.0009, 0.0055, True, 60, lv).reason == "FREE-ROLL BREAKEVEN"
    assert xp.evaluate_exit(lv.hard_tp, lv.hard_tp, True, 60, lv).reason == "HARD TAKE-PROFIT"
    assert xp.evaluate_exit(0.0002, 0.003, False, xp.STALE_SECONDS + 1, lv).reason == "STALE PRUNE"


def test_expected_move_blends_horizons():
    assert abs(xp.expected_move_pct([1.0, 1.0, 1.0]) - 1.0) < 1e-9
    assert abs(xp.expected_move_pct(np.array([0.0, 1.0, 0.0])) - xp.HORIZON_WEIGHTS[1]) < 1e-9
    assert xp.expected_move_pct(None) == 0.0


# ---------------------------------------------------------------- slot layout
def test_layout_matches_previous_hardcoded_indices():
    assert L.LOG_VOLUME_SLOTS == (30, 31, 32, 39, 41, 42, 101, 102)
    assert L.SPREAD_SLOT == 40
    assert L.MARKET_LOG_VOLUME_FEATURES == (10, 11, 12, 19, 21, 22, 81, 82)
    assert L.MARKET_SPREAD_FEATURE == 20
    assert all(L.SLOTS[i].status == L.RESERVED for i in range(150, 200))


# ---------------------------------------------------------------- feature engine
def _feed(engine, n_seconds, start, price_fn, vol=1.0):
    for i in range(n_seconds):
        engine.add_tick({"mid_price": price_fn(i), "best_bid": price_fn(i) * 0.9999, "best_ask": price_fn(i) * 1.0001,
                         "volume": vol, "buy_volume": vol * 0.6, "bid_qty": 5.0, "ask_qty": 3.0}, ts=start + i)


def test_bar_indicators_and_volume_are_live():
    e = FeatureEngine("TEST")
    t0 = (int(time.time()) // 60 - 1100) * 60
    klines = [[(t0 + 60 * i) * 1000, 100 + i * 0.01, 100 + i * 0.01 + 0.05, 100 + i * 0.01 - 0.05, 100 + i * 0.01 + 0.005, 10.0]
              for i in range(1000)]
    e.seed_bars(klines)
    assert len(e.bars) == 1000
    assert e.volatility_1h() > 0
    _feed(e, 120, time.time() - 120, lambda i: 110.0 + 0.001 * i)
    f = e.compute_features()
    assert f.shape == (90,)
    # previously always 0: volume, trade imbalance, rsi 5m/15m/1h, ema50/200, sma50/200, stoch_d, adx
    for idx in (10, 11, 13, 36, 37, 38, 44, 45, 46, 47, 53, 54):
        assert f[idx] != 0.0, f"feature {idx} should be live"
    assert abs(f[13] - 0.2) < 1e-3  # 60% buy volume -> imbalance 0.2
    assert f[14] == 0.0 and f[15] == 0.0, "deprecated raw-VWAP slots stay 0"
    assert 0.0 <= f[52] <= 1.0, "stoch_k bounded"


# ---------------------------------------------------------------- model scaling
def test_model_feature_scaling_is_backward_compatible():
    torch.manual_seed(0)
    m = SingleSymbolActorCritic()
    x = torch.randn(2, 5, 200)
    legacy_sd = {k: v for k, v in m.state_dict().items() if k not in m.FEATURE_BUFFERS}
    m2 = SingleSymbolActorCritic()
    m2.load_state_dict(legacy_sd)
    m.eval(); m2.eval()
    with torch.no_grad():
        assert torch.allclose(m(x)[0], m2(x)[0])

    mask = torch.ones(200); mask[150:] = 0.0
    m2.set_feature_stats(torch.zeros(200), torch.ones(200), mask)
    x_noise = x.clone(); x_noise[..., 150:] = torch.randn(2, 5, 50) * 3
    with torch.no_grad():
        assert torch.allclose(m2(x)[0], m2(x_noise)[0]), "masked RESERVED slots must not influence the model"
