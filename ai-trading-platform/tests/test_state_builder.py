"""Shared state-vector builders (core/schemas/state_builder.py), used by the live bot and the history replay."""
import time
import types

import numpy as np
import torch

from core.schemas import state_builder as sb
from core.schemas.state_layout import MARKET_LOG_VOLUME_FEATURES, MARKET_SPREAD_FEATURE, MARKET_DEPRECATED_FEATURES


def test_market_block_transforms():
    f = [0.5] * 90
    f[MARKET_LOG_VOLUME_FEATURES[0]] = 1e6
    f[MARKET_SPREAD_FEATURE] = 250.0
    f[2] = 9.0
    out = sb.market_block(f)
    assert len(out) == 90
    assert abs(out[MARKET_LOG_VOLUME_FEATURES[0]] - np.log1p(1e6) / 10.0) < 1e-9
    assert out[MARKET_SPREAD_FEATURE] == 2.5
    assert out[2] == 5.0  # clamped
    assert all(out[i] == 0.0 for i in MARKET_DEPRECATED_FEATURES)
    assert sb.market_block(None) == [0.0] * 90 and sb.market_block([1.0] * 10) == [0.0] * 90


def test_macro_block_uses_given_time_and_btc_beta():
    macro = types.SimpleNamespace(sentiment_score=0.1, volatility_expectation=0.5, regime=0.0, sp500_momentum=0.01,
                                  dxy_momentum=0.0, fear_greed_index=0.4, vix_momentum=0.0, gold_momentum=0.0,
                                  treasury_yield_momentum=0.0, ndx_momentum=0.0, defi_tvl_momentum=0.0)
    sym, btc = [0.0] * 90, [0.0] * 90
    sym[2], btc[2], btc[4] = 0.002, 0.001, 0.003
    noon = time.gmtime(1792497600)  # 2026-10-20 12:00 UTC
    out = sb.macro_block(noon, sym, btc, macro)
    assert len(out) == 20
    assert abs(out[0] - 0.0) < 1e-9 and abs(out[1] + 1.0) < 1e-9  # minute 720 of 1440
    assert out[4] == 2.0 and out[9] == 0.001 and out[10] == 0.003 and out[13] == 0.4


def test_cross_section_needs_four_fresh_coins_and_ranks():
    def feats(r15):
        f = [0.0] * 90
        f[4] = r15
        return f
    fresh = {"A": feats(0.01), "B": feats(0.02), "C": feats(0.03)}
    assert sb.cross_section_block("A", fresh) == [0.0] * 10
    fresh["D"] = feats(0.04)
    out = sb.cross_section_block("D", fresh)
    assert out[4] == 1.0 and abs(out[1] - (0.04 - 0.025)) < 1e-12 and abs(out[8] - 0.025) < 1e-12


def test_whale_block_normalizes_flow():
    out = sb.whale_statarb_block({"whale_buy_pressure": 3e5, "whale_sell_pressure": 1e5}, {"okx_premium": 1e-4})
    assert len(out) == 20 and abs(out[2] - 0.5) < 1e-6 and 0 < out[3] <= 1.0 and out[10] == 1e-4
    assert sb.whale_statarb_block(None, None)[2:4] == [0.0, 0.0]


def test_finalize_matches_the_bot_torch_safety_net():
    rng = np.random.default_rng(0)
    obs = list(rng.normal(0, 4, 200))
    obs[3], obs[7], obs[11] = float("nan"), float("inf"), float("-inf")
    ref = torch.nan_to_num(torch.tensor(obs, dtype=torch.float32), nan=0.0, posinf=5.0, neginf=-5.0).clamp_(-5.0, 5.0)
    assert np.array_equal(sb.finalize(obs), ref.numpy())
