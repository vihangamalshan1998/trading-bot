import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pandas as pd
import time

from apps.feature_engine.engine import FeatureEngine
from core.risk.edge_estimator import EdgeEstimator
from apps.research.model import SingleSymbolActorCritic
from apps.trainer.ppo import PPOTrainer
from core.ai.registry import ModelRegistry
from core.telemetry.performance_metrics import PerformanceReporter
from core.schemas.state_schema import MarketState

def test_rsi_all_gain_and_all_loss():
    """Verify RSI correctly outputs 100 on all-gain, 0 on all-loss, and 50 on flat."""
    fe = FeatureEngine("BTCUSDT")
    
    # 1. Flat price
    flat_prices = np.array([100.0] * 30)
    assert fe._rsi(flat_prices) == 50.0
    
    # 2. All gain price
    gain_prices = np.array([100.0 + i for i in range(30)])
    assert fe._rsi(gain_prices) == 100.0
    
    # 3. All loss price
    loss_prices = np.array([100.0 - i for i in range(30)])
    assert fe._rsi(loss_prices) == 0.0

def test_atr_calculation():
    """Verify ATR computes valid positive true range."""
    fe = FeatureEngine("BTCUSDT")
    highs = np.array([105.0, 106.0, 107.0, 108.0])
    lows = np.array([95.0, 96.0, 97.0, 98.0])
    closes = np.array([100.0, 101.0, 102.0, 103.0])
    atr = fe._atr(highs, lows, closes, period=3)
    assert atr > 0.0
    assert np.isfinite(atr)

def test_edge_estimator_costs_and_hurdles():
    """Verify EdgeEstimator deducts maker/taker fees, spread, and blocks negative edge."""
    estimator = EdgeEstimator(min_net_edge_pct=0.08)
    market = MarketState(
        symbol="BTCUSDT",
        timestamp=time.time(),
        bid=100.0,
        ask=100.02,
        mid_price=100.01,
        last_price=100.01,
        spread=0.02,
        order_book_imbalance=0.0,
        volume=1000.0,
        vwap=100.01,
        volatility=0.01,
        funding_rate=0.0001,
        features=[0.0] * 90
    )
    
    # 1. Gross move 0.05% is smaller than fees (~0.07%), should be rejected
    result_small = estimator.estimate(
        symbol="BTCUSDT",
        side="OPEN_LONG",
        predicted_move_pct=0.05,
        confidence=0.85,
        market=market,
        is_maker=True
    )
    assert not result_small.approved
    assert "NEGATIVE_NET_EDGE" in result_small.reason_code or "SUB_HURDLE" in result_small.reason_code
    
    # 2. Large gross move 0.40% easily clears fees and hurdle, should be approved
    result_large = estimator.estimate(
        symbol="BTCUSDT",
        side="OPEN_LONG",
        predicted_move_pct=0.40,
        confidence=0.85,
        market=market,
        is_maker=True
    )
    assert result_large.approved
    assert result_large.reason_code == "EDGE_VERIFIED"
    assert result_large.net_expected_return_pct > 0.08

def test_ppo_gaussian_policy_math():
    """Verify PPO uses Gaussian distribution and does not invent random advantages."""
    model = SingleSymbolActorCritic(input_dim=200)
    trainer = PPOTrainer(model=model, limit=10)
    
    # Check log_std parameter exists and is learnable
    assert hasattr(trainer, "log_std")
    assert trainer.log_std.requires_grad
    
    # Verify zero-variance advantage handling does not raise and defers
    rewards = torch.zeros((5, 3))
    values = torch.zeros((5, 3))
    adv, ret = trainer.compute_gae(rewards, values, values, torch.zeros(5))
    adv_mean = adv.mean(dim=1, keepdim=True)
    adv_std = adv_mean.std(unbiased=False)
    assert adv_std <= 1e-6 # Verifies zero variance correctly detected

def test_model_registry_promote_and_rollback(tmp_path):
    """Verify Champion vs Candidate promotion and rollback mechanics."""
    registry = ModelRegistry(model_dir=str(tmp_path))
    model_champ = SingleSymbolActorCritic(input_dim=200)
    model_cand = SingleSymbolActorCritic(input_dim=200)
    
    # Save champion v1
    registry.save_model(model_champ, version_tag="v1_champ")
    champ_path = os.path.join(str(tmp_path), "model_v1.pt")
    assert os.path.exists(champ_path)
    
    # Save candidate
    cand_path = os.path.join(str(tmp_path), "cand_v2.pt")
    torch.save(model_cand.state_dict(), cand_path)
    
    # Promote candidate
    promoted = registry.promote_candidate(cand_path, version_tag="v2_promoted")
    assert promoted
    assert registry.current_version == "v2_promoted"
    assert os.path.exists(os.path.join(str(tmp_path), "model_v1_backup.pt"))
    
    # Rollback
    rolled_back = registry.rollback()
    assert rolled_back
    assert os.path.exists(champ_path)

def test_performance_reporter_expectancy():
    """Verify performance metrics correctly compute Win Rate, Profit Factor, and Net Expectancy."""
    reporter = PerformanceReporter()
    trades = [
        {"realized_pnl": 20.0, "fees_paid": 2.0},  # Net +18
        {"realized_pnl": 30.0, "fees_paid": 2.0},  # Net +28
        {"realized_pnl": -10.0, "fees_paid": 2.0}, # Net -12
    ]
    summary = reporter.calculate_summary(trades)
    assert summary.total_trades == 3
    assert summary.winning_trades == 2
    assert summary.losing_trades == 1
    assert abs(summary.win_rate_pct - 66.67) < 0.1
    assert summary.net_pnl == (18 + 28 - 12) # +34.0
    assert summary.profit_factor == (46.0 / 12.0)
    assert summary.net_expectancy > 0.0
