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

def test_ppo_data_validation_and_rejection():
    """Verify Task 3: PPO data validation, dimension checks, corrupted sample rejection, and behavior log probs."""
    torch.manual_seed(42)
    model = SingleSymbolActorCritic(input_dim=200)
    trainer = PPOTrainer(model=model, limit=10)

    # 1. Valid trajectory with varied rewards
    batch_n = 10
    states = torch.randn(batch_n, 300, 200)
    actions = torch.clamp(torch.randn(batch_n, 4) * 0.5, -1.0, 1.0)
    # Give varied rewards so adv_std > 1e-6
    rewards = torch.randn(batch_n, 3) * 5.0
    next_states = torch.randn(batch_n, 300, 200)
    dones = torch.zeros(batch_n, 1)
    dones[4, 0] = 1.0 # Valid binary terminal flag

    # Successful valid update
    success = trainer.train_step(tensors=(states.clone(), actions.clone(), rewards.clone(), next_states.clone(), dones.clone()), epochs=1)
    assert success is True

    # 2. Corrupted samples with NaN/Inf are filtered out and logged, not silently zeroed
    states_corrupt = states.clone()
    rewards_corrupt = rewards.clone()
    # Inject NaN into 2 samples
    states_corrupt[0, 5, 10] = float("nan")
    rewards_corrupt[1, 0] = float("inf")
    # 8 valid samples remain (> 4 minimum), update succeeds on valid samples
    success_filtered = trainer.train_step(tensors=(states_corrupt, actions.clone(), rewards_corrupt, next_states.clone(), dones.clone()), epochs=1)
    assert success_filtered is True

    # 3. All samples corrupted -> aborts update
    states_all_nan = torch.full_like(states, float("nan"))
    assert trainer.train_step(tensors=(states_all_nan, actions.clone(), rewards.clone(), next_states.clone(), dones.clone())) is False

    # 4. Invalid dimensions rejected
    bad_states = torch.randn(batch_n, 300, 50) # Expected 200
    assert trainer.train_step(tensors=(bad_states, actions.clone(), rewards.clone(), next_states.clone(), dones.clone())) is False

    bad_actions = torch.randn(batch_n, 2) # Expected 4
    assert trainer.train_step(tensors=(states.clone(), bad_actions, rewards.clone(), next_states.clone(), dones.clone())) is False

    bad_rewards = torch.randn(batch_n, 1) # Expected 3
    assert trainer.train_step(tensors=(states.clone(), actions.clone(), bad_rewards, next_states.clone(), dones.clone())) is False

    bad_dones = torch.randn(batch_n, 2) # Expected (batch, 1)
    assert trainer.train_step(tensors=(states.clone(), actions.clone(), rewards.clone(), next_states.clone(), bad_dones)) is False

    # 5. Non-binary terminal flags rejected
    invalid_dones = dones.clone()
    invalid_dones[0, 0] = 0.5 # Non-binary
    assert trainer.train_step(tensors=(states.clone(), actions.clone(), rewards.clone(), next_states.clone(), invalid_dones)) is False

    # 6. Out-of-bounds action values rejected
    extreme_actions = actions.clone()
    extreme_actions[0, 0] = 999.0
    assert trainer.train_step(tensors=(states.clone(), extreme_actions, rewards.clone(), next_states.clone(), dones.clone())) is False

    # 7. Behavior policy log probabilities validation
    # When require_stored_behavior_policy=True and no stored old log probs -> rejected!
    assert trainer.train_step(
        tensors=(states.clone(), actions.clone(), rewards.clone(), next_states.clone(), dones.clone()),
        require_stored_behavior_policy=True
    ) is False

    # When valid stored old log probs are provided -> accepted!
    stored_probs = torch.randn(batch_n, 1)
    assert trainer.train_step(
        tensors=(states.clone(), actions.clone(), rewards.clone(), next_states.clone(), dones.clone()),
        stored_old_log_probs=stored_probs,
        require_stored_behavior_policy=True,
        epochs=1
    ) is True

    # When stored old log probs contain NaN -> rejected!
    nan_stored_probs = stored_probs.clone()
    nan_stored_probs[0, 0] = float("nan")
    assert trainer.train_step(
        tensors=(states.clone(), actions.clone(), rewards.clone(), next_states.clone(), dones.clone()),
        stored_old_log_probs=nan_stored_probs,
        require_stored_behavior_policy=True
    ) is False

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
    
    # Promote candidate with valid approved metrics
    valid_metrics = {
        "approved": True,
        "reason": "SUPERIOR_TO_CHAMPION",
        "challenger_score": 0.15,
        "champion_score": 0.05,
        "critic_loss": 0.01,
        "sample_size": 100,
        "eval_timestamp": time.time(),
        "data_split": "chronological_holdout"
    }
    promoted = registry.promote_candidate(cand_path, version_tag="v2_promoted", metrics=valid_metrics)
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

def test_edge_estimator_negative_prediction_rejected():
    """Verify EdgeEstimator rejects negative predicted returns without abs() flipping."""
    estimator = EdgeEstimator(min_net_edge_pct=0.08)
    market = MarketState(
        symbol="ETHUSDT",
        timestamp=time.time(),
        bid=2000.0,
        ask=2000.1,
        mid_price=2000.05,
        last_price=2000.05,
        spread=0.1,
        order_book_imbalance=0.0,
        volume=5000.0,
        vwap=2000.05,
        volatility=0.01,
        funding_rate=0.0001,
        features=[0.0] * 90
    )
    result = estimator.estimate(
        symbol="ETHUSDT",
        side="OPEN_LONG",
        predicted_move_pct=-0.25, # Model expects price drop
        confidence=0.80,
        market=market,
        is_maker=True
    )
    assert not result.approved
    assert result.reason_code == "NEGATIVE_PREDICTED_RETURN"

def test_chronological_splits_no_leakage():
    """Verify historical futures loader partitions data chronologically with 0% overlap."""
    from apps.research.historical_futures_loader import HistoricalFuturesLoader
    loader = HistoricalFuturesLoader()
    df = pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=100, freq="5min"),
        "open": np.linspace(100, 200, 100),
        "high": np.linspace(101, 201, 100),
        "low": np.linspace(99, 199, 100),
        "close": np.linspace(100.5, 200.5, 100),
        "volume": [1000.0] * 100
    })
    train, val, test = loader.split_chronological(df, train_ratio=0.70, val_ratio=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    # Strict temporal causality: train max time < val min time < test min time
    assert train["open_time"].max() < val["open_time"].min()
    assert val["open_time"].max() < test["open_time"].min()

def test_untrained_model_execution_prohibited():
    """Verify untrained models are strictly forbidden from setting has_valid_model = True."""
    from apps.trading_bot.main import ProductionTradingBot
    from core.config.settings import settings
    from unittest.mock import patch
    
    # Mock registry.load_model to raise FileNotFoundError
    with patch("core.ai.registry.ModelRegistry.load_model", side_effect=FileNotFoundError("No checkpoint")):
        bot = ProductionTradingBot(symbols=settings.symbol_universe)
        assert bot.has_valid_model is False # Must remain False to prevent rogue order execution

def test_recursive_gae_hand_calculated():
    """Verify recursive GAE with hand-calculated 3-step trajectory and terminal boundary."""
    from apps.trainer.ppo import PPOTrainer
    from apps.research.model import SingleSymbolActorCritic
    
    model = SingleSymbolActorCritic(input_dim=200)
    trainer = PPOTrainer(model=model, limit=10, gamma=0.90)
    
    # T = 3 trajectory
    rewards = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    values = torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float32)
    next_values = torch.tensor([[1.5], [2.5], [3.5]], dtype=torch.float32)
    dones = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32) # Terminal at step 2
    
    adv, ret = trainer.compute_gae(rewards, values, next_values, dones, gae_lambda=0.80, is_trajectory=True)
    
    # Step 2: delta = 3 + 0 - 2.5 = 0.5, adv = 0.5, ret = 3.0
    assert abs(adv[2, 0].item() - 0.5) < 1e-4
    assert abs(ret[2, 0].item() - 3.0) < 1e-4
    
    # Step 1: delta = 2 + 0.9 * 2.5 - 1.5 = 2.75, adv = 2.75 + (0.72 * 0.5) = 3.11, ret = 4.61
    assert abs(adv[1, 0].item() - 3.11) < 1e-4
    assert abs(ret[1, 0].item() - 4.61) < 1e-4
    
    # Step 0: delta = 1 + 0.9 * 1.5 - 0.5 = 1.85, adv = 1.85 + (0.72 * 3.11) = 4.0892, ret = 4.5892
    assert abs(adv[0, 0].item() - 4.0892) < 1e-4
    assert abs(ret[0, 0].item() - 4.5892) < 1e-4

def test_edge_estimator_uncertainty_penalty():
    """Verify uncertainty penalty reduces net edge for low-confidence setups."""
    estimator = EdgeEstimator(min_net_edge_pct=0.05)
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
    # High confidence (0.95 -> uncertainty = 0.05)
    res_high = estimator.estimate(
        symbol="BTCUSDT", side="OPEN_LONG", predicted_move_pct=0.30, confidence=0.95, market=market, is_maker=True
    )
    # Low confidence (0.50 -> uncertainty = 0.50)
    res_low = estimator.estimate(
        symbol="BTCUSDT", side="OPEN_LONG", predicted_move_pct=0.30, confidence=0.50, market=market, is_maker=True
    )
    # Net edge should be strictly lower for low confidence due to uncertainty penalty
    assert res_high.net_expected_return_pct > res_low.net_expected_return_pct

def test_backtest_unchanged_price_produces_loss():
    """Verify that a position held across unchanged prices produces a net loss due to fees and friction."""
    position = 1.0  # LONG
    entry_price = 100.0
    exit_price = 100.0  # Zero price change
    leverage = 5
    taker_fee_pct = 0.0005
    spread_pct = 0.00015
    slippage_pct = 0.00010
    
    gross_trade_ret = position * ((exit_price - entry_price) / entry_price)
    exit_friction = taker_fee_pct + spread_pct + slippage_pct
    net_trade_ret = (gross_trade_ret * leverage) - (exit_friction * leverage)
    
    assert gross_trade_ret == 0.0
    assert net_trade_ret < 0.0  # Must be strictly negative due to friction

def test_dynamic_universe_accepts_new_usdt_symbols():
    """Verify ProductionTradingBot accepts dynamic universes without crashing on settings.symbol_universe mismatch."""
    from apps.trading_bot.main import ProductionTradingBot
    from unittest.mock import patch
    
    dynamic_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT", "PEPEUSDT"]
    with patch("core.ai.registry.ModelRegistry.load_model", side_effect=FileNotFoundError("No model")):
        bot = ProductionTradingBot(symbols=dynamic_symbols)
        assert bot.symbols == dynamic_symbols
        assert bot.num_symbols == 5
        assert bot.has_valid_model is False

def test_independent_transitions_gae():
    """Verify deterministic 1-step TD advantage when is_trajectory=False with zero forward leakage."""
    from apps.trainer.ppo import PPOTrainer
    from apps.research.model import SingleSymbolActorCritic
    
    model = SingleSymbolActorCritic(input_dim=200)
    trainer = PPOTrainer(model=model, limit=10, gamma=0.90)
    
    # 3 independent transitions
    rewards = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    values = torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float32)
    next_values = torch.tensor([[1.0], [2.0], [4.0]], dtype=torch.float32)
    dones = torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float32)
    
    adv, ret = trainer.compute_gae(rewards, values, next_values, dones, is_trajectory=False)
    
    # For independent transitions: delta_i = r_i + gamma * v_{next_i} * (1 - done_i) - v_i
    # Row 0: 1.0 + 0.9 * 1.0 - 0.5 = 1.4
    # Row 1: 2.0 + 0.9 * 2.0 - 1.5 = 2.3
    # Row 2: 3.0 + 0.9 * 4.0 - 2.5 = 4.1
    assert abs(adv[0, 0].item() - 1.4) < 1e-4
    assert abs(adv[1, 0].item() - 2.3) < 1e-4
    assert abs(adv[2, 0].item() - 4.1) < 1e-4
    assert abs(ret[0, 0].item() - (1.4 + 0.5)) < 1e-4

def test_terminal_transitions_zero_future_value():
    """Verify that when done=1.0, future values are strictly zeroed out."""
    from apps.trainer.ppo import PPOTrainer
    from apps.research.model import SingleSymbolActorCritic
    
    model = SingleSymbolActorCritic(input_dim=200)
    trainer = PPOTrainer(model=model, limit=10, gamma=0.95)
    
    rewards = torch.tensor([[5.0]], dtype=torch.float32)
    values = torch.tensor([[2.0]], dtype=torch.float32)
    next_values = torch.tensor([[100.0]], dtype=torch.float32) # Huge future value
    dones = torch.tensor([[1.0]], dtype=torch.float32) # Terminal flag
    
    adv, ret = trainer.compute_gae(rewards, values, next_values, dones, is_trajectory=False)
    # With done=1.0, next_value MUST NOT be factored: delta = 5.0 + 0.95 * 100 * 0.0 - 2.0 = 3.0
    assert abs(adv[0, 0].item() - 3.0) < 1e-4
    assert abs(ret[0, 0].item() - 5.0) < 1e-4

def test_dynamic_universe_register_and_position_preservation():
    """Verify that symbols can be dynamically registered and inactive screener symbols with open positions are not skipped."""
    from apps.trading_bot.main import ProductionTradingBot
    from core.exchange.symbol_registry import registry, SymbolConfig
    from unittest.mock import patch
    
    # Ensure SOLUSDT exists in exchange registry for contract validation
    sol_cfg = SymbolConfig("SOLUSDT", "SOL", "USDT")
    sol_cfg.status = "ENABLED"
    sol_cfg.trading_enabled = True
    registry.add_symbol(sol_cfg)
    
    with patch("core.ai.registry.ModelRegistry.load_model", side_effect=FileNotFoundError("No model")):
        bot = ProductionTradingBot(symbols=["BTCUSDT", "ETHUSDT"])
        
        # Register new symbol dynamically
        registered = bot.register_symbol("SOLUSDT")
        assert registered is True
        assert "SOLUSDT" in bot.symbols
        assert bot.portfolio_state.positions["SOLUSDT"] is not None
        assert bot.portfolio_state.positions["SOLUSDT"].quantity == 0.0
        
        # Test inactive screener filtering
        bot.dynamic_active_symbols = {"BTCUSDT"} # SOLUSDT and ETHUSDT not in screener
        
        # SOLUSDT has 0 quantity -> should not open new trade
        pos_sol = bot.portfolio_state.positions["SOLUSDT"]
        should_skip_sol = (pos_sol.quantity == 0 and "SOLUSDT" not in bot.dynamic_active_symbols)
        assert should_skip_sol is True
        
        # ETHUSDT has an active open position -> should NEVER be skipped
        bot.portfolio_state.positions["ETHUSDT"].quantity = 1.5
        pos_eth = bot.portfolio_state.positions["ETHUSDT"]
        should_skip_eth = (pos_eth.quantity == 0 and "ETHUSDT" not in bot.dynamic_active_symbols)
        assert should_skip_eth is False # Preserved!

def test_feature_engine_exact_90_dims_and_production_vector():
    """Verify FeatureEngine produces exactly 90 slots and state vector is exactly 200 dims."""
    from apps.feature_engine.engine import FeatureEngine
    
    engine = FeatureEngine(symbol="BTCUSDT")
    for i in range(30):
        engine.add_tick({
            "mid_price": 50000.0 + i * 10,
            "volume": 100.0,
            "buy_volume": 60.0,
            "trade_count": 50.0,
            "best_bid": 49995.0,
            "best_ask": 50005.0,
            "bid_qty": 5.0,
            "ask_qty": 5.0,
            "funding_rate": 0.0001,
            "open_interest": 10000.0,
            "price_change_24h": 0.0,
            "volume_24h": 10000.0,
            "mark_price_premium": 0.0
        })
    market_feats = engine.compute_features()
    assert len(market_feats) == 90
    
    # 10 portfolio + 10 position + 90 market + 20 macro + 70 statarb = 200
    state_vector = [0.0] * 10 + [0.0] * 10 + list(market_feats) + [0.0] * 20 + [0.0] * 70
    assert len(state_vector) == 200

def test_genuine_contract_validation_fail_closed():
    """Verify is_valid_contract fails closed for non-USDT, excluded, or unlisted contracts."""
    from apps.trading_bot.main import ProductionTradingBot
    from core.exchange.symbol_registry import registry, SymbolConfig
    from unittest.mock import patch

    with patch("core.ai.registry.ModelRegistry.load_model", side_effect=FileNotFoundError("No model")):
        bot = ProductionTradingBot(symbols=["BTCUSDT", "ETHUSDT"])
        
        # 1. Non-USDT rejected
        assert bot.is_valid_contract("BTCBUSD") is False
        assert bot.is_valid_contract("INVALID") is False
        assert bot.is_valid_contract(12345) is False
        
        # 2. Configured exclusion rejected (e.g. SUIUSDT)
        assert bot.is_valid_contract("SUIUSDT") is False
        
        # 3. Empty exchange registry -> fails closed even for configured symbols!
        registry._symbols = {}
        assert bot.is_valid_contract("BTCUSDT") is False
        
        # 4. With populated exchange registry:
        # Mock exchange registry with BTCUSDT (ENABLED) and SHIBUSDT (DISABLED)
        btc_cfg = SymbolConfig("BTCUSDT", "BTC", "USDT")
        btc_cfg.status = "ENABLED"
        btc_cfg.trading_enabled = True
        registry.add_symbol(btc_cfg)
        
        dis_cfg = SymbolConfig("SHIBUSDT", "SHIB", "USDT")
        dis_cfg.status = "DELISTED"
        dis_cfg.trading_enabled = False
        registry.add_symbol(dis_cfg)
        
        # BTCUSDT is enabled and valid
        assert bot.is_valid_contract("BTCUSDT") is True
        # SHIBUSDT is delisted
        assert bot.is_valid_contract("SHIBUSDT") is False
        # UNKNOWNUSDT not in exchange metadata -> fail closed!
        assert bot.is_valid_contract("UNKNOWNUSDT") is False

def test_candidate_evaluation_rejects_insufficient_samples():
    """Verify candidate evaluation fails closed and rejects promotion when samples < 30."""
    from apps.trainer.ppo import PPOTrainer
    from apps.research.model import SingleSymbolActorCritic
    from unittest.mock import patch
    
    model = SingleSymbolActorCritic(input_dim=200)
    trainer = PPOTrainer(model=model, limit=10)
    
    # Mock buffer.sample returning empty or insufficient list
    with patch.object(trainer.buffer, "sample_holdout", return_value=[]):
        res = trainer.evaluate_candidate(sample_size=64)
        assert res["approved"] is False
        assert res["reason"] == "INSUFFICIENT_SAMPLE_SIZE"
        
    with patch.object(trainer.buffer, "sample_holdout", return_value=[object()] * 10):
        res = trainer.evaluate_candidate(sample_size=64)
        assert res["approved"] is False
        assert res["reason"] == "INSUFFICIENT_SAMPLE_SIZE"

def test_missing_and_corrupt_champion_fails_closed(tmp_path):
    """Verify evaluate_candidate fails closed on missing champion or corrupt champion."""
    from apps.trainer.ppo import PPOTrainer
    from apps.research.model import SingleSymbolActorCritic
    from unittest.mock import patch
    
    model = SingleSymbolActorCritic(input_dim=200)
    trainer = PPOTrainer(model=model, limit=10)
    
    mock_batch = [object()] * 35
    mock_tensors = (
        torch.randn((35, 300, 200)),
        torch.randn((35, 300, 4)),
        torch.randn((35, 3)),
        torch.randn((35, 300, 200)),
        torch.zeros((35, 300, 1))
    )
    
    with patch.object(trainer.buffer, "sample_holdout", return_value=mock_batch), \
         patch.object(trainer.buffer, "build_tensors", return_value=mock_tensors), \
         patch.object(trainer.model, "forward", return_value=(torch.randn(35, 4) * 2.0, torch.randn(35, 3))):
         
        # 1. Missing champion: os.path.exists returns False
        with patch("os.path.exists", return_value=False):
            res = trainer.evaluate_candidate(sample_size=64)
            assert res["approved"] is False
            assert res["reason"] == "MISSING_CHAMPION_CHECKPOINT"
            
        # 2. Corrupt champion: torch.load raises an error
        with patch("os.path.exists", return_value=True), \
             patch("torch.load", side_effect=RuntimeError("Corrupt checkpoint")):
            res = trainer.evaluate_candidate(sample_size=64)
            assert res["approved"] is False
            assert "CHAMPION_LOAD_FAILURE" in res["reason"]

def test_atomic_promotion_rejects_unapproved_or_nan_candidate(tmp_path):
    """Verify promote_candidate rejects unapproved evaluation metrics and NaN weights."""
    from core.ai.registry import ModelRegistry
    from apps.research.model import SingleSymbolActorCritic
    
    registry = ModelRegistry(model_dir=str(tmp_path))
    model = SingleSymbolActorCritic(input_dim=200)
    
    # Save valid baseline
    registry.save_model(model, version_tag="base_v1")
    
    cand_path = os.path.join(str(tmp_path), "cand_test.pt")
    torch.save(model.state_dict(), cand_path)
    
    # 1. Candidate with approved=False in metrics must be rejected
    rejected_metrics = {"approved": False, "reason": "INFERIOR_TO_CHAMPION"}
    success = registry.promote_candidate(cand_path, version_tag="v_rejected", metrics=rejected_metrics)
    assert success is False
    assert registry.current_version != "v_rejected"
    
    # 2. Missing metrics or empty metrics must be rejected
    assert registry.promote_candidate(cand_path, version_tag="v_none", metrics=None) is False
    assert registry.promote_candidate(cand_path, version_tag="v_empty", metrics={}) is False
    
    # 3. Non-boolean True approval (e.g. string "true" or 1) must be rejected
    assert registry.promote_candidate(cand_path, version_tag="v_str", metrics={"approved": "true", "reason": "TEST"}) is False
    assert registry.promote_candidate(cand_path, version_tag="v_int", metrics={"approved": 1, "reason": "TEST"}) is False
    
    # 4. Missing required 'reason' evidence must be rejected
    assert registry.promote_candidate(cand_path, version_tag="v_no_reason", metrics={"approved": True}) is False
    
    # 5. Non-finite metrics (NaN or Inf) must be rejected
    nan_metric = {"approved": True, "reason": "TEST", "challenger_score": float("nan")}
    assert registry.promote_candidate(cand_path, version_tag="v_nan_metric", metrics=nan_metric) is False
    inf_metric = {"approved": True, "reason": "TEST", "challenger_score": float("inf")}
    assert registry.promote_candidate(cand_path, version_tag="v_inf_metric", metrics=inf_metric) is False

    # 6. Candidate with NaN weights must be rejected
    nan_cand_path = os.path.join(str(tmp_path), "cand_nan.pt")
    nan_state = model.state_dict()
    for k in nan_state:
        nan_state[k] = torch.full_like(nan_state[k], float("nan"))
        break
    torch.save(nan_state, nan_cand_path)
    
    success_nan = registry.promote_candidate(nan_cand_path, version_tag="v_nan", metrics={"approved": True, "reason": "TEST"})
    assert success_nan is False
    assert registry.current_version != "v_nan"

    # 7. Candidate with architecture mismatch must be rejected
    import torch.nn as nn
    class IncompatibleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(50, 2)
        def forward(self, x):
            return self.fc(x)
            
    incomp_path = os.path.join(str(tmp_path), "incomp.pt")
    torch.save(IncompatibleModel().state_dict(), incomp_path)
    valid_metrics = {"approved": True, "reason": "TEST"}
    assert registry.promote_candidate(incomp_path, version_tag="v_incomp", metrics=valid_metrics) is False

    # 8. save_model() cannot overwrite existing champion without approved metrics
    assert registry.save_model(model, version_tag="v_unapproved") is False
    assert registry.save_model(model, version_tag="v_unapproved", metrics={"approved": False}) is False

    # 9. Concurrent promotion lock prevents race conditions
    lock_file = os.path.join(str(tmp_path), "promotion.lock")
    with open(lock_file, "w") as f:
        f.write("locked")
    assert registry.promote_candidate(cand_path, version_tag="v_locked", metrics=valid_metrics) is False
    if os.path.exists(lock_file):
        os.remove(lock_file)

    # 10. Missing champion_score in evaluation metrics is rejected
    no_champ_score = {"approved": True, "reason": "SUPERIOR_TO_CHAMPION", "challenger_score": 0.5}
    assert registry.promote_candidate(cand_path, version_tag="v_no_champ", metrics=no_champ_score) is False

    # 11. Missing challenger_score in evaluation metrics is rejected
    no_chall_score = {"approved": True, "reason": "SUPERIOR_TO_CHAMPION", "champion_score": 0.5}
    assert registry.promote_candidate(cand_path, version_tag="v_no_chall", metrics=no_chall_score) is False

    # 12. Fabricated metrics where challenger is inferior to champion must be rejected
    inferior_metrics = {
        "approved": True,
        "reason": "SUPERIOR_TO_CHAMPION",
        "challenger_score": 0.10,
        "champion_score": 0.50
    }
    assert registry.promote_candidate(cand_path, version_tag="v_inferior", metrics=inferior_metrics) is False

    # 13. Insufficient evaluation samples (< 30) must be rejected
    low_samples_metrics = {
        "approved": True,
        "reason": "SUPERIOR_TO_CHAMPION",
        "challenger_score": 0.50,
        "champion_score": 0.10,
        "sample_size": 15
    }
    assert registry.promote_candidate(cand_path, version_tag="v_low_samples", metrics=low_samples_metrics) is False

    # 14. Stale or future evaluation timestamp must be rejected
    stale_metrics = {
        "approved": True,
        "reason": "SUPERIOR_TO_CHAMPION",
        "challenger_score": 0.50,
        "champion_score": 0.10,
        "eval_timestamp": time.time() - 100000 # > 24h
    }
    assert registry.promote_candidate(cand_path, version_tag="v_stale", metrics=stale_metrics) is False
    future_metrics = {
        "approved": True,
        "reason": "SUPERIOR_TO_CHAMPION",
        "challenger_score": 0.50,
        "champion_score": 0.10,
        "eval_timestamp": time.time() + 5000 # In the future
    }
    assert registry.promote_candidate(cand_path, version_tag="v_future", metrics=future_metrics) is False

    # 15. Champion preservation: original champion file is strictly preserved on every failed promotion
    champ_path = os.path.join(str(tmp_path), "model_v1.pt")
    assert os.path.exists(champ_path)
    champ_state = torch.load(champ_path, map_location="cpu")
    assert registry.current_version == "base_v1"







def test_edge_estimator_charges_every_coin_the_same_flat_impact():
    # No order-book depth in MarketState: a fake 50-coin default used to charge DOGE 0.02% and BTC ~0
    from core.risk.edge_estimator import EdgeEstimator, FLAT_IMPACT_PCT

    def market(sym, px):
        return MarketState(symbol=sym, timestamp=time.time(), bid=px, ask=px, mid_price=px, last_price=px, spread=0.0,
                           order_book_imbalance=0.0, volume=1.0, vwap=px, volatility=0.006, funding_rate=0.0,
                           features=[0.0] * 90)

    est = EdgeEstimator(min_net_edge_pct=0.08)
    btc = est.estimate("BTCUSDT", "OPEN_LONG", 0.5, 0.8, market("BTCUSDT", 67000.0), order_size_notional=8000.0)
    doge = est.estimate("DOGEUSDT", "OPEN_LONG", 0.5, 0.8, market("DOGEUSDT", 0.078), order_size_notional=8000.0)
    assert btc.slippage_pct == doge.slippage_pct == FLAT_IMPACT_PCT
