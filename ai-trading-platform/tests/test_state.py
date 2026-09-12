import pytest
import torch
import time
from core.schemas.state_schema import MarketState, PortfolioState, PositionState
from core.schemas.dimension_config import get_expected_observation_dimension, validate_observation
from core.config.settings import settings

def test_market_state():
    state = MarketState(
        symbol="BTCUSDT", timestamp=time.time(), bid=100.0, ask=101.0, 
        mid_price=100.5, last_price=100.5, spread=1.0, order_book_imbalance=0.1, 
        volume=1000.0, vwap=100.2, volatility=0.5, funding_rate=0.001, 
        features=[0.0] * 25
    )
    assert state.mid_price == 100.5
    assert len(state.features) == 25

def test_portfolio_state():
    state = PortfolioState(
        wallet_balance=100.0, equity=100.0, used_margin=0.0, 
        free_margin=100.0, total_unrealized_pnl=0.0, total_exposure=0.0, 
        positions={"BTCUSDT": PositionState(symbol="BTCUSDT")}
    )
    assert state.equity == 100.0
    assert "BTCUSDT" in state.positions

def test_observation_dimension_validation():
    num_symbols = len(settings.symbol_universe)
    expected_dim = get_expected_observation_dimension(num_symbols)
    obs = torch.zeros((1, expected_dim))
    
    assert validate_observation(obs, expected_dim, settings.symbol_universe) is True

def test_observation_rejects_nan():
    num_symbols = len(settings.symbol_universe)
    expected_dim = get_expected_observation_dimension(num_symbols)
    obs = torch.zeros((1, expected_dim))
    obs[0, 0] = float('nan')
    
    with pytest.raises(ValueError, match="NaN detected"):
        validate_observation(obs, expected_dim, settings.symbol_universe)

def test_observation_rejects_inf():
    num_symbols = len(settings.symbol_universe)
    expected_dim = get_expected_observation_dimension(num_symbols)
    obs = torch.zeros((1, expected_dim))
    obs[0, 0] = float('inf')
    
    with pytest.raises(ValueError, match="Inf detected"):
        validate_observation(obs, expected_dim, settings.symbol_universe)
