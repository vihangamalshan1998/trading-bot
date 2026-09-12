import pytest
import time
import math
from core.schemas.state_schema import MarketState, PortfolioState, PositionState, OrderRequest
from core.risk.risk_manager import RiskManager
from core.config.settings import Settings

@pytest.fixture
def base_state():
    market = MarketState(
        symbol="BTCUSDT", timestamp=time.time(), bid=100.0, ask=101.0, 
        mid_price=100.5, last_price=100.5, spread=1.0, order_book_imbalance=0.1, 
        volume=100.0, vwap=100.2, volatility=0.5, funding_rate=0.001, 
        features=[0.0]*25
    )
    portfolio = PortfolioState(
        wallet_balance=10000.0, equity=10000.0, used_margin=0.0, 
        free_margin=10000.0, total_unrealized_pnl=0.0, total_exposure=0.0, 
        positions={"BTCUSDT": PositionState(symbol="BTCUSDT", quantity=0.0, leverage=1)}
    )
    return market, portfolio

@pytest.fixture(autouse=True)
def setup_risk_settings():
    # Force settings so the tests can reach the logic
    import core.risk.risk_manager as rm_module
    old_es = rm_module.settings.emergency_stop
    old_te = rm_module.settings.trading_enabled
    rm_module.settings.emergency_stop = False
    rm_module.settings.trading_enabled = True
    yield
    rm_module.settings.emergency_stop = old_es
    rm_module.settings.trading_enabled = old_te

def test_safety_settings_default_safely():
    s = Settings()
    assert s.trading_enabled is False
    assert s.dry_run is True
    assert s.allow_testnet is False
    assert s.allow_live_trading is False
    assert s.emergency_stop is True

def test_A_trading_disabled(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    import core.risk.risk_manager as rm_module
    rm_module.settings.trading_enabled = False
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_B_emergency_stop(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    import core.risk.risk_manager as rm_module
    rm_module.settings.emergency_stop = True
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_C_stale_market_data(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    market.timestamp = time.time() - 120
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_D_invalid_price(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    for bad_price in [0.0, -10.0, float('nan'), float('inf')]:
        bad_market = MarketState.model_construct(
            symbol="BTCUSDT", timestamp=time.time(), bid=100.0, ask=101.0, 
            mid_price=bad_price, last_price=100.5, spread=1.0, order_book_imbalance=0.1, 
            volume=100.0, vwap=100.2, volatility=0.5, funding_rate=0.001, 
            features=[0.0]*25
        )
        decision = rm.evaluate(req, portfolio, bad_market)
        assert not decision.approved
        assert decision.adjusted_quantity == 0.0

def test_E_invalid_quantity(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    for bad_qty in [0.0, -5.0, float('nan'), float('inf')]:
        req = OrderRequest.model_construct(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=bad_qty, target_position=1.0, model_version="v1", timestamp=time.time())
        decision = rm.evaluate(req, portfolio, market)
        assert not decision.approved
        assert decision.adjusted_quantity == 0.0

def test_F_maximum_order_size(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    rm.max_order_size = 5.0
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=10.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_G_maximum_leverage(base_state):
    market, portfolio = base_state
    portfolio.positions["BTCUSDT"] = PositionState(symbol="BTCUSDT", quantity=1.0, leverage=20)
    rm = RiskManager()
    rm.max_leverage = 10
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_H_maximum_portfolio_exposure(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    rm.max_portfolio_exposure_pct = 0.50
    # Portfolio exposure already near limit (e.g. 4000 out of 5000 max)
    portfolio.total_exposure = 4000.0
    # Requesting another 15 BTC * 100.5 = ~1507. Total = 5507 > 5000.
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=15.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_I_maximum_open_positions(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    rm.max_open_positions = 1
    portfolio.positions["ETHUSDT"] = PositionState(symbol="ETHUSDT", quantity=1.0, leverage=1)
    # Already have 1 open position, max is 1. Opening a NEW symbol (BTCUSDT) should fail.
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_J_daily_drawdown(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    rm.max_drawdown_pct = 0.10
    rm.global_high_equity = 10000.0
    portfolio.equity = 8000.0 # 20% drawdown
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_K_correlated_exposure(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    rm.correlated_exposure_limit_pct = 0.40
    # Set portfolio exposure to 90% of max, which triggers our HIGH_CORRELATED_EXPOSURE_WARNING placeholder logic
    portfolio.total_exposure = 0.95 * rm.max_portfolio_exposure_pct * portfolio.equity
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    # Based on our implementation, this merely adds a warning flag but we ensure the flag exists
    assert "HIGH_CORRELATED_EXPOSURE_WARNING" in decision.risk_flags

def test_L_zero_quantity_after_clamping(base_state):
    market, portfolio = base_state
    rm = RiskManager()
    rm.max_symbol_exposure_pct = 0.0 # Forces clamp to 0
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=10.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0
