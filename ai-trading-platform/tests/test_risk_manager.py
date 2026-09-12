import pytest
import time
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

def test_O_configuration_is_used(base_state):
    import core.risk.risk_manager as rm_module
    # Change settings
    rm_module.settings.max_order_size = 999.0
    rm_module.settings.max_leverage = 777
    rm_module.settings.max_market_data_age_seconds = 123.0
    
    # Initialize RM
    rm = RiskManager()
    
    # Assert it grabbed the settings
    assert rm.max_order_size == 999.0
    assert rm.max_leverage == 777
    assert rm.max_market_data_age_seconds == 123.0

def test_A_trading_disabled(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.trading_enabled = False
    rm = RiskManager()
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_B_emergency_stop(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.emergency_stop = True
    rm = RiskManager()
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_C_stale_market_data(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_market_data_age_seconds = 60.0
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
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_order_size = 5.0
    rm = RiskManager()
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=10.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_G_maximum_leverage(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_leverage = 2
    rm_module.settings.max_order_size = 500.0 # Bypass order size limit
    rm_module.settings.max_position_size = 500.0 # Bypass position size limit
    
    # Portfolio equity is 10,000. Notional size for max_leverage 2 is 20,000.
    portfolio.free_margin = 999999.0 # Bypass free margin limit to test effective leverage
    rm = RiskManager()
    
    # Requesting 250 BTC @ 100.5 = 25,125 notional. Effective leverage > 2.5
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=250.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "EXCESSIVE_PORTFOLIO_EXPOSURE" or decision.reason == "INSUFFICIENT_FREE_MARGIN"
    assert decision.adjusted_quantity == 0.0

def test_P_max_position_size_clamping(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_position_size = 5.0
    portfolio.positions["BTCUSDT"] = PositionState(symbol="BTCUSDT", quantity=3.0, leverage=1)
    
    rm = RiskManager()
    
    # Existing = 3. Requesting = 3. Should clamp to 2.
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=3.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "MAX_POSITION_SIZE_EXCEEDED"
    assert decision.adjusted_quantity == 0.0

def test_Q_max_position_size_rejection(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_position_size = 5.0
    portfolio.positions["BTCUSDT"] = PositionState(symbol="BTCUSDT", quantity=6.0, leverage=1)
    
    rm = RiskManager()
    
    # Existing = 6. Requesting = 2. Allowed = 5. Since safe qty becomes <= 0 (0), hard reject.
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=2.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "MAX_POSITION_SIZE_EXCEEDED"
    assert decision.adjusted_quantity == 0.0

def test_H_maximum_portfolio_exposure(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_portfolio_exposure_pct = 0.50
    rm_module.settings.max_order_size = 500.0 # Bypass order size limit
    rm_module.settings.max_position_size = 500.0 # Bypass position size limit
    
    rm = RiskManager()
    # Portfolio exposure already near limit (e.g. 4000 out of 5000 max)
    portfolio.total_exposure = 4000.0
    # Requesting another 15 BTC * 100.5 = ~1507. Total = 5507 > 5000.
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=15.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_I_maximum_open_positions(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_open_positions = 1
    rm = RiskManager()
    portfolio.positions["ETHUSDT"] = PositionState(symbol="ETHUSDT", quantity=1.0, leverage=1)
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0

def test_M_daily_loss(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_daily_loss_pct = 0.05
    rm = RiskManager()
    rm.daily_high_equity = 10000.0
    portfolio.equity = 9000.0 # 10% daily loss
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "MAX_DAILY_LOSS"
    assert decision.adjusted_quantity == 0.0

def test_N_overall_drawdown(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_drawdown_pct = 0.10
    rm = RiskManager()
    rm.daily_high_equity = 8000.0
    rm.global_high_equity = 10000.0
    portfolio.equity = 8000.0 # 20% drawdown
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "MAX_DRAWDOWN"
    assert decision.adjusted_quantity == 0.0

def test_K_correlated_exposure(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.correlated_exposure_limit_pct = 0.40
    rm_module.settings.max_open_positions = 10
    rm = RiskManager()
    
    # Bypass other limits
    rm.max_portfolio_exposure_pct = 1.0
    
    # Set up existing highly correlated position (e.g. 3500 value out of 10000 equity)
    portfolio.positions["ETHUSDT"] = PositionState(symbol="ETHUSDT", quantity=1.0, current_price=3500.0, leverage=1)
    
    # Request another 1000 in BTCUSDT -> Total correlated = 4500 > 4000 (40% limit)
    # Using mid_price = 100.5, request 10 => 1005 notional
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=10.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    
    assert not decision.approved
    assert decision.reason == "CORRELATED_EXPOSURE_LIMIT"

def test_L_zero_quantity_after_clamping(base_state):
    market, portfolio = base_state
    import core.risk.risk_manager as rm_module
    rm_module.settings.max_symbol_exposure_pct = 0.0 # Forces clamp to 0
    rm = RiskManager()
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=10.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0
