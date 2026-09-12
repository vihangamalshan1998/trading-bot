import pytest
import time
from core.schemas.state_schema import MarketState, PortfolioState, OrderRequest
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
        wallet_balance=100.0, equity=100.0, used_margin=0.0, 
        free_margin=100.0, total_unrealized_pnl=0.0, total_exposure=0.0, 
        positions={}
    )
    return market, portfolio

def test_safety_settings_default_safely():
    # C. Safety settings default safely
    s = Settings()
    assert s.trading_enabled is False
    assert s.dry_run is True
    assert s.allow_testnet is False
    assert s.allow_live_trading is False
    assert s.emergency_stop is True

def test_risk_rejects_stale_market_data(base_state):
    # D. Risk rejects stale market data.
    market, portfolio = base_state
    market.timestamp = time.time() - 120
    rm = RiskManager()
    
    # We must mock the global settings so it doesn't fail on emergency_stop
    import core.risk.risk_manager as rm_module
    old_es = rm_module.settings.emergency_stop
    old_te = rm_module.settings.trading_enabled
    rm_module.settings.emergency_stop = False
    rm_module.settings.trading_enabled = True
    
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "STALE_MARKET_DATA"
    
    rm_module.settings.emergency_stop = old_es
    rm_module.settings.trading_enabled = old_te

def test_risk_rejects_invalid_zero_price(base_state):
    # E. Risk rejects invalid/zero price.
    market, portfolio = base_state
    market.mid_price = 0.0
    rm = RiskManager()
    
    import core.risk.risk_manager as rm_module
    old_es = rm_module.settings.emergency_stop
    old_te = rm_module.settings.trading_enabled
    rm_module.settings.emergency_stop = False
    rm_module.settings.trading_enabled = True
    
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "INVALID_PRICE"
    
    rm_module.settings.emergency_stop = old_es
    rm_module.settings.trading_enabled = old_te

def test_risk_rejects_when_emergency_stop_is_active(base_state):
    # H. Risk rejects when emergency stop is active.
    market, portfolio = base_state
    rm = RiskManager()
    
    import core.risk.risk_manager as rm_module
    old_es = rm_module.settings.emergency_stop
    old_te = rm_module.settings.trading_enabled
    rm_module.settings.emergency_stop = True
    rm_module.settings.trading_enabled = True
    
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "EMERGENCY_STOP"
    
    rm_module.settings.emergency_stop = old_es
    rm_module.settings.trading_enabled = old_te

def test_risk_rejects_when_trading_is_disabled(base_state):
    # I. Risk rejects when trading is disabled.
    market, portfolio = base_state
    rm = RiskManager()
    
    import core.risk.risk_manager as rm_module
    old_te = rm_module.settings.trading_enabled
    rm_module.settings.trading_enabled = False
    
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "TRADING_DISABLED"
    
    rm_module.settings.trading_enabled = old_te
