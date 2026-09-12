import pytest
import time
from core.schemas.state_schema import MarketState, PortfolioState, PositionState, OrderRequest
from core.risk.risk_manager import RiskManager

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

def test_risk_rejects_excessive_leverage(base_state):
    # F. Risk rejects excessive leverage.
    market, portfolio = base_state
    portfolio.positions["BTCUSDT"] = PositionState(symbol="BTCUSDT", quantity=1.0, leverage=20)
    rm = RiskManager(max_leverage=10)
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "MAX_LEVERAGE_EXCEEDED"
    assert decision.adjusted_quantity == 0.0

def test_risk_rejects_excessive_portfolio_exposure(base_state):
    # G. Risk rejects excessive portfolio exposure.
    market, portfolio = base_state
    rm = RiskManager(max_portfolio_exposure_pct=0.50, max_order_size=1000.0)
    # Requesting 100 BTC * 100.5 = 10,050. Equity is 10,000. Max is 5,000.
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=100.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "EXCESSIVE_PORTFOLIO_EXPOSURE"
    assert decision.adjusted_quantity == 0.0

def test_risk_clamps_symbol_exposure_only_when_resulting_quantity_remains_valid(base_state):
    # J. Risk clamps symbol exposure only when the resulting quantity remains valid.
    market, portfolio = base_state
    rm = RiskManager(max_symbol_exposure_pct=0.50, max_portfolio_exposure_pct=1.0, max_order_size=1000.0)
    # Equity 10000, Max Symbol = 5000. Requested 60 BTC @ 100.5 = ~6030. Should clamp to ~49 BTC.
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=60.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert decision.approved
    assert "MAX_SYMBOL_EXPOSURE" in decision.risk_flags
    assert 0.0 < decision.adjusted_quantity < 60.0

def test_risk_rejects_when_safe_quantity_becomes_zero(base_state):
    # K. Risk rejects when safe quantity becomes zero.
    market, portfolio = base_state
    # Set symbol exposure to 0.0 to force clamp to 0
    rm = RiskManager(max_symbol_exposure_pct=0.0, max_portfolio_exposure_pct=1.0, max_order_size=1000.0)
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=60.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.adjusted_quantity == 0.0
    assert decision.reason == "INVALID_QUANTITY"
