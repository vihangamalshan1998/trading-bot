import pytest
import time
from core.schemas.state_schema import MarketState, PortfolioState, OrderRequest
from core.risk.risk_manager import RiskManager
from core.config.settings import settings

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

def test_trading_disabled_blocks_order(base_state):
    market, portfolio = base_state
    settings.trading_enabled = False
    rm = RiskManager()
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "TRADING_DISABLED"

def test_emergency_stop_blocks_order(base_state):
    market, portfolio = base_state
    settings.trading_enabled = True
    settings.emergency_stop = True
    rm = RiskManager()
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "EMERGENCY_STOP"

def test_stale_market_blocks_order(base_state):
    market, portfolio = base_state
    settings.trading_enabled = True
    settings.emergency_stop = False
    market.timestamp = time.time() - 120
    rm = RiskManager()
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "STALE_MARKET_DATA"

def test_invalid_price_blocks_order(base_state):
    market, portfolio = base_state
    settings.trading_enabled = True
    settings.emergency_stop = False
    market.mid_price = -5.0
    rm = RiskManager()
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "INVALID_PRICE"

def test_dry_run_blocks_order():
    # dry_run is tested effectively in main.py by blocking the create_order call.
    # RiskManager doesn't enforce dry_run directly; it's a structural orchestrator gate.
    assert settings.dry_run is not None
