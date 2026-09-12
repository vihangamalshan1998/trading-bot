import pytest
import time
from core.schemas.state_schema import MarketState, PortfolioState, OrderRequest
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
        wallet_balance=100.0, equity=100.0, used_margin=0.0, 
        free_margin=100.0, total_unrealized_pnl=0.0, total_exposure=0.0, 
        positions={}
    )
    return market, portfolio

def test_order_blocked_when_trading_disabled(base_state):
    market, portfolio = base_state
    rm = RiskManager(trading_enabled=False)
    request = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is False
    assert decision.reason == "TRADING_DISABLED"

def test_order_blocked_in_dry_run(base_state):
    # In our architecture, dry_run is enforced at the orchestrator/bot level.
    # RiskManager handles TRADING_ENABLED. We test that if trading_enabled=False (set by bot on dry_run=True), it blocks.
    market, portfolio = base_state
    rm = RiskManager(trading_enabled=False) # Bot forces this on DRY_RUN
    request = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is False

def test_order_blocked_when_risk_rejects(base_state):
    market, portfolio = base_state
    market.mid_price = -100 # Invalid
    rm = RiskManager(trading_enabled=True)
    request = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is False

def test_order_blocked_when_market_data_stale(base_state):
    market, portfolio = base_state
    market.timestamp = time.time() - 120 # 2 minutes old
    rm = RiskManager(trading_enabled=True)
    request = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is False
    assert decision.reason == "STALE_MARKET_DATA"
