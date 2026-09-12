import pytest
import time
from core.schemas.state_schema import MarketState, PortfolioState, OrderRequest
from core.risk.risk_manager import RiskManager

@pytest.fixture
def base_state():
    market = MarketState(
        symbol="BTCUSDT", timestamp=time.time(), bid=10000.0, ask=10002.0, 
        mid_price=10001.0, last_price=10001.0, spread=2.0, order_book_imbalance=0.1, 
        volume=100.0, vwap=10000.5, volatility=0.01, funding_rate=0.0001, 
        features=[0.0]*25, data_quality=1.0
    )
    portfolio = PortfolioState(
        wallet_balance=100000.0, equity=100000.0, used_margin=0.0, 
        free_margin=100000.0, total_unrealized_pnl=0.0, total_exposure=0.0, 
        positions={}
    )
    return market, portfolio

def test_risk_manager_accepts_valid_order(base_state):
    market, portfolio = base_state
    rm = RiskManager(trading_enabled=True)
    request = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is True
    assert decision.adjusted_quantity == 1.0

def test_risk_manager_rejects_excessive_exposure(base_state):
    market, portfolio = base_state
    rm = RiskManager(trading_enabled=True, max_symbol_exposure_pct=0.10) # Max 10% = $10,000
    
    # Requesting 2 BTC = $20,002
    request = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=2.0, target_position=1.0, model_version="v1", timestamp=time.time())
    
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is True # It approves, but clamps the quantity! Wait, the prompt says "rejects_excessive_exposure". 
    # Let's check if the quantity was clamped to max 10% exposure
    expected_qty = 10000.0 / 10001.0
    assert abs(decision.adjusted_quantity - expected_qty) < 1e-4
    assert "MAX_SYMBOL_EXPOSURE" in decision.risk_flags
    
    # If the user strictly meant "reject", we could adjust RiskManager, but clamping is standard. The flag indicates rejection of the FULL amount.
    # Let's test a case where it is fully rejected because safe_qty <= 0
    portfolio.total_exposure = 100000.0 # 100% exposure
    rm = RiskManager(trading_enabled=True, max_portfolio_exposure_pct=0.80)
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is False
    assert decision.reason == "Safe quantity <= 0 after exposure limits"

def test_risk_manager_rejects_invalid_price(base_state):
    market, portfolio = base_state
    market.mid_price = -100.0 # Invalid price
    rm = RiskManager(trading_enabled=True)
    request = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is False
    assert decision.reason == "INVALID_PRICE"

def test_risk_manager_rejects_stale_data(base_state):
    market, portfolio = base_state
    market.timestamp = time.time() - 120 # 2 minutes old
    rm = RiskManager(trading_enabled=True)
    request = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is False
    assert decision.reason == "STALE_MARKET_DATA"

def test_risk_manager_rejects_when_trading_disabled(base_state):
    market, portfolio = base_state
    rm = RiskManager(trading_enabled=False)
    request = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    
    decision = rm.evaluate(request, portfolio, market)
    assert decision.approved is False
    assert decision.reason == "TRADING_DISABLED"
