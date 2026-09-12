import pytest
import time
from core.schemas.state_schema import MarketState, PortfolioState, PositionState, OrderRequest
from core.risk.risk_manager import RiskManager
from core.config.settings import settings

@pytest.fixture
def base_state():
    settings.trading_enabled = True
    settings.emergency_stop = False
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

def test_max_leverage_blocks_order(base_state):
    market, portfolio = base_state
    # Position has 20x leverage, max is 10
    portfolio.positions["BTCUSDT"] = PositionState(symbol="BTCUSDT", quantity=1.0, leverage=20)
    rm = RiskManager(max_leverage=10)
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "MAX_LEVERAGE_EXCEEDED"

def test_max_order_size_blocks_order(base_state):
    market, portfolio = base_state
    rm = RiskManager(max_order_size=5.0)
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=10.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert decision.approved  # It clamps it, which is the designed behavior, but logs MAX_ORDER_SIZE_EXCEEDED
    assert decision.adjusted_quantity == 5.0
    assert "MAX_ORDER_SIZE_EXCEEDED" in decision.risk_flags

def test_max_portfolio_exposure_blocks_order(base_state):
    market, portfolio = base_state
    rm = RiskManager(max_portfolio_exposure_pct=0.50)
    # Requesting 100 BTC * 100.5 = 10,050. Equity is 10,000. Max is 5,000.
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=100.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert decision.approved
    assert "EXCESSIVE_PORTFOLIO_EXPOSURE" in decision.risk_flags
    assert decision.adjusted_quantity < 100.0 # Clamped

def test_max_open_positions_blocks_order(base_state):
    market, portfolio = base_state
    rm = RiskManager(max_open_positions=1)
    portfolio.positions["ETHUSDT"] = PositionState(symbol="ETHUSDT", quantity=1.0) # 1 active position
    req = OrderRequest(symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=time.time())
    decision = rm.evaluate(req, portfolio, market)
    assert not decision.approved
    assert decision.reason == "MAX_OPEN_POSITIONS"
