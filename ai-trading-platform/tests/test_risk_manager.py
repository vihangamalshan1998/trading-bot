import pytest
from core.schemas.state_schema import MarketState, PortfolioState, OrderRequest, RiskDecision
from core.risk.risk_manager import RiskManager

def test_risk_manager_daily_drawdown():
    rm = RiskManager(max_daily_drawdown_pct=0.05)
    
    # Setup state
    market = MarketState(
        symbol="BTCUSDT", timestamp=1000, bid=100, ask=101, mid_price=100.5,
        last_price=100.5, spread=1, order_book_imbalance=0.0, volume=10,
        vwap=100.5, volatility=0.1, funding_rate=0.0, features=[0]*25
    )
    
    # Portfolio drawn down by 6% (initial high was updated internally to 10000 on first check)
    portfolio = PortfolioState(
        wallet_balance=9400, equity=9400, used_margin=0, free_margin=9400,
        total_unrealized_pnl=0, total_exposure=0, positions={}
    )
    
    # First check establishes the high water mark
    rm.daily_high_equity = 10000.0
    
    request = OrderRequest(
        symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9,
        requested_quantity=1.0, target_position=1.0, model_version="v1", timestamp=1000
    )
    
    decision = rm.evaluate(request, portfolio, market)
    
    assert decision.approved is False
    assert decision.adjusted_quantity == 0.0
    assert "DAILY_DRAWDOWN_BREACH" in decision.risk_flags

def test_risk_manager_max_symbol_exposure():
    rm = RiskManager(max_symbol_exposure_pct=0.20)
    
    market = MarketState(
        symbol="BTCUSDT", timestamp=1000, bid=100, ask=101, mid_price=10000.0,
        last_price=10000.0, spread=1, order_book_imbalance=0.0, volume=10,
        vwap=10000.0, volatility=0.1, funding_rate=0.0, features=[0]*25
    )
    
    portfolio = PortfolioState(
        wallet_balance=100000, equity=100000, used_margin=0, free_margin=100000,
        total_unrealized_pnl=0, total_exposure=0, positions={}
    )
    
    # Requesting 3 BTC = $30,000 exposure (30% > 20% limit)
    request = OrderRequest(
        symbol="BTCUSDT", action_type="OPEN_LONG", confidence=0.9,
        requested_quantity=3.0, target_position=3.0, model_version="v1", timestamp=1000
    )
    
    decision = rm.evaluate(request, portfolio, market)
    
    assert decision.approved is True
    # Max allowed should be 20% of 100,000 = $20,000. 20,000 / 10000 = 2.0 BTC
    assert decision.adjusted_quantity == 2.0
    assert "MAX_SYMBOL_EXPOSURE" in decision.risk_flags
