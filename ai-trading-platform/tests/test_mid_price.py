import pytest
from core.schemas.state_schema import MarketState

def test_mid_price_not_feature_index():
    state = MarketState(
        symbol="BTCUSDT", timestamp=1000.0, bid=99.0, ask=101.0, 
        mid_price=100.0, last_price=100.0, spread=2.0, order_book_imbalance=0.1, 
        volume=1000.0, vwap=100.0, volatility=0.5, funding_rate=0.001, 
        features=[float(i) for i in range(25)]
    )
    assert state.mid_price == 100.0
    assert state.features[6] == 6.0
    assert state.mid_price != state.features[6]
    
    state.features.reverse()
    assert state.mid_price == 100.0
