import pytest
from core.schemas.state_schema import MarketState

def test_mid_price_is_not_feature_index():
    """
    Proves that modifying the feature vector ordering does not change execution logic
    because mid_price is an explicitly named field, not an array index like features[6].
    """
    # Create market state with explicit mid price
    state = MarketState(
        symbol="BTCUSDT", timestamp=1000.0, bid=99.0, ask=101.0, 
        mid_price=100.0, last_price=100.0, spread=2.0, order_book_imbalance=0.1, 
        volume=1000.0, vwap=100.0, volatility=0.5, funding_rate=0.001, 
        features=[float(i) for i in range(25)] # feature[6] is 6.0
    )
    
    assert state.mid_price == 100.0
    assert state.features[6] == 6.0
    assert state.mid_price != state.features[6]
    
    # Modify feature vector ordering
    state.features.reverse()
    
    # mid_price must remain unchanged
    assert state.mid_price == 100.0
