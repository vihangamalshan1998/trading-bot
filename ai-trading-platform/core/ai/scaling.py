import numpy as np

def scale_portfolio_state(
    wallet_balance: float, 
    equity: float, 
    used_margin: float, 
    free_margin: float, 
    total_unrealized_pnl: float,
    total_exposure: float,
    initial_balance: float,
    max_equity: float
) -> np.ndarray:
    """
    Normalizes portfolio metrics relative to the initial/max equity to ensure 
    the neural network processes relative percentages rather than raw dollar amounts.
    Returns a 9-dim array matching PortfolioState size.
    """
    # 1. Growth relative to start
    growth_pct = (equity - initial_balance) / initial_balance if initial_balance > 0 else 0.0
    
    # 2. Drawdown from peak
    drawdown_pct = (equity - max_equity) / max_equity if max_equity > 0 else 0.0
    
    # 3. Margin utilization
    margin_utilization = used_margin / equity if equity > 0 else 0.0
    
    # 4. Free margin ratio
    free_margin_ratio = free_margin / equity if equity > 0 else 0.0
    
    # 5. PnL ratio
    pnl_ratio = total_unrealized_pnl / equity if equity > 0 else 0.0
    
    # 6. Total exposure ratio (Gross Leverage)
    exposure_ratio = total_exposure / equity if equity > 0 else 0.0
    
    # Return 9-dim to match the expected PortfolioState output
    return np.array([
        growth_pct,           # wallet_balance equivalent
        drawdown_pct,         # available_balance equivalent
        exposure_ratio,       # equity equivalent
        margin_utilization,   # used_margin
        free_margin_ratio,    # free_margin
        pnl_ratio,            # total_unrealized_pnl
        exposure_ratio,       # total_exposure (duplicate for padding/compatibility)
        0.0,                  # long_exposure (placeholder)
        0.0                   # short_exposure (placeholder)
    ], dtype=np.float32)

def scale_position_state(
    side: str,
    quantity: float,
    entry_price: float,
    current_price: float,
    unrealized_pnl: float,
    realized_pnl: float,
    leverage: float,
    margin: float,
    liquidation_price: float,
    funding_net: float,
    portfolio_equity: float
) -> np.ndarray:
    """
    Normalizes a single position's state relative to current price and portfolio equity.
    Returns a 12-dim array matching PositionState size.
    """
    side_val = 0.0 if side == "FLAT" else (1.0 if side == "LONG" else -1.0)
    
    # 1. Size relative to portfolio
    notional = abs(quantity) * current_price
    size_ratio = notional / portfolio_equity if portfolio_equity > 0 else 0.0
    
    # 2. Entry distance
    entry_dist = (current_price - entry_price) / entry_price if entry_price > 0 and side != "FLAT" else 0.0
    
    # 3. Liquidation distance
    liq_dist = (current_price - liquidation_price) / current_price if liquidation_price > 0 and side != "FLAT" else 0.0
    
    # 4. PnL ratios
    unrealized_pnl_ratio = unrealized_pnl / portfolio_equity if portfolio_equity > 0 else 0.0
    realized_pnl_ratio = realized_pnl / portfolio_equity if portfolio_equity > 0 else 0.0
    
    # 5. Margin ratio
    margin_ratio = margin / portfolio_equity if portfolio_equity > 0 else 0.0
    
    # 6. Normalized leverage (Assuming 125x max)
    norm_leverage = leverage / 125.0
    
    # 7. Funding impact
    funding_impact = funding_net / portfolio_equity if portfolio_equity > 0 else 0.0
    
    return np.array([
        side_val,
        size_ratio,          # quantity equivalent
        entry_dist,          # entry_price equivalent
        0.0,                 # current_price equivalent (handled by market features)
        unrealized_pnl_ratio,
        realized_pnl_ratio,
        norm_leverage,       # leverage
        margin_ratio,        # margin
        margin_ratio,        # initial_margin
        margin_ratio * 0.5,  # maintenance_margin (approx)
        liq_dist,            # liquidation_price equivalent
        funding_impact       # funding_paid - received
    ], dtype=np.float32)
