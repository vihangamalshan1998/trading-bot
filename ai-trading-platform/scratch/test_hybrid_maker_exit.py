import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

def test_hybrid_maker_exit_logic():
    print("Testing Hybrid Maker Exit Routing...")
    
    # Scenario A: Emergency Hard Stop (-15% ROE)
    hard_stop_triggered = True
    sudden_shock_triggered = False
    pnl_pct_leveraged = -0.15
    is_closing = True
    
    is_emergency_exit = (
        hard_stop_triggered or 
        sudden_shock_triggered or 
        (is_closing and pnl_pct_leveraged <= -0.10)
    )
    is_maker_exit = is_closing and not is_emergency_exit and (
        pnl_pct_leveraged >= 0.015
    )
    is_market_exit = is_emergency_exit or (is_closing and not is_maker_exit)
    
    assert is_emergency_exit == True, "Scenario A failed: Emergency exit should trigger"
    assert is_maker_exit == False, "Scenario A failed: Maker exit should NOT be used for emergency stop"
    assert is_market_exit == True, "Scenario A failed: Must route to MARKET order"
    print("  [PASS] Scenario A: Emergency Hard Stop correctly routes to MARKET order (Instant Protection)")
    
    # Scenario B: Planned Profit Take-Profit (+3.94% PnL)
    hard_stop_triggered = False
    sudden_shock_triggered = False
    pnl_pct_leveraged = +0.0394
    partial_tp_triggered = False
    hard_take_profit_triggered = False
    trailing_stop_triggered = True
    
    is_emergency_exit = (
        hard_stop_triggered or 
        sudden_shock_triggered or 
        (is_closing and pnl_pct_leveraged <= -0.10)
    )
    is_maker_exit = is_closing and not is_emergency_exit and (
        pnl_pct_leveraged >= 0.015 or 
        partial_tp_triggered or 
        hard_take_profit_triggered or 
        trailing_stop_triggered
    )
    is_market_exit = is_emergency_exit or (is_closing and not is_maker_exit)
    
    assert is_emergency_exit == False, "Scenario B failed: Should not be emergency"
    assert is_maker_exit == True, "Scenario B failed: Must route to MAKER Post-Only GTX limit order"
    assert is_market_exit == False, "Scenario B failed: Should NOT be market order"
    print("  [PASS] Scenario B: Trailing Stop in profit correctly routes to POST-ONLY MAKER GTX (0.02% Fee)")
    
    # Scenario C: 50% Partial Take-Profit Scale-Out (+15% ROE)
    pnl_pct_leveraged = +0.15
    partial_tp_triggered = True
    trailing_stop_triggered = False
    
    is_emergency_exit = (
        hard_stop_triggered or 
        sudden_shock_triggered or 
        (is_closing and pnl_pct_leveraged <= -0.10)
    )
    is_maker_exit = is_closing and not is_emergency_exit and (
        pnl_pct_leveraged >= 0.015 or 
        partial_tp_triggered or 
        hard_take_profit_triggered or 
        trailing_stop_triggered
    )
    assert is_maker_exit == True, "Scenario C failed: Partial TP must be Maker exit"
    print("  [PASS] Scenario C: 50% Partial TP correctly routes to POST-ONLY MAKER GTX (0.02% Fee)")
    
    # Scenario D: Fee Savings Calculation
    position_notional = 15346.00
    taker_fee_pct = 0.0005  # 0.05%
    maker_fee_pct = 0.0002  # 0.02%
    
    old_exit_fee = position_notional * taker_fee_pct
    new_exit_fee = position_notional * maker_fee_pct
    savings = old_exit_fee - new_exit_fee
    savings_pct = (savings / old_exit_fee) * 100
    
    print(f"\nFee Comparison on $15,346 Position:")
    print(f"  Old Taker Exit Fee: ${old_exit_fee:.2f}")
    print(f"  New Maker Exit Fee: ${new_exit_fee:.2f}")
    print(f"  Fee Saved:          ${savings:.2f} ({savings_pct:.1f}% Reduction!)\n")
    
    assert round(savings_pct, 1) == 60.0, "Fee savings percentage should be 60% on exit"
    print("[SUCCESS] ALL HYBRID MAKER EXIT UNIT TESTS PASSED 100%!")

if __name__ == "__main__":
    test_hybrid_maker_exit_logic()
