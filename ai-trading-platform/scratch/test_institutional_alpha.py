import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.schemas.state_schema import MarketState
from core.ai.institutional_alpha import InstitutionalAlphaEngine

def test_all_6_pillars():
    symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "NEARUSDT"]
    engine = InstitutionalAlphaEngine(symbols=symbols)
    
    # 1. Test Pillar 1: Stop-Hunt Mapping
    print("Testing Pillar 1: Stop-Hunt Mapping...")
    features = np.zeros(90, dtype=np.float32)
    m_xrp = MarketState(
        symbol="XRPUSDT",
        timestamp=time.time(),
        last_price=0.5500,
        mid_price=0.5500,
        bid=0.5499,
        ask=0.5501,
        spread=0.0002,
        volume=100000.0,
        volatility=0.015,
        funding_rate=0.0001,
        order_book_imbalance=0.25, # High buy book skew near high
        vwap=0.5450,
        features=features
    )
    # Populate price history reaching up to 0.5510 (0.5500 is within 0.18% of high)
    for p in np.linspace(0.5400, 0.5510, 150):
        engine.price_history["XRPUSDT"].append(p)
        
    ok, reason = engine.evaluate_stop_hunt("XRPUSDT", m_xrp, "SHORT")
    assert not ok, f"Pillar 1 should have vetoed Short near swing high: {reason}"
    print(f"  [PASS] Pillar 1 Vetoed Short: {reason}")
    
    # 2. Test Pillar 2: Open Interest Delta Divergence
    print("Testing Pillar 2: Open Interest Divergence...")
    features_oi = np.zeros(90, dtype=np.float32)
    features_oi[4] = 0.015  # ret_15m = +1.5%
    features_oi[83] = 0.020 # oi_delta = +2.0%
    m_oi = MarketState(
        symbol="XRPUSDT",
        timestamp=time.time(),
        last_price=0.5500,
        mid_price=0.5500,
        bid=0.5499,
        ask=0.5501,
        spread=0.0002,
        volume=100000.0,
        volatility=0.015,
        funding_rate=0.0001,
        order_book_imbalance=0.0,
        vwap=0.5450,
        features=features_oi
    )
    ok_oi_short, _, reason_short = engine.evaluate_oi_divergence("XRPUSDT", m_oi, "SHORT")
    assert not ok_oi_short, "Pillar 2 should have blocked Short into bullish OI expansion!"
    print(f"  [PASS] Pillar 2 Vetoed Short: {reason_short}")
    
    ok_oi_long, mult_long, reason_long = engine.evaluate_oi_divergence("XRPUSDT", m_oi, "LONG")
    assert ok_oi_long and mult_long > 1.0, "Pillar 2 should have approved and boosted Long!"
    print(f"  [PASS] Pillar 2 Boosted Long ({mult_long}x): {reason_long}")

    # 3. Test Pillar 3: Lead-Lag Cross-Asset Velocity (BTC Driver)
    print("Testing Pillar 3: BTC Lead-Lag...")
    btc_feat = np.zeros(90, dtype=np.float32)
    btc_feat[3] = 0.0060 # BTC up +0.60% in 5m
    m_btc = MarketState(
        symbol="BTCUSDT",
        timestamp=time.time(),
        last_price=68000.0,
        mid_price=68000.0,
        bid=67999.0,
        ask=68001.0,
        spread=2.0,
        volume=5000000.0,
        volatility=0.010,
        funding_rate=0.0001,
        order_book_imbalance=0.30,
        vwap=67800.0,
        features=btc_feat
    )
    all_markets = {"BTCUSDT": m_btc, "XRPUSDT": m_xrp}
    ok_lead, reason_lead = engine.evaluate_btc_lead_lag("XRPUSDT", all_markets, "SHORT")
    assert not ok_lead, "Pillar 3 should have vetoed XRP Short while BTC is ripping!"
    print(f"  [PASS] Pillar 3 Vetoed Altcoin Short during BTC rally: {reason_lead}")

    # 4. Test Pillar 4: Perpetual-Spot Basis
    print("Testing Pillar 4: Perpetual-Spot Basis...")
    features_basis = np.zeros(90, dtype=np.float32)
    features_basis[85] = 0.0020 # Mark price premium = +0.20% (paper pump)
    m_basis = MarketState(
        symbol="NEARUSDT",
        timestamp=time.time(),
        last_price=5.00,
        mid_price=5.00,
        bid=4.99,
        ask=5.01,
        spread=0.02,
        volume=10000.0,
        volatility=0.020,
        funding_rate=0.0001,
        order_book_imbalance=0.0,
        vwap=4.95,
        features=features_basis
    )
    ok_basis, reason_basis = engine.evaluate_perp_spot_basis("NEARUSDT", m_basis, "LONG")
    assert not ok_basis, "Pillar 4 should have vetoed Long into overheated paper pump!"
    print(f"  [PASS] Pillar 4 Vetoed Long: {reason_basis}")

    # 5. Test Pillar 5: Order Flow Toxicity (VPIN)
    print("Testing Pillar 5: Order Flow Toxicity (VPIN)...")
    # Feed toxic buckets (all buyers, 1.0 imbalance)
    for _ in range(20):
        engine.vpin_history["XRPUSDT"].append(0.85) # High toxic flow
    m_vpin = MarketState(
        symbol="XRPUSDT",
        timestamp=time.time(),
        last_price=0.5500,
        mid_price=0.5500,
        bid=0.5499,
        ask=0.5501,
        spread=0.0002,
        volume=100000.0,
        volatility=0.015,
        funding_rate=0.0001,
        order_book_imbalance=0.20,
        vwap=0.5450,
        features=features
    )
    ok_vpin, reason_vpin = engine.evaluate_vpin_toxicity("XRPUSDT", m_vpin, "SHORT")
    assert not ok_vpin, "Pillar 5 should have vetoed Short against toxic institutional buyer flow!"
    print(f"  [PASS] Pillar 5 Vetoed Short: {reason_vpin}")

    # 6. Test Pillar 6: Real-Time Market Regime Classification
    print("Testing Pillar 6: Market Regime Classifier...")
    # Shock test
    features_shock = np.zeros(90, dtype=np.float32)
    features_shock[81] = 200_000.0 # $200k liquidations in 1m
    m_shock = MarketState(
        symbol="XRPUSDT",
        timestamp=time.time(),
        last_price=0.5500,
        mid_price=0.5500,
        bid=0.5499,
        ask=0.5501,
        spread=0.0002,
        volume=100000.0,
        volatility=0.015,
        funding_rate=0.0001,
        order_book_imbalance=0.0,
        vwap=0.5450,
        features=features_shock
    )
    ok_shock, reg_shock, reason_shock = engine.classify_and_filter_regime("XRPUSDT", m_shock, "NEUTRAL", "LONG")
    assert not ok_shock and reg_shock == "REGIME_SHOCK", "Pillar 6 should have halted trading in shock regime!"
    print(f"  [PASS] Pillar 6 Shock Quarantine Triggered: {reason_shock}")

    # Trend test
    features_trend = np.zeros(90, dtype=np.float32)
    features_trend[5] = 0.025 # +2.5% return in 30m
    m_trend = MarketState(
        symbol="NEARUSDT",
        timestamp=time.time(),
        last_price=5.00,
        mid_price=5.00,
        bid=4.99,
        ask=5.01,
        spread=0.02,
        volume=10000.0,
        volatility=0.020,
        funding_rate=0.0001,
        order_book_imbalance=0.0,
        vwap=4.95,
        features=features_trend
    )
    ok_trend, reg_trend, reason_trend = engine.classify_and_filter_regime("NEARUSDT", m_trend, "STRONG_BULL", "SHORT")
    assert not ok_trend and reg_trend == "REGIME_TREND", "Pillar 6 should have vetoed counter-trend short!"
    print(f"  [PASS] Pillar 6 Trend Filter Vetoed Counter-Trend Short: {reason_trend}")

    # 7. Test Pillar 7: Market Breadth (Advance/Decline & VWAP Ratio)
    print("Testing Pillar 7: Market Breadth...")
    all_green = {}
    for i, s in enumerate(symbols):
        f = np.zeros(90, dtype=np.float32)
        f[4] = 0.010 # All green (+1.0% in 15m)
        all_green[s] = MarketState(
            symbol=s, timestamp=time.time(), last_price=10.0, mid_price=10.0,
            bid=9.99, ask=10.01, spread=0.02, volume=10000.0, volatility=0.015,
            funding_rate=0.0001, order_book_imbalance=0.1, vwap=9.8, features=f
        )
    ok_b_short, _, reason_b_short = engine.evaluate_market_breadth(all_green, "SHORT")
    assert not ok_b_short, "Pillar 7 should have vetoed Short into broad bullish expansion!"
    print(f"  [PASS] Pillar 7 Vetoed Short: {reason_b_short}")
    ok_b_long, b_mult, reason_b_long = engine.evaluate_market_breadth(all_green, "LONG")
    assert ok_b_long and b_mult > 1.0, "Pillar 7 should have boosted Long in broad bull market!"
    print(f"  [PASS] Pillar 7 Boosted Long ({b_mult}x): {reason_b_long}")

    # 8. Test Pillar 8: Market-Wide CVD
    print("Testing Pillar 8: Market-Wide CVD...")
    whale_states_buy = {
        s: {"whale_net_flow_15m": 1_000_000.0, "timestamp": time.time()} for s in symbols
    } # Total = +$4,000,000 net buying
    ok_cvd, _, reason_cvd = engine.evaluate_market_wide_cvd("XRPUSDT", m_xrp, whale_states_buy, "SHORT")
    assert not ok_cvd, "Pillar 8 should have vetoed Short during market-wide institutional buying!"
    print(f"  [PASS] Pillar 8 Vetoed Short: {reason_cvd}")

    # 9. Test Pillar 9: Funding Rate Heatmap & Crowded Trade Index
    print("Testing Pillar 9: Funding Rate Heatmap...")
    all_hot_funding = {}
    for s in symbols:
        f = np.zeros(90, dtype=np.float32)
        all_hot_funding[s] = MarketState(
            symbol=s, timestamp=time.time(), last_price=10.0, mid_price=10.0,
            bid=9.99, ask=10.01, spread=0.02, volume=10000.0, volatility=0.015,
            funding_rate=0.00045, # Overheated (+0.045% per 8h)
            order_book_imbalance=0.0, vwap=10.0, features=f
        )
    ok_fund, _, reason_fund = engine.evaluate_funding_heatmap("XRPUSDT", all_hot_funding, "LONG")
    assert not ok_fund, "Pillar 9 should have vetoed Long into overheated funding!"
    print(f"  [PASS] Pillar 9 Vetoed Overcrowded Long: {reason_fund}")

    # 10. Test Pillar 10: Bitcoin Dominance (BTC.D) & Altcoin Season Index
    print("Testing Pillar 10: Bitcoin Dominance Regime...")
    btc_f_surge = np.zeros(90, dtype=np.float32)
    btc_f_surge[4] = 0.030 # BTC +3.0% in 1h
    m_btc_surge = MarketState(
        symbol="BTCUSDT", timestamp=time.time(), last_price=68000.0, mid_price=68000.0,
        bid=67999.0, ask=68001.0, spread=2.0, volume=5000000.0, volatility=0.010,
        funding_rate=0.0001, order_book_imbalance=0.0, vwap=67000.0, features=btc_f_surge
    )
    all_dom = {"BTCUSDT": m_btc_surge, "XRPUSDT": m_xrp}
    ok_dom, dom_mult, reason_dom = engine.evaluate_btc_dominance_regime("XRPUSDT", all_dom, "LONG")
    assert ok_dom and dom_mult < 1.0, "Pillar 10 should have dampened altcoin sizing during BTC dominance surge!"
    print(f"  [PASS] Pillar 10 Dampened Altcoin Sizing ({dom_mult}x): {reason_dom}")

    # 11. Test Pillar 11: Multi-Exchange Liquidation Ratio (Squeeze vs Flush)
    print("Testing Pillar 11: Liquidation Cascade...")
    all_liq_squeeze = {}
    for s in symbols:
        f = np.zeros(90, dtype=np.float32)
        f[4] = 0.020 # Rising price
        f[82] = 150_000.0 # $150k liq each (Total = $600k)
        all_liq_squeeze[s] = MarketState(
            symbol=s, timestamp=time.time(), last_price=10.0, mid_price=10.0,
            bid=9.99, ask=10.01, spread=0.02, volume=10000.0, volatility=0.015,
            funding_rate=0.0001, order_book_imbalance=0.0, vwap=10.0, features=f
        )
    ok_liq, reason_liq = engine.evaluate_liquidation_cascade("XRPUSDT", all_liq_squeeze, "SHORT")
    assert not ok_liq, "Pillar 11 should have vetoed Short into active short squeeze cascade!"
    print(f"  [PASS] Pillar 11 Vetoed Short into Cascade: {reason_liq}")

    # 12. Test Pillar 12: Volatility Squeeze Indicator (John Carter Squeeze)
    print("Testing Pillar 12: Volatility Squeeze...")
    # Squeeze breakout test
    m_sq_breakout = MarketState(
        symbol="NEARUSDT", timestamp=time.time(), last_price=10.60, mid_price=10.60,
        bid=10.59, ask=10.61, spread=0.02, volume=10000.0, volatility=0.020,
        funding_rate=0.0001, order_book_imbalance=0.0, vwap=10.30, features=features_trend
    )
    for p in np.linspace(10.0, 10.50, 50): # Rising prices out of consolidation
        engine.price_history["NEARUSDT"].append(p)
    ok_sq, sq_mult, reason_sq = engine.evaluate_volatility_squeeze("NEARUSDT", m_sq_breakout, "LONG")
    assert ok_sq and sq_mult > 1.0, "Pillar 12 should have boosted sizing on bullish squeeze firing!"
    print(f"  [PASS] Pillar 12 Fired Bullish Squeeze ({sq_mult}x): {reason_sq}")

    print("\n=======================================================")
    print("ALL 12 INSTITUTIONAL ALPHA PILLARS & MARKET HEALTH INDICATORS")
    print("PASSED WITH 100% RIGOROUS VALIDATION!")
    print("=======================================================")

if __name__ == "__main__":
    test_all_6_pillars()
