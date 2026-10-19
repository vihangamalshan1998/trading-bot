import collections
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.logging.logger import logger
from core.schemas.state_schema import MarketState

class InstitutionalAlphaEngine:
    """
    Proprietary Institutional Alpha Engine.
    Implements the 6 structural pillars used by top quantitative hedge funds
    (Citadel, Jump Trading, Jane Street, Renaissance Technologies):
    
    1. Liquidation Cluster & Stop-Hunt Mapping
    2. Open Interest (OI) Delta Divergence
    3. Lead-Lag Cross-Asset Velocity (BTC/ETH Macro Bellwethers)
    4. Perpetual-Spot Basis & Mark Price Premium Gate
    5. Order Flow Toxicity (VPIN - Volume-Synchronized Probability of Toxicity)
    6. Real-Time Market Regime Classification (Trend vs Chop vs Shock)
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        
        # Pillar 1: Liquidation & Swing High/Low Tracking (30-min rolling window)
        self.price_history = {sym: collections.deque(maxlen=1800) for sym in symbols}
        
        # Pillar 2: Open Interest History (15-min rolling window: timestamp, oi, price)
        self.oi_history = {sym: collections.deque(maxlen=900) for sym in symbols}
        
        # Pillar 5: VPIN Bucketing
        # Standard volume bucket size: $100k for majors, $40k for altcoins
        self.bucket_sizes = {
            sym: (100_000.0 if sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"] else 40_000.0)
            for sym in symbols
        }
        self.current_bucket = {sym: {"buy": 0.0, "sell": 0.0, "total": 0.0} for sym in symbols}
        self.vpin_history = {sym: collections.deque(maxlen=30) for sym in symbols}
        
        # Pillar 6: Real-Time Market Regime State
        self.current_regime = {sym: "REGIME_CHOP" for sym in symbols}
        self.shock_freeze_until = {sym: 0.0 for sym in symbols}

    def update_tick(self, sym: str, market: MarketState):
        """Ingests live market tick to update rolling microstructural metrics."""
        now = time.time()
        px = market.mid_price
        if px <= 0:
            return
            
        self.price_history[sym].append(px)
        
        # Update Open Interest History
        if len(market.features) == 90:
            current_oi = float(market.features[83]) # oi_1m_delta or relative oi
            self.oi_history[sym].append((now, current_oi, px))
            
            # Update VPIN Volume Bucketing
            vol_usd = float(market.volume) * px if market.volume > 0 else 5_000.0
            imbalance = float(market.order_book_imbalance) if market.order_book_imbalance is not None else 0.0
            buy_share = np.clip((imbalance + 1.0) / 2.0, 0.0, 1.0)
            buy_vol = vol_usd * buy_share
            sell_vol = vol_usd * (1.0 - buy_share)
            
            b = self.current_bucket[sym]
            b["buy"] += buy_vol
            b["sell"] += sell_vol
            b["total"] += vol_usd
            
            target_vol = self.bucket_sizes.get(sym, 40_000.0)
            if b["total"] >= target_vol:
                # Bucket completed: calculate toxic imbalance
                bucket_imbalance = abs(b["buy"] - b["sell"]) / b["total"]
                self.vpin_history[sym].append(bucket_imbalance)
                # Reset bucket
                self.current_bucket[sym] = {"buy": 0.0, "sell": 0.0, "total": 0.0}

    # =========================================================================
    # PILLAR 1: Liquidation Cluster & Stop-Hunt Mapping
    # =========================================================================
    def evaluate_stop_hunt(self, sym: str, market: MarketState, side: str) -> Tuple[bool, str]:
        """
        Detects whether price is hovering in a retail liquidation/stop-hunt zone.
        Whales sweep stops beyond recent swing highs/lows before reversing.
        """
        prices = list(self.price_history.get(sym, []))
        if len(prices) < 120:
            return True, "Insufficient price history"
            
        cur_px = market.mid_price
        h_30m = float(np.max(prices))
        l_30m = float(np.min(prices))
        
        dist_to_high = (h_30m - cur_px) / cur_px
        dist_to_low = (cur_px - l_30m) / cur_px
        
        # 1. Short Squeeze Stop-Hunt Magnet:
        # Price is within 0.35% of the 30m high with upward orderbook skew
        if "SHORT" in side and 0.0 <= dist_to_high <= 0.0035:
            if hasattr(market, 'order_book_imbalance') and market.order_book_imbalance > 0.15:
                return False, f"Price is within {dist_to_high*100:.2f}% of 30m High (${h_30m:.4f}). Retail short stop-hunt magnet active!"
                
        # 2. Long Liquidation Flush Magnet:
        # Price is within 0.35% of the 30m low with downward orderbook skew
        if "LONG" in side and 0.0 <= dist_to_low <= 0.0035:
            if hasattr(market, 'order_book_imbalance') and market.order_book_imbalance < -0.15:
                return False, f"Price is within {dist_to_low*100:.2f}% of 30m Low (${l_30m:.4f}). Retail long liquidation flush magnet active!"
                
        return True, "Stop-hunt clear"

    # =========================================================================
    # PILLAR 2: Open Interest (OI) Delta Divergence
    # =========================================================================
    def evaluate_oi_divergence(self, sym: str, market: MarketState, side: str) -> Tuple[bool, float, str]:
        """
        Evaluates Price vs Open Interest dynamics.
        Returns: (approved, conviction_multiplier, reason)
        """
        if len(market.features) < 90:
            return True, 1.0, "No OI features"
            
        ret_15m = float(market.features[4])
        oi_1m_delta = float(market.features[83])
        
        # 1. Price Rising + OI Surging -> Genuine Institutional Capital Expansion
        if ret_15m >= 0.012 and oi_1m_delta >= 0.010:
            if "SHORT" in side:
                return False, 0.0, f"Bullish OI Expansion (Ret15m: {ret_15m*100:+.2f}%, OI Delta: {oi_1m_delta*100:+.2f}%). New money entering, shorts forbidden!"
            elif "LONG" in side:
                return True, 1.15, "Confirmed Bullish OI Expansion: Boosting Long conviction 1.15x"

        # 2. Price Dumping + OI Surging -> Aggressive Institutional Short Building
        if ret_15m <= -0.012 and oi_1m_delta >= 0.010:
            if "LONG" in side:
                return False, 0.0, f"Bearish OI Expansion (Ret15m: {ret_15m*100:+.2f}%, OI Delta: {oi_1m_delta*100:+.2f}%). Aggressive shorts entering, longs forbidden!"
            elif "SHORT" in side:
                return True, 1.15, "Confirmed Bearish OI Expansion: Boosting Short conviction 1.15x"

        # 3. Price Rising + OI Dumping -> Short Covering Squeeze Exhaustion (Do not buy top)
        if ret_15m >= 0.018 and oi_1m_delta <= -0.015:
            if "LONG" in side:
                return False, 0.0, f"Squeeze Exhaustion (Ret15m: {ret_15m*100:+.2f}%, OI Delta: {oi_1m_delta*100:+.2f}%). Rally running on liquidations, do not chase!"

        # 4. Price Dumping + OI Dumping -> Long Capitulation Exhaustion (Do not sell bottom)
        if ret_15m <= -0.018 and oi_1m_delta <= -0.015:
            if "SHORT" in side:
                return False, 0.0, f"Capitulation Exhaustion (Ret15m: {ret_15m*100:+.2f}%, OI Delta: {oi_1m_delta*100:+.2f}%). Dumping running on margin calls, do not short bottom!"

        return True, 1.00, "OI dynamics neutral"

    # =========================================================================
    # PILLAR 3: Lead-Lag Cross-Asset Velocity (BTC/ETH Drivers)
    # =========================================================================
    def evaluate_btc_lead_lag(self, sym: str, all_markets: Dict[str, MarketState], side: str) -> Tuple[bool, str]:
        """
        Global Bellwether Check:
        Altcoins lag BTC/ETH by 2-15 seconds. If BTC is ripping, altcoin shorts are forbidden.
        If BTC is dumping, altcoin longs are forbidden.
        """
        if sym in ["BTCUSDT", "ETHUSDT"]:
            return True, "Is bellwether symbol"
            
        btc_market = all_markets.get("BTCUSDT")
        if not btc_market or len(btc_market.features) < 90:
            return True, "BTC market state unavailable"
            
        btc_ret_1m = float(btc_market.features[2])
        btc_ret_5m = float(btc_market.features[3])
        
        # BTC Bullish Impulse: BTC up >= +0.40% in 5m or >= +0.25% in 1m
        if btc_ret_5m >= 0.0040 or btc_ret_1m >= 0.0025:
            if "SHORT" in side:
                return False, f"BTC Lead-Lag Shield: BTC is in strong upward impulse (1m: {btc_ret_1m*100:+.2f}%, 5m: {btc_ret_5m*100:+.2f}%). Vetoing Altcoin Short!"
                
        # BTC Bearish Impulse: BTC down <= -0.40% in 5m or <= -0.25% in 1m
        if btc_ret_5m <= -0.0040 or btc_ret_1m <= -0.0025:
            if "LONG" in side:
                return False, f"BTC Lead-Lag Shield: BTC is in sharp downward impulse (1m: {btc_ret_1m*100:+.2f}%, 5m: {btc_ret_5m*100:+.2f}%). Vetoing Altcoin Long!"
                
        return True, "BTC momentum neutral"

    # =========================================================================
    # PILLAR 4: Perpetual-Spot Basis & Mark Price Premium Gate
    # =========================================================================
    def evaluate_perp_spot_basis(self, sym: str, market: MarketState, side: str) -> Tuple[bool, str]:
        """
        Perpetual vs Spot Index Discrepancy Gate:
        Detects 'Paper Pumps' (futures price decoupled from spot cash demand).
        """
        if len(market.features) < 90:
            return True, "No basis features"
            
        mp_premium = float(market.features[85])
        
        # 1. Overheated Paper Pump: Perps trading at extreme premium (> +0.15%) over spot
        if "LONG" in side and mp_premium >= 0.0015:
            return False, f"Overheated Perp Basis: Mark price premium is +{mp_premium*100:.3f}% over spot. Retail paper pump detected, refusing Long!"
            
        # 2. Overheated Paper Dump: Perps trading at extreme discount (< -0.15%) under spot
        if "SHORT" in side and mp_premium <= -0.0015:
            return False, f"Overheated Perp Basis: Mark price discount is {mp_premium*100:.3f}% under spot. Retail panic dump detected, refusing Short!"
            
        return True, "Basis alignment healthy"

    # =========================================================================
    # PILLAR 5: Order Flow Toxicity (VPIN)
    # =========================================================================
    def evaluate_vpin_toxicity(self, sym: str, market: MarketState, side: str) -> Tuple[bool, str]:
        """
        Volume-Synchronized Probability of Toxicity (VPIN):
        When VPIN > 0.65, informed institutional toxic flow dominates the book.
        Never take a counter-trend position into high VPIN flow!
        """
        buckets = list(self.vpin_history.get(sym, []))
        if len(buckets) < 10:
            return True, "Gathering VPIN volume buckets"
            
        vpin = float(np.mean(buckets[-15:]))
        
        # High Toxicity threshold = 0.65
        if vpin >= 0.65:
            ob_imb = float(market.order_book_imbalance) if market.order_book_imbalance is not None else 0.0
            
            # If toxic flow is buyer-driven, strictly block Shorts
            if "SHORT" in side and ob_imb >= 0.10:
                return False, f"High Order Flow Toxicity (VPIN: {vpin:.2f} >= 0.65). Informed institutional buyer aggression active, Shorts vetoed!"
                
            # If toxic flow is seller-driven, strictly block Longs
            if "LONG" in side and ob_imb <= -0.10:
                return False, f"High Order Flow Toxicity (VPIN: {vpin:.2f} >= 0.65). Informed institutional seller aggression active, Longs vetoed!"
                
        return True, "VPIN toxicity normal"

    # =========================================================================
    # PILLAR 6: Real-Time Market Regime Classification
    # =========================================================================
    def classify_and_filter_regime(self, sym: str, market: MarketState, htf_trend: str, side: str) -> Tuple[bool, str, str]:
        """
        Machine Learning / Quantitative Regime Classifier:
        1. REGIME_SHOCK: Liquidation cascades or volatility spikes -> 100% Cash freeze!
        2. REGIME_TREND: Strong directional momentum -> Counter-trend entries forbidden!
        3. REGIME_CHOP: Ranging consolidation -> Standard range execution.
        Returns: (approved, regime_name, reason)
        """
        now = time.time()
        
        # Check active shock freeze
        if now < self.shock_freeze_until.get(sym, 0.0):
            rem = (self.shock_freeze_until[sym] - now) / 60.0
            return False, "REGIME_SHOCK", f"Symbol in Volatility Shock quarantine ({rem:.1f}m remaining). Preserving capital!"
            
        vol = float(market.volatility) if hasattr(market, 'volatility') and market.volatility is not None else 0.0
        liq_1m = float(market.features[81]) if len(market.features) == 90 else 0.0
        ret_30m = float(market.features[5]) if len(market.features) == 90 else 0.0
        
        # 1. SHOCK REGIME: Cascading liquidations (>= $150k in 1m) or extreme volatility (> 0.030)
        is_shock = (liq_1m >= 150_000.0) or (vol >= 0.030)
        if is_shock:
            self.shock_freeze_until[sym] = now + 600.0 # 10-minute freeze
            self.current_regime[sym] = "REGIME_SHOCK"
            return False, "REGIME_SHOCK", f"Volatility Shock triggered (Liq1m: ${liq_1m:,.0f}, Vol: {vol*100:.2f}%). Halting entries for 10 mins!"
            
        # 2. TREND REGIME: Strong HTF trend or >= 1.5% 30-min price expansion
        is_trend = (htf_trend in ["STRONG_BULL", "STRONG_BEAR"]) or (abs(ret_30m) >= 0.015)
        if is_trend:
            self.current_regime[sym] = "REGIME_TREND"
            # Strict Trend-Following: Zero counter-trend trades allowed
            if (htf_trend == "STRONG_BULL" or ret_30m >= 0.015) and "SHORT" in side:
                return False, "REGIME_TREND", f"REGIME_TREND Active (Bullish Expansion). Counter-trend Short strictly vetoed!"
            if (htf_trend == "STRONG_BEAR" or ret_30m <= -0.015) and "LONG" in side:
                return False, "REGIME_TREND", f"REGIME_TREND Active (Bearish Breakdown). Counter-trend Long strictly vetoed!"
            return True, "REGIME_TREND", "Trend-aligned entry approved"
            
        # 3. CHOP REGIME: Range-bound consolidation
        self.current_regime[sym] = "REGIME_CHOP"
        return True, "REGIME_CHOP", "Chop regime entry approved"

    # =========================================================================
    # PILLAR 7 (HEALTH 1): Market Breadth (Advance/Decline & VWAP Ratio)
    # =========================================================================
    def evaluate_market_breadth(self, all_markets: Dict[str, MarketState], side: str) -> Tuple[bool, float, str]:
        """
        Calculates cross-sectional breadth (% of universe in uptrend / above VWAP).
        Returns: (approved, sizing_multiplier, reason)
        """
        valid_markets = [m for m in all_markets.values() if m and len(m.features) == 90]
        if len(valid_markets) < 3:
            return True, 1.0, "Insufficient market breadth data"
            
        green_count = sum(1 for m in valid_markets if float(m.features[4]) > 0.0) # ret_15m > 0
        vwap_count = sum(1 for m in valid_markets if m.mid_price > m.vwap)
        total = len(valid_markets)
        
        breadth_ratio = green_count / total
        vwap_ratio = vwap_count / total
        
        # 1. Broad Bullish Expansion: >= 72% green and >= 65% above VWAP
        if breadth_ratio >= 0.72 and vwap_ratio >= 0.65:
            if "SHORT" in side:
                return False, 0.0, f"Strong Bullish Market Breadth ({breadth_ratio*100:.1f}% green, {vwap_ratio*100:.1f}% > VWAP). Global tailwind lifts all alts, Shorts forbidden!"
            elif "LONG" in side:
                return True, 1.15, f"Confirmed Bullish Breadth ({breadth_ratio*100:.1f}% green). Boosting Long sizing 1.15x"
                
        # 2. Broad Bearish Breakdown: <= 28% green and <= 35% above VWAP
        if breadth_ratio <= 0.28 and vwap_ratio <= 0.35:
            if "LONG" in side:
                return False, 0.0, f"Strong Bearish Market Breadth ({breadth_ratio*100:.1f}% green, {vwap_ratio*100:.1f}% > VWAP). Global selloff sinking all alts, Longs forbidden!"
            elif "SHORT" in side:
                return True, 1.15, f"Confirmed Bearish Breadth ({breadth_ratio*100:.1f}% green). Boosting Short sizing 1.15x"
                
        # 3. Fragmented / Divergent Breadth (40% - 60% green)
        if 0.40 <= breadth_ratio <= 0.60:
            return True, 0.85, f"Fragmented Market Breadth ({breadth_ratio*100:.1f}% green). Sizing dampened to 0.85x"
            
        return True, 1.0, f"Market Breadth neutral ({breadth_ratio*100:.1f}% green)"

    # =========================================================================
    # PILLAR 8 (HEALTH 2): Market-Wide CVD (Total Net Taker Flow Across All Coins)
    # =========================================================================
    def evaluate_market_wide_cvd(self, sym: str, market: MarketState, whale_states: Dict[str, Any], side: str) -> Tuple[bool, float, str]:
        """
        Aggregates institutional taker buy vs sell dollars across all symbols.
        Detects passive distribution (rising price during market-wide selling).
        """
        valid_whales = [w for w in whale_states.values() if w and isinstance(w, dict)]
        if not valid_whales:
            return True, 1.0, "Whale states unavailable"
            
        total_net_flow_15m = sum(float(w.get("whale_net_flow_15m", 0.0)) for w in valid_whales)
        
        # 1. Extreme Market-Wide Institutional Buying: >= +$3,000,000
        if total_net_flow_15m >= 3_000_000.0:
            if "SHORT" in side:
                return False, 0.0, f"Market-Wide Institutional Taker Buying (+${total_net_flow_15m:,.0f} in 15m). Whales sweeping entire market, Shorts forbidden!"
            elif "LONG" in side:
                return True, 1.10, "Market-wide whale accumulation confirmed (+10% sizing)"
                
        # 2. Extreme Market-Wide Institutional Selling: <= -$3,000,000
        if total_net_flow_15m <= -3_000_000.0:
            if "LONG" in side:
                return False, 0.0, f"Market-Wide Institutional Taker Dumping (-${abs(total_net_flow_15m):,.0f} in 15m). Whales dumping entire market, Longs forbidden!"
            elif "SHORT" in side:
                return True, 1.10, "Market-wide whale dumping confirmed (+10% sizing)"
                
        # 3. Passive Distribution Trap: Coin is up > +1.2%, but entire market net flow is dumping <= -$1,500,000
        ret_15m = float(market.features[4]) if len(market.features) == 90 else 0.0
        if "LONG" in side and ret_15m >= 0.012 and total_net_flow_15m <= -1_500_000.0:
            return False, 0.0, f"Passive Distribution Trap: [{sym}] pumped +{ret_15m*100:.2f}% while market-wide flow is dumping (-${abs(total_net_flow_15m):,.0f}). Refusing Long trap!"
            
        return True, 1.0, f"Market-wide CVD balanced (${total_net_flow_15m:,.0f})"

    # =========================================================================
    # PILLAR 9 (HEALTH 3): Funding Rate Heatmap & Crowded Trade Index
    # =========================================================================
    def evaluate_funding_heatmap(self, sym: str, all_markets: Dict[str, MarketState], side: str) -> Tuple[bool, float, str]:
        """
        Cross-sectional funding rate analysis.
        Identifies crowded long flushes and crowded short squeezes.
        """
        rates = [float(m.funding_rate) for m in all_markets.values() if m and m.funding_rate is not None]
        if not rates:
            return True, 1.0, "No funding rate data"
            
        mean_funding = float(np.mean(rates))
        
        # 1. Overheated Market-Wide Longs (Funding >= +0.030% per 8h)
        if mean_funding >= 0.00030:
            if "LONG" in side:
                return False, 0.0, f"Overheated Market Funding (Mean: {mean_funding*100:.4f}%). Retail over-leveraged long, high Long Flush risk. Longs forbidden!"
            elif "SHORT" in side:
                return True, 1.15, "Crowded Long Flush Setup: Boosting Short sizing 1.15x"

        # 2. Overcrowded Market-Wide Shorts (Funding <= -0.015% per 8h)
        if mean_funding <= -0.00015:
            if "SHORT" in side:
                return False, 0.0, f"Negative Market Funding (Mean: {mean_funding*100:.4f}%). Retail crowded short, high Short Squeeze risk. Shorts forbidden!"
            elif "LONG" in side:
                return True, 1.15, "Crowded Short Squeeze Setup: Boosting Long sizing 1.15x"

        return True, 1.0, f"Funding rate healthy (Mean: {mean_funding*100:.4f}%)"

    # =========================================================================
    # PILLAR 10 (HEALTH 4): Bitcoin Dominance (BTC.D) & Altcoin Season Index
    # =========================================================================
    def evaluate_btc_dominance_regime(self, sym: str, all_markets: Dict[str, MarketState], side: str) -> Tuple[bool, float, str]:
        """
        Detects Bitcoin liquidity drain vs Altcoin Season momentum.
        """
        if sym in ["BTCUSDT", "ETHUSDT"]:
            return True, 1.0, "Major bellwether"
            
        btc_m = all_markets.get("BTCUSDT")
        if not btc_m or len(btc_m.features) < 90:
            return True, 1.0, "BTC market unavailable"
            
        btc_ret_1h = float(btc_m.features[4])
        alt_returns_1h = [
            float(m.features[4]) for s, m in all_markets.items()
            if s not in ["BTCUSDT", "ETHUSDT"] and len(m.features) == 90
        ]
        if not alt_returns_1h:
            return True, 1.0, "No altcoin data"
            
        alt_mean_1h = float(np.mean(alt_returns_1h))
        dominance_impulse = btc_ret_1h - alt_mean_1h
        
        # 1. BTC Dominance Surge: BTC outperforming alts by >= +1.5% in 1h
        if dominance_impulse >= 0.015:
            if "LONG" in side:
                return True, 0.70, f"BTC Dominance Surge (BTC +{btc_ret_1h*100:.2f}% vs Alts {alt_mean_1h*100:+.2f}%). BTC draining alt liquidity; Altcoin Long sized down to 0.70x"
                
        # 2. Altcoin Season Impulse: Alts outperforming BTC by >= +1.5% in 1h
        if dominance_impulse <= -0.015:
            if "LONG" in side:
                return True, 1.15, f"Altcoin Season Wave (Alts outperforming BTC by {abs(dominance_impulse)*100:.2f}%). Boosting Altcoin Long sizing 1.15x"
                
        return True, 1.0, "Dominance regime neutral"

    # =========================================================================
    # PILLAR 11 (HEALTH 5): Multi-Exchange Liquidation Ratio (Squeeze vs Flush)
    # =========================================================================
    def evaluate_liquidation_cascade(self, sym: str, all_markets: Dict[str, MarketState], side: str) -> Tuple[bool, str]:
        """
        Monitors aggregate portfolio liquidation volumes to detect active cascades.
        """
        total_liq_15m = sum(
            float(m.features[82]) for m in all_markets.values()
            if len(m.features) == 90
        )
        
        # If market liquidation volume >= $400,000 in 15m
        if total_liq_15m >= 400_000.0:
            # Check market direction
            green_count = sum(1 for m in all_markets.values() if len(m.features) == 90 and float(m.features[4]) > 0)
            total = len(all_markets) or 1
            breadth = green_count / total
            
            # Massive Short Squeeze (Breadth >= 65% and high liquidations)
            if breadth >= 0.65 and "SHORT" in side:
                return False, f"Active Short Liquidation Cascade (${total_liq_15m:,.0f} liquidated, {breadth*100:.1f}% green). Forcing shorts to cover, Shorts forbidden!"
                
            # Massive Long Flush (Breadth <= 35% and high liquidations)
            if breadth <= 0.35 and "LONG" in side:
                return False, f"Active Long Liquidation Cascade (${total_liq_15m:,.0f} liquidated, {breadth*100:.1f}% green). Margin calls flushing longs, Longs forbidden!"
                
        return True, "No active liquidation cascade"

    # =========================================================================
    # PILLAR 12 (HEALTH 6): Volatility Squeeze Indicator (John Carter Squeeze)
    # =========================================================================
    def evaluate_volatility_squeeze(self, sym: str, market: MarketState, side: str) -> Tuple[bool, float, str]:
        """
        John Carter Squeeze: Bollinger Bands inside Keltner Channels.
        Identifies volatility compression (Squeeze ON) and explosive breakout momentum (Squeeze Fired).
        """
        prices = list(self.price_history.get(sym, []))
        if len(prices) < 40:
            return True, 1.0, "Gathering squeeze price history"
            
        p20 = np.array(prices[-20:])
        sma = float(np.mean(p20))
        std = float(np.std(p20))
        
        atr = float(market.features[51]) if len(market.features) == 90 and float(market.features[51]) > 0 else (std * 1.2 or 0.001)
        
        bb_upper = sma + (2.0 * std)
        bb_lower = sma - (2.0 * std)
        kc_upper = sma + (1.5 * atr)
        kc_lower = sma - (1.5 * atr)
        
        # Check if Squeeze is ON (BB inside KC)
        squeeze_on = (bb_upper < kc_upper) and (bb_lower > kc_lower)
        
        # Momentum direction: distance from SMA
        cur_px = market.mid_price
        mom = cur_px - sma
        
        if squeeze_on:
            # Squeeze is tightly compressed (low vol consolidation)
            return True, 0.85, f"Volatility Squeeze Active (Compression). Sizing conservative (0.85x) to avoid chop wicks."
        else:
            # Squeeze has fired! Check if momentum aligns with trade
            if mom > 0 and "LONG" in side:
                return True, 1.15, "VOLATILITY SQUEEZE FIRED BULLISH: Explosive momentum breakout! Sizing boosted 1.15x"
            elif mom < 0 and "SHORT" in side:
                return True, 1.15, "VOLATILITY SQUEEZE FIRED BEARISH: Explosive momentum breakdown! Sizing boosted 1.15x"
                
        return True, 1.0, "Volatility squeeze neutral"
