import time
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from core.config.settings import settings
from core.logging.logger import logger
from core.schemas.state_schema import PortfolioState, MarketState

# Institutional Baseline Cross-Asset Betas vs BTC (Calibrated from 180-day derivatives data)
DEFAULT_SYMBOL_BETAS = {
    "BTCUSDT": 1.00,
    "ETHUSDT": 1.12,
    "SOLUSDT": 1.38,
    "BNBUSDT": 0.82,
    "XRPUSDT": 1.15,
    "LINKUSDT": 1.25,
    "ADAUSDT": 1.35,
    "AVAXUSDT": 1.48,
    "NEARUSDT": 1.65,
    "LTCUSDT": 0.95,
    "SUIUSDT": 1.85,
    "DOGEUSDT": 1.72,
    "RENDERUSDT": 1.52,
    "1000PEPEUSDT": 1.95,
    "APTUSDT": 1.42,
    "TAOUSDT": 1.55,
    "WIFUSDT": 1.90,
    "INJUSDT": 1.48,
}
DEFAULT_ALT_BETA = 1.40

class MarketBetaHedger:
    """
    Institutional Market-Beta Hedger (Delta-Neutral Portfolio Engine).
    
    Eliminates systemic crypto market drawdown by calculating the portfolio's
    aggregate Beta exposure across all active Altcoins, and maintaining an exact
    offsetting micro-hedge on BTC Futures.
    
    Allows the trading bot to profit purely from altcoin idiosyncratic alpha
    without suffering from broad market crashes.
    """

    def __init__(self):
        self.betas: Dict[str, float] = dict(DEFAULT_SYMBOL_BETAS)
        self.last_rebalance_time = 0.0
        self.rebalance_cooldown_seconds = 15.0  # At most one rebalance every 15s to prevent churn
        self.historical_returns: Dict[str, List[float]] = {}
        
    def get_symbol_beta(self, symbol: str) -> float:
        """Returns the market beta of the given symbol relative to BTC."""
        return self.betas.get(symbol, DEFAULT_ALT_BETA)

    def update_historical_returns(self, symbol: str, returns: List[float]):
        """Ingests historical returns to dynamically calibrate online beta."""
        if len(returns) >= 20:
            self.historical_returns[symbol] = returns

    def calibrate_betas_from_history(self):
        """
        Dynamically recalibrates Betas against BTC using rolling covariance:
        Beta_i = Cov(R_i, R_BTC) / Var(R_BTC)
        """
        btc_rets = self.historical_returns.get("BTCUSDT")
        if not btc_rets or len(btc_rets) < 20:
            return
            
        btc_arr = np.array(btc_rets)
        btc_var = float(np.var(btc_arr))
        if btc_var <= 1e-8:
            return
            
        for sym, rets in self.historical_returns.items():
            if sym == "BTCUSDT":
                self.betas[sym] = 1.00
                continue
                
            if len(rets) == len(btc_rets):
                sym_arr = np.array(rets)
                cov = float(np.cov(sym_arr, btc_arr)[0, 1])
                calc_beta = cov / btc_var
                # Clamp within reasonable sanity bounds [0.4, 3.0]
                clamped_beta = float(np.clip(calc_beta, 0.40, 3.00))
                self.betas[sym] = clamped_beta
                
        logger.debug(f"Recalibrated Beta Hedger universe Betas: {self.betas}")

    def calculate_portfolio_delta(
        self,
        portfolio: PortfolioState,
        market_states: Dict[str, MarketState]
    ) -> Dict[str, Any]:
        """
        Calculates the net beta-weighted market delta of the entire portfolio.
        Returns:
            {
                "net_altcoin_delta_usd": float,
                "current_btc_notional_usd": float,
                "target_btc_hedge_usd": float,
                "hedge_discrepancy_usd": float,
                "positions_breakdown": dict
            }
        """
        net_alt_delta_usd = 0.0
        current_btc_notional_usd = 0.0
        positions_breakdown = {}
        
        btc_price = 0.0
        if "BTCUSDT" in market_states and market_states["BTCUSDT"]:
            btc_price = market_states["BTCUSDT"].mid_price

        for sym, pos in portfolio.positions.items():
            qty = float(pos.quantity)
            if qty == 0:
                continue
                
            # Determine current price
            m = market_states.get(sym)
            px = m.mid_price if m and m.mid_price > 0 else (pos.current_price or pos.entry_price)
            if px <= 0:
                continue
                
            notional = qty * px  # Signed: positive for Long, negative for Short
            beta = self.get_symbol_beta(sym)
            beta_delta = notional * beta
            
            positions_breakdown[sym] = {
                "quantity": qty,
                "price": px,
                "notional_usd": notional,
                "beta": beta,
                "beta_delta_usd": beta_delta
            }
            
            if sym == "BTCUSDT":
                current_btc_notional_usd += notional
                if btc_price <= 0:
                    btc_price = px
            else:
                net_alt_delta_usd += beta_delta

        hedge_ratio = settings.beta_hedge_ratio # default 1.0 (100% neutral)
        target_btc_hedge_usd = -1.0 * net_alt_delta_usd * hedge_ratio
        hedge_discrepancy_usd = target_btc_hedge_usd - current_btc_notional_usd
        
        return {
            "btc_price": btc_price,
            "net_altcoin_delta_usd": net_alt_delta_usd,
            "current_btc_notional_usd": current_btc_notional_usd,
            "target_btc_hedge_usd": target_btc_hedge_usd,
            "hedge_discrepancy_usd": hedge_discrepancy_usd,
            "positions_breakdown": positions_breakdown
        }

    def evaluate_rebalance(
        self,
        portfolio: PortfolioState,
        market_states: Dict[str, MarketState]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates whether a BTC hedge adjustment order is necessary.
        Returns rebalance order specification if required, or None.
        """
        if not settings.beta_hedging_enabled:
            return None
            
        now = time.time()
        if (now - self.last_rebalance_time) < self.rebalance_cooldown_seconds:
            return None
            
        delta_info = self.calculate_portfolio_delta(portfolio, market_states)
        btc_price = delta_info["btc_price"]
        discrepancy_usd = delta_info["hedge_discrepancy_usd"]
        threshold_usd = settings.beta_hedge_rebalance_threshold_usd # default $250
        
        if btc_price <= 0:
            return None
            
        # Check if discrepancy exceeds hurdle
        if abs(discrepancy_usd) >= threshold_usd:
            # Rebalance required!
            needed_btc_qty = abs(discrepancy_usd) / btc_price
            
            # If discrepancy_usd > 0: target hedge is more positive (less short/more long) -> BUY BTC
            # If discrepancy_usd < 0: target hedge is more negative (more short/less long) -> SELL BTC
            side = "BUY" if discrepancy_usd > 0 else "SELL"
            action_type = "CLOSE_SHORT" if (side == "BUY" and delta_info["current_btc_notional_usd"] < 0) else (
                "OPEN_SHORT" if side == "SELL" else "OPEN_LONG"
            )
            
            self.last_rebalance_time = now
            logger.info(
                f"🛡️ BETA HEDGER REBALANCE: Altcoin Delta=${delta_info['net_altcoin_delta_usd']:+,.2f} | "
                f"BTC Notional=${delta_info['current_btc_notional_usd']:+,.2f} | "
                f"Target BTC=${delta_info['target_btc_hedge_usd']:+,.2f} | "
                f"Discrepancy=${discrepancy_usd:+,.2f} >= ${threshold_usd:.0f}. "
                f"Adjusting BTC hedge: {side} {needed_btc_qty:.4f} BTC (${abs(discrepancy_usd):,.2f})"
            )
            
            return {
                "symbol": "BTCUSDT",
                "side": side,
                "action_type": action_type,
                "quantity": needed_btc_qty,
                "notional_usd": abs(discrepancy_usd),
                "btc_price": btc_price,
                "reason": "BETA_DELTA_NEUTRAL_HEDGE",
                "delta_info": delta_info
            }
            
        return None

# Global Singleton Instance
beta_hedger = MarketBetaHedger()
