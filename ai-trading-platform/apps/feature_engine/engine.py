import numpy as np
import collections
from typing import Dict, List, Optional
import time

class FeatureEngine:
    """
    Computes real-time rolling features (Price, Volume, Orderbook, VWAP, Derivatives, Volatility).
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Buffers for rolling calculations
        self.max_len = 900 # 15 minutes assuming 1 data point per second
        
        # Price and Volume
        self.prices = collections.deque(maxlen=self.max_len)
        self.volumes = collections.deque(maxlen=self.max_len)
        self.buy_volumes = collections.deque(maxlen=self.max_len)
        self.sell_volumes = collections.deque(maxlen=self.max_len)
        self.trade_counts = collections.deque(maxlen=self.max_len)
        
        # Orderbook
        self.bids = collections.deque(maxlen=self.max_len)
        self.asks = collections.deque(maxlen=self.max_len)
        self.bid_vols = collections.deque(maxlen=self.max_len)
        self.ask_vols = collections.deque(maxlen=self.max_len)
        
        # Derivatives
        self.funding_rates = collections.deque(maxlen=self.max_len)
        self.liquidations = collections.deque(maxlen=self.max_len)
        
        self.last_update_time = time.time()
        
    def add_tick(self, data: Dict):
        """Ingests raw market data from WebSocket"""
        self.last_update_time = time.time()
        
        mid_price = data.get("mid_price", 0.0)
        if mid_price: self.prices.append(mid_price)
            
        vol = data.get("volume", 0.0)
        buy_vol = data.get("buy_volume", 0.0)
        self.volumes.append(vol)
        self.buy_volumes.append(buy_vol)
        self.sell_volumes.append(max(0.0, vol - buy_vol))
        self.trade_counts.append(data.get("trade_count", 0.0))
        
        best_bid = data.get("best_bid", mid_price)
        best_ask = data.get("best_ask", mid_price)
        self.bids.append(best_bid)
        self.asks.append(best_ask)
        
        self.bid_vols.append(data.get("bid_qty", 0.0))
        self.ask_vols.append(data.get("ask_qty", 0.0))
        
        self.funding_rates.append(data.get("funding_rate", 0.0))
        self.liquidations.append(data.get("liquidation_volume", 0.0))

    def compute_features(self) -> np.ndarray:
        """Returns a dense vector of computed features (Size ~ 20-25)"""
        if len(self.prices) < 2:
            return np.zeros(25, dtype=np.float32)
            
        prices = np.array(self.prices)
        volumes = np.array(self.volumes)
        bids = np.array(self.bids)
        asks = np.array(self.asks)
        buy_vols = np.array(self.buy_volumes)
        sell_vols = np.array(self.sell_volumes)
        
        current_price = prices[-1]
        
        # 1. Price Features
        ret_1s = (prices[-1] / prices[-2]) - 1.0
        ret_1m = (prices[-1] / prices[-60]) - 1.0 if len(prices) >= 60 else 0.0
        ret_5m = (prices[-1] / prices[-300]) - 1.0 if len(prices) >= 300 else 0.0
        ret_15m = (prices[-1] / prices[-900]) - 1.0 if len(prices) >= 900 else 0.0
        
        high_60 = np.max(prices[-60:]) if len(prices) >= 60 else np.max(prices)
        low_60 = np.min(prices[-60:]) if len(prices) >= 60 else np.min(prices)
        price_range = (high_60 - low_60) / current_price if current_price > 0 else 0.0
        momentum = ret_1m
        
        # 2. Volume & Trade Features
        vol_60 = np.sum(volumes[-60:]) if len(volumes) >= 60 else np.sum(volumes)
        buy_vol_60 = np.sum(buy_vols[-60:]) if len(buy_vols) >= 60 else np.sum(buy_vols)
        sell_vol_60 = np.sum(sell_vols[-60:]) if len(sell_vols) >= 60 else np.sum(sell_vols)
        trade_imbalance = (buy_vol_60 - sell_vol_60) / (vol_60 + 1e-8)
        
        # 3. Order-Book Features
        spread = asks[-1] - bids[-1]
        spread_bps = (spread / current_price) * 10000 if current_price > 0 else 0.0
        bid_qty = self.bid_vols[-1]
        ask_qty = self.ask_vols[-1]
        ob_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-8)
        
        # 4. VWAP & Volatility Features
        if vol_60 > 0:
            vwap_60 = np.sum(prices[-60:] * volumes[-60:]) / vol_60 if len(prices) >= 60 else np.sum(prices * volumes) / np.sum(volumes)
        else:
            vwap_60 = current_price
            
        vwap_deviation = (current_price - vwap_60) / vwap_60 if vwap_60 > 0 else 0.0
        
        realized_vol = np.std(np.diff(np.log(prices[-60:]))) if len(prices) >= 60 else 0.0
        
        # 5. Derivatives Features
        funding_rate = self.funding_rates[-1] if len(self.funding_rates) > 0 else 0.0
        recent_liq = np.sum(list(self.liquidations)[-60:]) if len(self.liquidations) > 0 else 0.0
        
        features = [
            # Price (6)
            ret_1s, ret_1m, ret_5m, ret_15m, price_range, momentum,
            # Volume (4)
            vol_60, buy_vol_60, sell_vol_60, trade_imbalance,
            # Orderbook (4)
            spread_bps, bid_qty, ask_qty, ob_imbalance,
            # VWAP & Vol (3)
            vwap_60, vwap_deviation, realized_vol,
            # Derivatives (2)
            funding_rate, recent_liq,
            # Add padding up to 25 to match neural network fixed shape
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        
        return np.array(features, dtype=np.float32)[:25]
