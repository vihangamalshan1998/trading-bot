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
        self.max_len = 1800 # 30 minutes assuming 1 data point per second
        
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

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        if len(data) == 0: return np.array([])
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
        return ema

    def _rsi(self, data: np.ndarray, period: int = 14) -> float:
        if len(data) < period + 1: return 50.0
        deltas = np.diff(data)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(data)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(data)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi[-1]

    def compute_features(self) -> np.ndarray:
        """Returns a dense vector of computed features (Size 90 for God Mode)"""
        if len(self.prices) < 2:
            return np.zeros(90, dtype=np.float32)
            
        prices = np.array(self.prices)
        volumes = np.array(self.volumes)
        bids = np.array(self.bids)
        asks = np.array(self.asks)
        buy_vols = np.array(self.buy_volumes)
        sell_vols = np.array(self.sell_volumes)
        
        current_price = prices[-1]
        
        # 1. Price Returns (Multi-timeframe) [10 slots]
        ret_1s = (prices[-1] / prices[-2]) - 1.0
        ret_10s = (prices[-1] / prices[-10]) - 1.0 if len(prices) >= 10 else 0.0
        ret_1m = (prices[-1] / prices[-60]) - 1.0 if len(prices) >= 60 else 0.0
        ret_5m = (prices[-1] / prices[-300]) - 1.0 if len(prices) >= 300 else 0.0
        ret_15m = (prices[-1] / prices[-900]) - 1.0 if len(prices) >= 900 else 0.0
        ret_30m = (prices[-1] / prices[-1800]) - 1.0 if len(prices) >= 1800 else 0.0
        high_30 = np.max(prices)
        low_30 = np.min(prices)
        price_range = (high_30 - low_30) / current_price if current_price > 0 else 0.0
        momentum = ret_1m
        
        # 2. Volume & VWAP [10 slots]
        vol_1m = np.sum(volumes[-60:]) if len(volumes) >= 60 else np.sum(volumes)
        buy_vol_1m = np.sum(buy_vols[-60:]) if len(buy_vols) >= 60 else np.sum(buy_vols)
        sell_vol_1m = np.sum(sell_vols[-60:]) if len(sell_vols) >= 60 else np.sum(sell_vols)
        trade_imbalance = (buy_vol_1m - sell_vol_1m) / (vol_1m + 1e-8)
        
        vwap_1m = np.sum(prices[-60:] * volumes[-60:]) / (vol_1m + 1e-8) if len(prices) >= 60 else current_price
        vol_15m = np.sum(volumes[-900:]) if len(volumes) >= 900 else np.sum(volumes)
        vwap_15m = np.sum(prices[-900:] * volumes[-900:]) / (vol_15m + 1e-8) if len(prices) >= 900 else current_price
        vwap_dev_1m = (current_price - vwap_1m) / vwap_1m if vwap_1m > 0 else 0.0
        vwap_dev_15m = (current_price - vwap_15m) / vwap_15m if vwap_15m > 0 else 0.0
        realized_vol = np.std(np.diff(np.log(prices[-60:]))) if len(prices) >= 60 else 0.0
        
        # 3. Deep Orderbook Dynamics [15 slots]
        spread_bps = ((asks[-1] - bids[-1]) / current_price) * 10000 if current_price > 0 else 0.0
        bid_qty = self.bid_vols[-1]
        ask_qty = self.ask_vols[-1]
        ob_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-8)
        # 11 padded deep book slots (reserved for L3 data integration)
        
        # 4. Technical Indicators (1-min downsample) [35 slots]
        m1_closes = prices[len(prices)%60::60] if len(prices) >= 60 else prices
        
        rsi_1m, rsi_5m, rsi_15m, rsi_1h = 0.0, 0.0, 0.0, 0.0
        macd_line, macd_sig, macd_hist = 0.0, 0.0, 0.0
        ema_9_dist, ema_21_dist, ema_50_dist, ema_200_dist = 0.0, 0.0, 0.0, 0.0
        sma_50_dist, sma_200_dist = 0.0, 0.0
        bb_width, bb_pos = 0.0, 0.0
        
        if len(m1_closes) >= 2:
            rsi_1m = (self._rsi(m1_closes, period=min(14, len(m1_closes)-1)) - 50.0) / 50.0
            ema_9_dist = (current_price - self._ema(m1_closes, 9)[-1]) / current_price
            ema_21_dist = (current_price - self._ema(m1_closes, min(21, len(m1_closes)))[-1]) / current_price
            
            if len(m1_closes) >= 5:
                period = min(20, len(m1_closes))
                sma = np.mean(m1_closes[-period:])
                std = np.std(m1_closes[-period:])
                if std > 0:
                    bb_width = (4 * std) / current_price
                    bb_pos = np.clip((current_price - sma) / (4 * std), -1.0, 1.0)
                    
            if len(m1_closes) >= 26:
                ema_12 = self._ema(m1_closes, 12)
                ema_26 = self._ema(m1_closes, 26)
                macd_series = ema_12 - ema_26
                macd_line = macd_series[-1] / current_price
                macd_sig = self._ema(macd_series, 9)[-1] / current_price
                macd_hist = macd_line - macd_sig

        # 5. Candlestick Geometry [10 slots]
        # Calculate O,H,L,C for the last 3 one-minute candles
        wicks, bodies = [], []
        if len(prices) >= 180:
            for i in range(3):
                c_slice = prices[-(i+1)*60 : -i*60 if i != 0 else None]
                if len(c_slice) == 0: continue
                c_open, c_close = c_slice[0], c_slice[-1]
                c_high, c_low = np.max(c_slice), np.min(c_slice)
                body = c_close - c_open
                upper = c_high - max(c_open, c_close)
                lower = min(c_open, c_close) - c_low
                bodies.append(body / current_price)
                wicks.extend([upper / current_price, lower / current_price])
        
        while len(wicks) < 6: wicks.append(0.0)
        while len(bodies) < 3: bodies.append(0.0)
        
        # 6. Derivatives & Liquidity [10 slots]
        funding_rate = self.funding_rates[-1] if len(self.funding_rates) > 0 else 0.0
        liq_1m = np.sum(list(self.liquidations)[-60:]) if len(self.liquidations) > 0 else 0.0
        liq_15m = np.sum(list(self.liquidations)[-900:]) if len(self.liquidations) > 0 else 0.0
        
        features = [
            # 1. Price [10]
            ret_1s, ret_10s, ret_1m, ret_5m, ret_15m, ret_30m, 0.0, 0.0, price_range, momentum,
            # 2. Volume [10]
            vol_1m, buy_vol_1m, sell_vol_1m, trade_imbalance, vwap_1m, vwap_15m, vwap_dev_1m, vwap_dev_15m, realized_vol, 0.0,
            # 3. Orderbook [15]
            spread_bps, bid_qty, ask_qty, ob_imbalance, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # 4. Tech Indicators [35]
            rsi_1m, rsi_5m, rsi_15m, rsi_1h, macd_line, macd_sig, macd_hist, ema_9_dist, ema_21_dist, ema_50_dist, ema_200_dist, sma_50_dist, sma_200_dist, bb_width, bb_pos, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # 5. Candlesticks [10]
            wicks[0], wicks[1], bodies[0], wicks[2], wicks[3], bodies[1], wicks[4], wicks[5], bodies[2], 0.0,
            # 6. Derivatives [10]
            funding_rate, liq_1m, liq_15m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
        
        # Ensure exactly 90 slots
        final_array = np.array(features, dtype=np.float32)
        if len(final_array) != 90:
            raise ValueError(f"Feature array length mismatch. Expected 90, got {len(final_array)}")
            
        return final_array
