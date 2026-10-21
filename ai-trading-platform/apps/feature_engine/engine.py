import numpy as np
import collections
from typing import Dict, List, Optional
import time


class FeatureEngine:
    """
    Computes real-time rolling features (Price, Volume, Orderbook, VWAP, Derivatives, Volatility).

    Two time bases:
      - ticks (~1 per second, last 30 minutes): fast returns, flow, order book, candle geometry
      - closed 1-minute bars (last ~16 hours, can be pre-loaded from exchange klines with seed_bars):
        multi-timeframe RSI, EMA/SMA 50/200, MACD, Bollinger, stochastic, ADX, ATR and the 1h ATR
        volatility used by the exit policy. These are recomputed once per closed bar, not every tick.

    Output: 90 floats in the fixed order documented in core/schemas/state_layout.py.
    """
    BAR_HISTORY = 1000  # 1m bars (~16.7h): enough for RSI(14) on 1h bars and EMA/SMA 200 on 1m

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

        # Derivatives & Macro
        self.funding_rates = collections.deque(maxlen=self.max_len)
        self.liquidations = collections.deque(maxlen=self.max_len)
        self.open_interests = collections.deque(maxlen=self.max_len)
        self.ls_ratios = collections.deque(maxlen=self.max_len)
        self.ema_4hs = collections.deque(maxlen=self.max_len)
        self.ob_skews_l2 = collections.deque(maxlen=self.max_len)
        self.price_changes_24h = collections.deque(maxlen=self.max_len)
        self.volumes_24h = collections.deque(maxlen=self.max_len)
        self.mark_price_premiums = collections.deque(maxlen=self.max_len)

        # Closed 1-minute bars: (minute_index, open, high, low, close, volume)
        self.bars = collections.deque(maxlen=self.BAR_HISTORY)
        self._cur_bar: Optional[List[float]] = None
        self._bar_cache: Dict[str, float] = {}

        self.last_update_time = time.time()

    def add_tick(self, data: Dict, ts: Optional[float] = None):
        """
        Ingests one snapshot (~1 per second). `volume` / `buy_volume` / `trade_count` / `liquidation_volume`
        must be the totals accumulated SINCE THE PREVIOUS TICK (the collector resets them after each tick).
        """
        now = time.time() if ts is None else ts
        self.last_update_time = now

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
        self.open_interests.append(data.get("open_interest", 0.0))
        self.ls_ratios.append(data.get("ls_ratio", 1.0))
        self.ema_4hs.append(data.get("ema_4h", 0.0))
        self.ob_skews_l2.append(data.get("ob_skew_l2", 0.0))
        self.price_changes_24h.append(data.get("price_change_24h", 0.0))
        self.volumes_24h.append(data.get("volume_24h", 0.0))
        self.mark_price_premiums.append(data.get("mark_price_premium", 0.0))

        if mid_price:
            self._update_bar(int(now // 60), float(mid_price), float(vol))

    # ------------------------------------------------------------------
    # 1-minute bars
    # ------------------------------------------------------------------
    def _update_bar(self, minute: int, price: float, vol: float):
        bar = self._cur_bar
        if bar is None or minute != bar[0]:
            if bar is not None:
                self.bars.append(tuple(bar))
                self._recompute_bar_indicators()
            self._cur_bar = [minute, price, price, price, price, vol]
            return
        bar[2] = max(bar[2], price)
        bar[3] = min(bar[3], price)
        bar[4] = price
        bar[5] += vol

    def seed_bars(self, klines: List[list]):
        """Pre-loads closed 1m bars from exchange klines ([openTime ms, open, high, low, close, volume, ...])."""
        if self.bars:
            return
        current_minute = int(time.time() // 60)
        for k in klines:
            minute = int(k[0]) // 60000
            if minute >= current_minute:
                continue  # skip the still-open candle
            self.bars.append((minute, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
        self._recompute_bar_indicators()

    @staticmethod
    def _resample(bars: np.ndarray, n: int) -> np.ndarray:
        """Groups 1m bars (rows: o,h,l,c) into n-minute bars ending at the most recent bar."""
        usable = (len(bars) // n) * n
        if usable == 0:
            return np.empty((0, 4))
        b = bars[len(bars) - usable:].reshape(-1, n, 4)
        return np.stack([b[:, 0, 0], b[:, :, 1].max(axis=1), b[:, :, 2].min(axis=1), b[:, -1, 3]], axis=1)

    def _recompute_bar_indicators(self):
        c = {}
        if len(self.bars) < 2:
            self._bar_cache = c
            return
        ohlc = np.array([[b[1], b[2], b[3], b[4]] for b in self.bars], dtype=np.float64)
        highs, lows, closes = ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]

        for name, n in (("rsi_1m", 1), ("rsi_5m", 5), ("rsi_15m", 15), ("rsi_1h", 60)):
            series = closes if n == 1 else self._resample(ohlc, n)[:, 3]
            if len(series) >= 15:
                c[name] = (self._rsi(series, 14) - 50.0) / 50.0

        for p in (9, 21, 50, 200):
            if len(closes) >= p:
                c[f"ema_{p}"] = float(self._ema(closes[-max(3 * p, 60):], p)[-1])
        for p in (20, 50, 200):
            if len(closes) >= p:
                c[f"sma_{p}"] = float(np.mean(closes[-p:]))
        if len(closes) >= 20:
            c["std_20"] = float(np.std(closes[-20:]))
        if len(closes) >= 35:
            ema_12 = self._ema(closes[-200:], 12)
            ema_26 = self._ema(closes[-200:], 26)
            macd_series = ema_12 - ema_26
            c["macd_line"] = float(macd_series[-1])
            c["macd_sig"] = float(self._ema(macd_series, 9)[-1])
        if len(closes) >= 16:
            ks = []
            for end in (len(closes) - 2, len(closes) - 1, len(closes)):
                hh, ll = highs[end - 14:end].max(), lows[end - 14:end].min()
                ks.append((closes[end - 1] - ll) / (hh - ll) if hh > ll else 0.5)
            c["stoch_d"] = float(np.mean(ks))
            c["high_14"], c["low_14"] = float(highs[-14:].max()), float(lows[-14:].min())
            c["atr_1m"] = self._atr(highs, lows, closes, 14)
        if len(closes) >= 30:
            c["adx"] = self._adx(highs, lows, closes, 14)
        h1 = self._resample(ohlc, 60)
        if len(h1) >= 3:
            c["volatility_1h"] = self._atr(h1[:, 1], h1[:, 2], h1[:, 3], 14) / max(h1[-1, 3], 1e-12)
        self._bar_cache = c

    def volatility_1h(self) -> float:
        """Average true range of 1h bars as a fraction of price (0.0 until enough history)."""
        return float(self._bar_cache.get("volatility_1h", 0.0))

    # ------------------------------------------------------------------
    # Indicator math
    # ------------------------------------------------------------------
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        if len(data) == 0: return np.array([])
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data, dtype=np.float64)
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
        for delta in deltas[period:]:
            up = (up * (period - 1) + max(delta, 0.0)) / period
            down = (down * (period - 1) + max(-delta, 0.0)) / period
        if down == 0 and up == 0:
            return 50.0
        if down == 0:
            return 100.0
        rs = up / down
        return float(np.clip(100.0 - (100.0 / (1.0 + rs)), 0.0, 100.0))

    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int=14) -> float:
        if len(closes) < 2 or len(highs) < 2 or len(lows) < 2: return 0.0
        # True Range: max(H-L, |H-C_prev|, |L-C_prev|)
        h_l = highs[1:] - lows[1:]
        h_cp = np.abs(highs[1:] - closes[:-1])
        l_cp = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(h_l, np.maximum(h_cp, l_cp))
        if len(tr) == 0: return 0.0
        if len(tr) < period: return float(np.mean(tr))
        return float(np.mean(tr[-period:]))

    def _adx(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Wilder ADX on the last ~4*period bars, scaled to [0, 1]."""
        h, l, c = highs[-4 * period:], lows[-4 * period:], closes[-4 * period:]
        up = h[1:] - h[:-1]
        down = l[:-1] - l[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        if len(tr) < period * 2:
            return 0.0

        def wilder(x):
            out = np.zeros_like(x)
            out[period - 1] = x[:period].sum()
            for i in range(period, len(x)):
                out[i] = out[i - 1] - out[i - 1] / period + x[i]
            return out[period - 1:]

        atr_s = wilder(tr)
        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = 100 * wilder(plus_dm) / atr_s
            mdi = 100 * wilder(minus_dm) / atr_s
            dx = np.nan_to_num(100 * np.abs(pdi - mdi) / (pdi + mdi))
        if len(dx) < period:
            return 0.0
        return float(np.mean(dx[-period:]) / 100.0)

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
        ret_24h = self.price_changes_24h[-1] if len(self.price_changes_24h) > 0 else 0.0
        high_30 = np.max(prices)
        low_30 = np.min(prices)
        price_range = (high_30 - low_30) / current_price if current_price > 0 else 0.0
        momentum = ret_1m

        # 2. Volume & VWAP [10 slots]
        vol_1m = np.sum(volumes[-60:]) if len(volumes) >= 60 else np.sum(volumes)
        buy_vol_1m = np.sum(buy_vols[-60:]) if len(buy_vols) >= 60 else np.sum(buy_vols)
        sell_vol_1m = np.sum(sell_vols[-60:]) if len(sell_vols) >= 60 else np.sum(sell_vols)
        trade_imbalance = (buy_vol_1m - sell_vol_1m) / (vol_1m + 1e-8)

        n = min(len(prices), len(volumes))
        p_al, v_al = prices[-n:], volumes[-n:]
        vwap_1m = np.sum(p_al[-60:] * v_al[-60:]) / (np.sum(v_al[-60:]) + 1e-8) if np.sum(v_al[-60:]) > 0 else current_price
        vwap_15m = np.sum(p_al[-900:] * v_al[-900:]) / (np.sum(v_al[-900:]) + 1e-8) if np.sum(v_al[-900:]) > 0 else current_price
        vwap_dev_1m = (current_price - vwap_1m) / vwap_1m if vwap_1m > 0 else 0.0
        vwap_dev_15m = (current_price - vwap_15m) / vwap_15m if vwap_15m > 0 else 0.0
        realized_vol = np.std(np.diff(np.log(prices[-60:]))) if len(prices) >= 60 else 0.0
        vol_24h = self.volumes_24h[-1] if len(self.volumes_24h) > 0 else 0.0

        # 3. Deep Orderbook Dynamics [15 slots]
        spread_bps = ((asks[-1] - bids[-1]) / current_price) * 10000 if current_price > 0 else 0.0
        bid_qty = self.bid_vols[-1]
        ask_qty = self.ask_vols[-1]
        ob_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-8)
        ob_skew_l2 = self.ob_skews_l2[-1] if len(self.ob_skews_l2) > 0 else 0.0
        # 11 padded deep book slots (reserved for L3 data integration)

        # Hurst Exponent (StatArb Math)
        hurst = 0.5
        try:
            if len(prices) >= 20:
                lags = range(2, 10)
                tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag])) + 1e-8) for lag in lags]
                m = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = m[0]*2.0 if not np.isnan(m[0]) else 0.5
        except Exception:
            pass

        acf, skew, kurt = 0.0, 0.0, 0.0
        if len(prices) >= 30:
            p_slice = prices[-30:]
            returns = np.diff(p_slice) / p_slice[:-1]
            if np.std(returns[:-1]) > 0 and np.std(returns[1:]) > 0:
                with np.errstate(divide='ignore', invalid='ignore'):
                    acf_matrix = np.corrcoef(returns[:-1], returns[1:])
                    acf = acf_matrix[0, 1] if len(returns) > 2 and not np.isnan(acf_matrix[0, 1]) else 0.0
                try:
                    from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt
                    skew = float(scipy_skew(returns))
                    kurt = float(scipy_kurt(returns))
                except ImportError:
                    mean_r = np.mean(returns)
                    std_r = np.std(returns)
                    if std_r > 1e-8:
                        skew = float(np.mean(((returns - mean_r) / std_r) ** 3))
                        kurt = float(np.mean(((returns - mean_r) / std_r) ** 4) - 3.0)
                    else:
                        skew, kurt = 0.0, 0.0

        # 4. Technical Indicators from closed 1-minute bars [35 slots]
        bc = self._bar_cache
        cp = current_price if current_price > 0 else 1.0

        def dist(key):
            v = bc.get(key)
            return (current_price - v) / cp if v else 0.0

        rsi_1m, rsi_5m = bc.get("rsi_1m", 0.0), bc.get("rsi_5m", 0.0)
        rsi_15m, rsi_1h = bc.get("rsi_15m", 0.0), bc.get("rsi_1h", 0.0)
        macd_line = bc.get("macd_line", 0.0) / cp
        macd_sig = bc.get("macd_sig", 0.0) / cp
        macd_hist = macd_line - macd_sig
        ema_9_dist, ema_21_dist = dist("ema_9"), dist("ema_21")
        ema_50_dist, ema_200_dist = dist("ema_50"), dist("ema_200")
        sma_50_dist, sma_200_dist = dist("sma_50"), dist("sma_200")
        bb_width, bb_pos = 0.0, 0.0
        std20 = bc.get("std_20", 0.0)
        if std20 > 0:
            bb_width = (4 * std20) / cp
            bb_pos = float(np.clip((current_price - bc["sma_20"]) / (4 * std20), -1.0, 1.0))
        stoch_k = 0.0
        if "high_14" in bc and bc["high_14"] > bc["low_14"]:
            hh, ll = max(bc["high_14"], current_price), min(bc["low_14"], current_price)
            stoch_k = (current_price - ll) / (hh - ll)
        stoch_d = bc.get("stoch_d", 0.0)
        atr = bc.get("atr_1m", 0.0) / cp
        adx = bc.get("adx", 0.0)

        # 4H EMA Deviation
        ema_4h_val = self.ema_4hs[-1] if len(self.ema_4hs) > 0 else 0.0
        ema_4h_dist = (current_price - ema_4h_val) / current_price if ema_4h_val > 0 else 0.0

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

        # New God Tier Derivatives
        current_oi = self.open_interests[-1] if len(self.open_interests) > 0 else 0.0
        past_oi = list(self.open_interests)[-60] if len(self.open_interests) >= 60 else current_oi
        oi_1m_delta = (current_oi - past_oi) / past_oi if past_oi > 0 else 0.0
        ls_ratio = self.ls_ratios[-1] if len(self.ls_ratios) > 0 else 1.0
        mp_premium = self.mark_price_premiums[-1] if len(self.mark_price_premiums) > 0 else 0.0

        features = [
            # 1. Price [10]
            ret_1s, ret_10s, ret_1m, ret_5m, ret_15m, ret_30m, ret_24h, 0.0, price_range, momentum,
            # 2. Volume [10] — slots 14/15 (raw VWAP prices) are DEPRECATED and kept at 0; the deviations carry the signal
            vol_1m, buy_vol_1m, sell_vol_1m, trade_imbalance, 0.0, 0.0, vwap_dev_1m, vwap_dev_15m, realized_vol, vol_24h,
            # 3. Orderbook & StatArb [15] (Slots 40-54)
            spread_bps, bid_qty, ask_qty, ob_imbalance, ob_skew_l2, 0.0, 0.0, 0.0, 0.0, 0.0, hurst, float(acf), float(skew), float(kurt), 0.0,
            # 4. Tech Indicators [35]
            rsi_1m, rsi_5m, rsi_15m, rsi_1h, macd_line, macd_sig, macd_hist, ema_9_dist, ema_21_dist, ema_50_dist, ema_200_dist, sma_50_dist, sma_200_dist, bb_width, bb_pos, ema_4h_dist, atr, stoch_k, stoch_d, adx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # 5. Candlesticks [10]
            wicks[0], wicks[1], bodies[0], wicks[2], wicks[3], bodies[1], wicks[4], wicks[5], bodies[2], 0.0,
            # 6. Derivatives [10]
            funding_rate, liq_1m, liq_15m, oi_1m_delta, ls_ratio, mp_premium, 0.0, 0.0, 0.0, 0.0
        ]

        # Ensure exactly 90 slots
        final_array = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if len(final_array) != 90:
            raise ValueError(f"Feature array length mismatch. Expected 90, got {len(final_array)}")

        return final_array
