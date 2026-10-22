"""
Pure builders for the blocks of the 200-slot state vector (layout: core/schemas/state_layout.py).

Shared by the live bot (apps/trading_bot/main.py _build_state_vector) and the history replay
(apps/replay), so replayed training records are built by exactly the same code as live ones.
"""
import time
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from core.schemas.state_layout import (
    MARKET_DEPRECATED_FEATURES, MARKET_LOG_VOLUME_FEATURES, MARKET_SPREAD_FEATURE, CROSS_SECTION_DIM, GLOBAL_KEYS,
)

MARKET_DIM = 90


def market_block(features: Optional[Sequence[float]]) -> List[float]:
    """Slots 20-109: FeatureEngine output with the per-slot transforms (log volume, spread scale, clamp)."""
    if features is None or len(features) != MARKET_DIM:
        return [0.0] * MARKET_DIM
    out = []
    for idx, val in enumerate(features):
        if idx in MARKET_DEPRECATED_FEATURES:
            out.append(0.0)
        elif idx in MARKET_LOG_VOLUME_FEATURES:
            v_abs = abs(float(val))
            scaled_v = np.sign(float(val)) * (np.log1p(v_abs) / 10.0)
            out.append(float(np.clip(scaled_v, -5.0, 5.0)))
        elif idx == MARKET_SPREAD_FEATURE:  # spread_bps (scale so 100 bps = 1.0)
            out.append(float(np.clip(float(val) / 100.0, 0.0, 5.0)))
        else:
            out.append(float(np.clip(float(val), -5.0, 5.0)))
    return out


def macro_block(now: Optional[time.struct_time], sym_features: Optional[Sequence[float]],
                btc_features: Optional[Sequence[float]], macro) -> List[float]:
    """
    Slots 110-129. `now` is the bot's local time (UTC on the VPS); `macro` has the MacroState fields.
    """
    if now is None:
        now = time.localtime()
    minute_of_day = now.tm_hour * 60 + now.tm_min
    time_sin = np.sin(2 * np.pi * minute_of_day / 1440.0)
    time_cos = np.cos(2 * np.pi * minute_of_day / 1440.0)
    day_w = now.tm_wday
    day_sin = np.sin(2 * np.pi * day_w / 7.0)
    day_cos = np.cos(2 * np.pi * day_w / 7.0)

    btc_ok = btc_features is not None and len(btc_features) == MARKET_DIM
    btc_mom_1m = btc_features[2] if btc_ok else 0.0
    btc_mom_15m = btc_features[4] if btc_ok else 0.0

    # Cross-Coin Beta (Relative Strength to BTC)
    sym_mom_1m = sym_features[2] if sym_features is not None and len(sym_features) == MARKET_DIM else 0.0
    cross_coin_beta = sym_mom_1m / btc_mom_1m if abs(btc_mom_1m) > 1e-6 else 0.0
    cross_coin_beta = float(np.clip(cross_coin_beta, -5.0, 5.0))

    return [
        float(time_sin), float(time_cos), float(day_sin), float(day_cos), float(cross_coin_beta), 0.0,
        float(macro.sentiment_score),
        float(macro.volatility_expectation),
        float(macro.regime),
        btc_mom_1m, btc_mom_15m,
        float(macro.sp500_momentum),
        float(macro.dxy_momentum),
        float(macro.fear_greed_index),
        float(macro.vix_momentum),
        float(macro.gold_momentum),
        float(macro.treasury_yield_momentum),
        float(macro.ndx_momentum),
        float(macro.defi_tvl_momentum),
        0.0,
    ]


def whale_statarb_block(whale: Optional[Mapping], statarb: Optional[Mapping]) -> List[float]:
    """Slots 130-149: whale order flow (normalized, never raw dollars) and cross-exchange premiums."""
    whale = whale or {}
    statarb = statarb or {}
    w_buy_raw = float(whale.get("whale_buy_pressure", 0.0))
    w_sell_raw = float(whale.get("whale_sell_pressure", 0.0))
    total_whale_vol = w_buy_raw + w_sell_raw
    whale_imbalance = (w_buy_raw - w_sell_raw) / (total_whale_vol + 1e-5) if total_whale_vol > 0 else 0.0
    whale_vol_norm = float(np.clip(np.log1p(total_whale_vol) / 16.0, 0.0, 1.0))
    return [
        # Whale Tracking (Slots 130 - 139)
        float(whale.get("buy_wall_distance", 0.0)),
        float(whale.get("sell_wall_distance", 0.0)),
        float(whale_imbalance),  # Normalized [-1.0, 1.0] Order Flow Imbalance
        float(whale_vol_norm),   # Normalized [0.0, 1.0] Log Volume
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # StatArb Tracking (Slots 140 - 149)
        float(statarb.get("okx_premium", 0.0)),
        float(statarb.get("bybit_premium", 0.0)),
        float(statarb.get("okx_momentum_lead", 0.0)),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]


def cross_section_block(symbol: str, fresh_features: Mapping[str, Sequence[float]]) -> List[float]:
    """
    Slots 150-159: ranks this coin against every coin with fresh data at the same moment:
    return relative to the basket average (5m / 15m / 30m / 24h), percentile rank in [-1, 1]
    (15m / 30m / 24h, funding) and the basket's own 15m / 30m move.
    `fresh_features` maps symbol -> 90 market features, only for coins with fresh data.
    """
    fresh = {s: f for s, f in fresh_features.items() if f is not None and len(f) == MARKET_DIM}
    if symbol not in fresh or len(fresh) < 4:
        return [0.0] * CROSS_SECTION_DIM

    def col(i):
        return {s: float(f[i]) for s, f in fresh.items()}

    def rel(d):
        return d[symbol] - sum(d.values()) / len(d)

    def rank(d):
        v = d[symbol]
        below = sum(1 for x in d.values() if x < v)
        equal = sum(1 for x in d.values() if x == v)
        return 2.0 * (below + (equal - 1) / 2.0) / (len(d) - 1) - 1.0

    r5, r15, r30, r24, fund = col(3), col(4), col(5), col(6), col(80)
    return [rel(r5), rel(r15), rel(r30), rel(r24),
            rank(r15), rank(r30), rank(r24), rank(fund),
            sum(r15.values()) / len(r15), sum(r30.values()) / len(r30)]


def global_block(g: Optional[Mapping]) -> List[float]:
    """Slots 160-169 from the market-wide context (pass {} when it is stale)."""
    g = g or {}
    vals = [float(g.get(k, 0.0) or 0.0) for k in GLOBAL_KEYS]
    return vals + [0.0] * (10 - len(vals))


def finalize(obs: Sequence[float]) -> np.ndarray:
    """
    Same safety net as the live bot (and the trainer's normalize_state_tensor): NaN -> 0, +-Inf -> +-5,
    every slot clamped to [-5, 5], float32.
    """
    arr = np.asarray(obs, dtype=np.float32)
    if arr.shape != (200,):
        raise ValueError(f"State vector dimension mismatch. Expected 200, got {arr.shape}")
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0).astype(np.float32)


def round_frame(frame: Sequence[float]) -> List[float]:
    """Storage rounding of experience records (6 significant digits, see the bot's _emit_market_snapshot)."""
    return [float(f"{float(v):.6g}") for v in frame]
