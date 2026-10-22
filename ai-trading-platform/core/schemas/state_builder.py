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


# ----------------------------------------------------------------------------------------------------
# Batch versions for the history replay (a whole day of frames at once). They must give the same values
# as the per-frame builders above (tests/test_state_builder.py compares them on random data).
# ----------------------------------------------------------------------------------------------------
_LOG_IDX = np.array(MARKET_LOG_VOLUME_FEATURES, dtype=np.int64)
_DEP_IDX = np.array(MARKET_DEPRECATED_FEATURES, dtype=np.int64)


def market_block_batch(features: np.ndarray) -> np.ndarray:
    """(n, 90) FeatureEngine outputs -> (n, 90) transformed slots (same as market_block per row)."""
    f = features.astype(np.float64)
    out = np.clip(f, -5.0, 5.0)
    v = f[:, _LOG_IDX]
    out[:, _LOG_IDX] = np.clip(np.sign(v) * (np.log1p(np.abs(v)) / 10.0), -5.0, 5.0)
    out[:, MARKET_SPREAD_FEATURE] = np.clip(f[:, MARKET_SPREAD_FEATURE] / 100.0, 0.0, 5.0)
    out[:, _DEP_IDX] = 0.0
    return out


def macro_block_batch(times_utc: np.ndarray, sym_features: np.ndarray, btc_features: Optional[np.ndarray],
                      macro) -> np.ndarray:
    """(n,) epoch seconds (the VPS clock is UTC) -> (n, 20), same as macro_block per row."""
    t = np.floor(times_utc).astype(np.int64)
    minute_of_day = (t // 60) % 1440
    day_w = (t // 86400 + 3) % 7  # 1970-01-01 was a Thursday; tm_wday counts Monday = 0
    n = len(t)
    out = np.zeros((n, 20), dtype=np.float64)
    out[:, 0] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out[:, 1] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    out[:, 2] = np.sin(2 * np.pi * day_w / 7.0)
    out[:, 3] = np.cos(2 * np.pi * day_w / 7.0)
    if btc_features is not None:
        btc1, btc15 = btc_features[:, 2].astype(np.float64), btc_features[:, 4].astype(np.float64)
    else:
        btc1 = btc15 = np.zeros(n)
    sym1 = sym_features[:, 2].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(np.abs(btc1) > 1e-6, sym1 / btc1, 0.0)
    out[:, 4] = np.clip(beta, -5.0, 5.0)
    out[:, 6:9] = [macro.sentiment_score, macro.volatility_expectation, macro.regime]
    out[:, 9], out[:, 10] = btc1, btc15
    out[:, 11:19] = [macro.sp500_momentum, macro.dxy_momentum, macro.fear_greed_index, macro.vix_momentum,
                     macro.gold_momentum, macro.treasury_yield_momentum, macro.ndx_momentum, macro.defi_tvl_momentum]
    return out


def whale_statarb_block_batch(whale_buy: np.ndarray, whale_sell: np.ndarray) -> np.ndarray:
    """(n,) 60s whale buy / sell USD -> (n, 20), same as whale_statarb_block with no statarb data."""
    buy, sell = whale_buy.astype(np.float64), whale_sell.astype(np.float64)
    total = buy + sell
    out = np.zeros((len(buy), 20), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, 2] = np.where(total > 0, (buy - sell) / (total + 1e-5), 0.0)
    out[:, 3] = np.clip(np.log1p(total) / 16.0, 0.0, 1.0)
    return out


def cross_section_block_batch(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    (coins, n, 90) features and (coins, n) fresh-data flags -> (coins, n, 10), same as cross_section_block
    for every coin and frame (zeros where the coin is not fresh or fewer than 4 coins are).
    """
    cols = features[:, :, [3, 4, 5, 6, 80]].astype(np.float64)        # (c, n, 5): r5, r15, r30, r24, funding
    v = valid.astype(np.float64)                                       # (c, n)
    cnt = v.sum(axis=0)                                                # (n,)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (cols * v[:, :, None]).sum(axis=0) / cnt[:, None]       # (n, 5)
    out = np.zeros(features.shape[:2] + (10,), dtype=np.float64)
    out[:, :, 0:4] = cols[:, :, 0:4] - mean[None, :, 0:4]
    # percentile rank among fresh coins: (below + (equal - 1) / 2) / (count - 1) mapped to [-1, 1]
    for j, c in ((4, 1), (5, 2), (6, 3), (7, 4)):
        x = cols[:, :, c]                                              # (c, n)
        below = ((x[None, :, :] < x[:, None, :]) * v[None, :, :]).sum(axis=1)
        equal = ((x[None, :, :] == x[:, None, :]) * v[None, :, :]).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, :, j] = 2.0 * (below + (equal - 1) / 2.0) / (cnt[None, :] - 1) - 1.0
    out[:, :, 8] = mean[None, :, 1]
    out[:, :, 9] = mean[None, :, 2]
    ok = (valid & (cnt[None, :] >= 4))[:, :, None]
    return np.where(ok, out, 0.0)


def finalize_batch(obs: np.ndarray) -> np.ndarray:
    """(n, 200) -> float32 with the same safety net as finalize()."""
    arr = obs.astype(np.float32)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0).astype(np.float32)
