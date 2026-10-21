"""
Canonical layout of the 200-slot state vector ("God Mode" canvas).

Single source of truth for slot positions. The live bot (_build_state_vector), the trainer
(normalize_state_tensor / feature statistics) and the replay buffer all read from here instead of
hard-coding indices, so adding a future data source means editing ONE table.

Status:
  ACTIVE      the slot carries data and is fed to the model.
  RESERVED    free canvas for future connections. Always 0 and masked out of the model.
  DEPRECATED  used to carry something meaningless (kept so old records stay aligned). Masked out.

Rules for adding a new input later:
  1. Pick a RESERVED slot, give it a name / source / transform, flip it to ACTIVE.
  2. Never change the meaning of an existing slot (old records would silently become wrong).
  3. Refit feature statistics (python -m apps.trainer.ppo --refit-feature-stats) once enough
     records contain the new input, then let the holdout tournament decide if it helps.

Transforms (applied by the live bot before the frame is stored):
  none        value used as-is (already a ratio / bounded indicator)
  log_volume  sign(x) * log1p(|x|) / 10   (raw coin quantities / USD volumes)
  spread_bps  x / 100                     (basis points -> 100 bps = 1.0)
"""
from dataclasses import dataclass
from typing import List, Tuple

STATE_DIM = 200
MARKET_OFFSET = 20  # FeatureEngine output index i lives in state slot MARKET_OFFSET + i
MARKET_DIM = 90

ACTIVE = "ACTIVE"
RESERVED = "RESERVED"
DEPRECATED = "DEPRECATED"


@dataclass(frozen=True)
class Slot:
    index: int
    name: str
    group: str
    status: str
    transform: str = "none"


def _block(start: int, group: str, entries: List[Tuple[str, str, str]]) -> List[Slot]:
    return [Slot(start + i, name, group, status, transform) for i, (name, status, transform) in enumerate(entries)]


A, R, D = ACTIVE, RESERVED, DEPRECATED
N, LOG, BPS = "none", "log_volume", "spread_bps"

_PORTFOLIO = _block(0, "portfolio", [
    ("wallet_balance_ratio", A, N), ("equity_ratio", A, N), ("free_margin_ratio", A, N), ("margin_utilization", A, N),
    ("exposure_ratio", A, N), ("portfolio_heat", A, N), ("session_win_rate", A, N), ("account_drawdown", A, N),
    ("portfolio_variance", A, N), ("time_since_win", A, N),
])
_POSITION = _block(10, "position", [
    ("position_direction", A, N), ("entry_distance", A, N), ("leverage_norm", A, N), ("current_pnl_pct", A, N),
    ("distance_to_liquidation", A, N), ("max_unrealized_pnl", A, N), ("max_drawdown_pnl", A, N), ("time_held", A, N),
    ("entry_vwap_distance", A, N), ("accumulated_funding", A, N),
])
_MARKET = _block(MARKET_OFFSET, "market", [
    # Price [10]
    ("ret_1s", A, N), ("ret_10s", A, N), ("ret_1m", A, N), ("ret_5m", A, N), ("ret_15m", A, N),
    ("ret_30m", A, N), ("ret_24h", A, N), ("pad_27", R, N), ("price_range_30m", A, N), ("momentum_1m", A, N),
    # Volume [10]
    ("vol_1m", A, LOG), ("buy_vol_1m", A, LOG), ("sell_vol_1m", A, LOG), ("trade_imbalance_1m", A, N),
    # vwap_1m / vwap_15m used to hold RAW prices (e.g. 82,000) -> meaningless after clamping. Deviation slots below carry the signal.
    ("vwap_1m_raw", D, N), ("vwap_15m_raw", D, N),
    ("vwap_dev_1m", A, N), ("vwap_dev_15m", A, N), ("realized_vol_1m", A, N), ("vol_24h", A, LOG),
    # Orderbook & StatArb [15]
    ("spread_bps", A, BPS), ("bid_qty", A, LOG), ("ask_qty", A, LOG), ("ob_imbalance", A, N), ("ob_skew_l2", A, N),
    ("pad_45", R, N), ("pad_46", R, N), ("pad_47", R, N), ("pad_48", R, N), ("pad_49", R, N),
    ("hurst", A, N), ("acf_1", A, N), ("return_skew", A, N), ("return_kurtosis", A, N), ("pad_54", R, N),
    # Technical indicators [35]
    ("rsi_1m", A, N), ("rsi_5m", A, N), ("rsi_15m", A, N), ("rsi_1h", A, N),
    ("macd_line", A, N), ("macd_signal", A, N), ("macd_hist", A, N),
    ("ema9_dist", A, N), ("ema21_dist", A, N), ("ema50_dist", A, N), ("ema200_dist", A, N),
    ("sma50_dist", A, N), ("sma200_dist", A, N), ("bb_width", A, N), ("bb_position", A, N),
    ("ema_4h_dist", A, N), ("atr_1m", A, N), ("stoch_k", A, N), ("stoch_d", A, N), ("adx", A, N),
] + [(f"pad_{75 + i}", R, N) for i in range(15)] + [
    # Candlesticks [10]
    ("wick_up_0", A, N), ("wick_dn_0", A, N), ("body_0", A, N), ("wick_up_1", A, N), ("wick_dn_1", A, N),
    ("body_1", A, N), ("wick_up_2", A, N), ("wick_dn_2", A, N), ("body_2", A, N), ("pad_99", R, N),
    # Derivatives [10]
    ("funding_rate", A, N), ("liq_1m", A, LOG), ("liq_15m", A, LOG), ("oi_delta_1m", A, N), ("long_short_ratio", A, N),
    ("mark_premium", A, N), ("pad_106", R, N), ("pad_107", R, N), ("pad_108", R, N), ("pad_109", R, N),
])
_MACRO = _block(110, "macro", [
    ("time_sin", A, N), ("time_cos", A, N), ("day_sin", A, N), ("day_cos", A, N), ("cross_coin_beta", A, N),
    ("pad_115", R, N), ("sentiment_score", A, N), ("volatility_expectation", A, N), ("regime", A, N),
    ("btc_mom_1m", A, N), ("btc_mom_15m", A, N), ("sp500_momentum", A, N), ("dxy_momentum", A, N),
    ("fear_greed_index", A, N), ("vix_momentum", A, N), ("gold_momentum", A, N), ("treasury_yield_momentum", A, N),
    ("ndx_momentum", A, N), ("defi_tvl_momentum", A, N), ("pad_129", R, N),
])
_WHALE = _block(130, "whale", [
    ("buy_wall_distance", A, N), ("sell_wall_distance", A, N), ("whale_imbalance", A, N), ("whale_volume_norm", A, N),
] + [(f"pad_{134 + i}", R, N) for i in range(6)])
_STATARB = _block(140, "statarb", [
    ("okx_premium", A, N), ("bybit_premium", A, N), ("okx_momentum_lead", A, N),
] + [(f"pad_{143 + i}", R, N) for i in range(7)])
_FUTURE = [Slot(i, f"reserved_{i}", "future", RESERVED) for i in range(150, STATE_DIM)]

SLOTS: List[Slot] = _PORTFOLIO + _POSITION + _MARKET + _MACRO + _WHALE + _STATARB + _FUTURE

assert len(SLOTS) == STATE_DIM and all(s.index == i for i, s in enumerate(SLOTS)), "state layout must cover 0..199 in order"

ACTIVE_SLOTS = tuple(s.index for s in SLOTS if s.status == ACTIVE)
LOG_VOLUME_SLOTS = tuple(s.index for s in SLOTS if s.transform == LOG)
SPREAD_SLOT = next(s.index for s in SLOTS if s.transform == BPS)
SLOT_BY_NAME = {s.name: s.index for s in SLOTS}

# FeatureEngine-relative indices (0..89) for the live bot's per-feature transforms
MARKET_LOG_VOLUME_FEATURES = tuple(i - MARKET_OFFSET for i in LOG_VOLUME_SLOTS if MARKET_OFFSET <= i < MARKET_OFFSET + MARKET_DIM)
MARKET_SPREAD_FEATURE = SPREAD_SLOT - MARKET_OFFSET
MARKET_DEPRECATED_FEATURES = tuple(s.index - MARKET_OFFSET for s in _MARKET if s.status == DEPRECATED)


def active_mask() -> List[float]:
    """1.0 for ACTIVE slots, 0.0 for RESERVED / DEPRECATED."""
    return [1.0 if s.status == ACTIVE else 0.0 for s in SLOTS]
