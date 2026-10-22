"""
Which state slots the history replay can rebuild (see core/schemas/state_layout.py for the names).

MISSING: no history exists (or it is not wired in yet), so replayed records carry a placeholder there.
         The trainer must not learn from these slots on replayed rows (it fills them with the fitted mean).
APPROX:  close to live but not identical (replay-vs-live correlation 0.75-0.9 on the 2026-10-22 check, or
         rebuilt from coarser data). Kept for training.
The lists are applied when the trainer reads replayed rows (not stored in them), so they can change without
rebuilding the replay. apps/replay/validate.py measures them against live records.
Everything else is rebuilt the way the live collector computes it.
"""
from core.schemas.state_layout import SLOT_BY_NAME, GLOBAL_KEYS

_MISSING_NAMES = (
    # best bid / ask and top-of-book (bookTicker archive stopped 2024-03-30, no depth5 archive)
    "spread_bps", "bid_qty", "ask_qty", "ob_imbalance", "ob_skew_l2",
    # second-scale price detail: replay vs live correlation < 0.75 (check of 2026-10-22, 34 clean records);
    # the live tick stream mixes book mids and trade prints in a way trades alone cannot reproduce
    "ret_1s", "ret_10s", "acf_1", "return_skew", "return_kurtosis",
    "wick_up_0", "wick_dn_0", "wick_dn_1", "wick_up_2", "wick_dn_2",
    # 5-minute open-interest history vs live 60 s polls; Bybit / OKX not replayed (corr < 0.65)
    "oi_delta_1m", "xex_funding_spread", "xex_funding_mean", "xex_oi_change_1h",
    # depth bands: no archive for the checked day; re-check against the bookDepth archive before trusting
    "depth_imb_0p1pct", "depth_imb_1pct", "depth_bid_usd_1pct", "depth_ask_usd_1pct",
    # liquidations (no archive)
    "liq_1m", "liq_15m",
    # news sentiment and the macro feed (not replayed yet)
    "sentiment_score", "volatility_expectation", "regime",
    "sp500_momentum", "dxy_momentum", "vix_momentum", "gold_momentum", "treasury_yield_momentum",
    "ndx_momentum", "defi_tvl_momentum",
    # whale walls need the 20-level book; OKX / Bybit prices are not replayed yet
    "buy_wall_distance", "sell_wall_distance", "okx_premium", "bybit_premium",
) + tuple(GLOBAL_KEYS)

# Correlation with live 0.75-0.9 on the 2026-10-22 check (or rebuilt from coarser data): kept for training
_APPROX_NAMES = (
    "vwap_dev_1m", "wick_up_1", "body_1", "mark_premium", "funding_rate", "rank_funding",
)

REPLAY_MISSING_SLOTS = tuple(sorted(SLOT_BY_NAME[n] for n in _MISSING_NAMES))
REPLAY_APPROX_SLOTS = tuple(sorted(SLOT_BY_NAME[n] for n in _APPROX_NAMES))
