"""
Which state slots the history replay can rebuild (see core/schemas/state_layout.py for the names).

MISSING: no history exists (or it is not wired in yet), so replayed records carry a placeholder there.
         The trainer must not learn from these slots on replayed rows (it fills them with the fitted mean).
APPROX:  rebuilt from coarser history than live (5-minute OI, 0.2% depth band, Binance-only cross-exchange
         inputs, last-trade instead of book-mid ticks). Usable; the replay-vs-live check measures how close.
Everything else is rebuilt the way the live collector computes it.
"""
from core.schemas.state_layout import SLOT_BY_NAME, GLOBAL_KEYS

_MISSING_NAMES = (
    # best bid / ask and top-of-book (bookTicker archive stopped 2024-03-30, no depth5 archive)
    "spread_bps", "bid_qty", "ask_qty", "ob_imbalance", "ob_skew_l2",
    # liquidations (no archive)
    "liq_1m", "liq_15m",
    # Bybit / OKX funding spread
    "xex_funding_spread",
    # news sentiment and the macro feed (not replayed yet)
    "sentiment_score", "volatility_expectation", "regime",
    "sp500_momentum", "dxy_momentum", "vix_momentum", "gold_momentum", "treasury_yield_momentum",
    "ndx_momentum", "defi_tvl_momentum",
    # whale walls need the 20-level book; OKX / Bybit prices are not replayed yet
    "buy_wall_distance", "sell_wall_distance", "okx_premium", "bybit_premium",
) + tuple(GLOBAL_KEYS)

_APPROX_NAMES = (
    "depth_imb_0p1pct", "depth_imb_1pct", "depth_bid_usd_1pct", "depth_ask_usd_1pct",
    "funding_rate", "oi_delta_1m", "xex_funding_mean", "xex_oi_change_1h", "rank_funding",
)

REPLAY_MISSING_SLOTS = tuple(sorted(SLOT_BY_NAME[n] for n in _MISSING_NAMES))
REPLAY_APPROX_SLOTS = tuple(sorted(SLOT_BY_NAME[n] for n in _APPROX_NAMES))
