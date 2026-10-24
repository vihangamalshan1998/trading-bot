"""
Units of the critic's 5m / 1h / 4h forecasts.

Legacy brains forecast the move in % of price. Brains with `critic_vol_units` = 1 forecast it in units of the
coin's own typical move over that horizon: the 1-minute ATR (state slot atr_1m, a fraction of price, the same
in live and replayed frames) scaled by sqrt(minutes). In % units the most volatile coins (PEPE, RENDER) made up
30-33% of all 1h forecast error (23 Oct 2026), so the critic mostly learned to fit them; in volatility units
every coin counts alike. Everything outside the model (live entry hurdles, tournament scoring, dashboard) keeps
working in %, through to_pct().
"""
import math

import numpy as np
import torch

from core.schemas.state_layout import SLOT_BY_NAME

ATR_SLOT = SLOT_BY_NAME["atr_1m"]
HORIZON_MINUTES = (5, 60, 240)
MIN_ATR = 0.0001          # floor of the 1-minute ATR (0.01% of price): flat or missing data never divides by ~0
MAX_UNITS = 5.0           # training targets are clipped to +-5 typical moves (data glitches, crashes)
# Training weight of each horizon in the critic loss: 5-minute moves are mostly noise (no skill for any brain on
# the 23 Oct 2026 holdout) and trades last 1-4h, so the 5m forecast only gets a small share.
HORIZON_LOSS_WEIGHTS = (0.2, 1.0, 1.0)
_SQRT_MIN = [math.sqrt(m) for m in HORIZON_MINUTES]


def uses_vol_units(model) -> bool:
    flag = getattr(model, "critic_vol_units", None)
    return flag is not None and float(flag) > 0.5


def scale_pct_torch(last_frames: torch.Tensor) -> torch.Tensor:
    """(n, 200) raw frames -> (n, 3) typical move in % of price for 5m / 1h / 4h."""
    atr = torch.clamp(torch.nan_to_num(last_frames[:, ATR_SLOT], nan=MIN_ATR), min=MIN_ATR)
    return atr.unsqueeze(1) * torch.tensor(_SQRT_MIN, dtype=last_frames.dtype) * 100.0


def scale_pct_np(last_frame) -> np.ndarray:
    """One raw frame (200,) -> (3,) typical move in % of price."""
    atr = float(np.nan_to_num(np.asarray(last_frame, dtype=np.float64)[ATR_SLOT], nan=MIN_ATR))
    return max(atr, MIN_ATR) * np.array(_SQRT_MIN) * 100.0


def to_pct_torch(model, vals: torch.Tensor, last_frames: torch.Tensor) -> torch.Tensor:
    """Critic outputs (n, 3) -> % of price, whatever units the model uses."""
    return vals * scale_pct_torch(last_frames) if uses_vol_units(model) else vals


def targets_torch(model, market_ret_pct: torch.Tensor, last_frames: torch.Tensor) -> torch.Tensor:
    """Training targets for the critic in the model's own units."""
    if not uses_vol_units(model):
        return market_ret_pct
    return torch.clamp(market_ret_pct / scale_pct_torch(last_frames), -MAX_UNITS, MAX_UNITS)


def to_pct_np(model, vals, last_frame, bias_pct=None) -> np.ndarray:
    """Live bot: one forecast (3,) -> % of price, minus the champion's holdout bias (trade policy) if any."""
    out = np.asarray(vals, dtype=np.float64)
    if uses_vol_units(model):
        out = out * scale_pct_np(last_frame)
    if bias_pct:
        out = out - np.asarray(bias_pct, dtype=np.float64)[:len(out)]
    return out
