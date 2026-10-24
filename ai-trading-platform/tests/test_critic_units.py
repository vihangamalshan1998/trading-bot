import math

import numpy as np
import torch
import torch.nn as nn

from apps.research.model import SingleSymbolActorCritic
from apps.trainer.ppo import PPOTrainer
from core.ai import critic_units as cu


def _frames(atr):
    f = torch.zeros(len(atr), 200)
    f[:, cu.ATR_SLOT] = torch.tensor(atr)
    return f


def test_legacy_brain_stays_in_percent():
    m = SingleSymbolActorCritic(input_dim=200)
    assert not cu.uses_vol_units(m)
    vals = torch.tensor([[0.1, 0.3, 0.5]])
    assert torch.equal(cu.to_pct_torch(m, vals, _frames([0.001])), vals)


def test_vol_units_round_trip_and_scale():
    m = SingleSymbolActorCritic(input_dim=200)
    m.critic_vol_units.fill_(1.0)
    frames = _frames([0.0005, 0.002])        # calm coin vs wild coin, 1-minute ATR as a fraction of price
    y_pct = torch.tensor([[0.05, 0.2, 0.4], [0.2, 0.8, 1.6]])
    z = cu.targets_torch(m, y_pct, frames)
    # the same move relative to each coin's own volatility gives the same target
    assert torch.allclose(z[0], z[1], atol=1e-6)
    assert abs(float(cu.scale_pct_torch(frames)[0, 1]) - 0.0005 * math.sqrt(60) * 100) < 1e-6
    assert torch.allclose(cu.to_pct_torch(m, z, frames), y_pct, atol=1e-6)
    # live path: same conversion, numpy, one frame, minus the shipped bias
    pct = cu.to_pct_np(m, z[1].numpy(), frames[1].numpy(), bias_pct=[0.0, 0.1, 0.2])
    assert np.allclose(pct, [0.2, 0.7, 1.4], atol=1e-6)


def test_missing_volatility_never_divides_by_zero():
    m = SingleSymbolActorCritic(input_dim=200)
    m.critic_vol_units.fill_(1.0)
    z = cu.targets_torch(m, torch.tensor([[1.0, 1.0, 1.0]]), _frames([0.0]))
    assert torch.isfinite(z).all() and float(z.abs().max()) <= cu.MAX_UNITS


def test_old_checkpoints_load_as_percent_brains():
    m = SingleSymbolActorCritic(input_dim=200)
    sd = {k: v for k, v in m.state_dict().items() if k != "critic_vol_units"}
    fresh = SingleSymbolActorCritic(input_dim=200)
    fresh.load_state_dict(sd)
    assert not cu.uses_vol_units(fresh)


class _AlwaysUp(nn.Module):
    """Forecasts +0.5% at every horizon whatever happens (the live champion's flaw on 23 Oct)."""
    feat_fitted = True

    def forward(self, x):
        n = x.shape[0]
        return torch.zeros(n, 4), torch.full((n, 3), 0.5)


def test_tournament_removes_the_forecast_bias_fitted_on_part_a():
    t = PPOTrainer.__new__(PPOTrainer)
    n = 200
    rewards = torch.zeros(n, 5)
    rewards[:, :3] = torch.randn(n, 3) * 0.1        # real moves around 0
    states = torch.zeros(n, 2, 200)
    raw = t._score_policy(_AlwaysUp(), states, rewards)
    fixed = t._score_policy(_AlwaysUp(), states, rewards, n_fit=n // 2)
    assert raw["critic_skill"][1] < -5               # a constant +0.5% guess is far worse than "no move"
    assert fixed["critic_skill"][1] > -0.5           # bias removed: about as good as "no move"
    assert abs(fixed["critic_bias"][1] - 0.5) < 0.05
