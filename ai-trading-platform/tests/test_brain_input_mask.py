from types import SimpleNamespace

import torch

from apps.research.model import SingleSymbolActorCritic
from apps.trainer.ppo import PPOTrainer, DIR_LOG_STD_MAX
from core.schemas.state_layout import (
    ACTIVE_SLOTS, BRAIN_EXCLUDED_SLOTS, MACRO_LIVE_SINCE, MACRO_PENDING_SLOTS, STATE_DIM,
)


class _Buffer:
    def __init__(self, states, timestamp=0):
        self.cache = [SimpleNamespace(timestamp=timestamp) for _ in range(64)]
        self._states = states

    def build_tensors(self, sample, seq_len=300):
        return self._states, None, None, None, None


def _trainer(model, states, timestamp=0):
    t = PPOTrainer.__new__(PPOTrainer)
    t.model = model
    t.buffer = _Buffer(states, timestamp)
    return t


def test_fitting_switches_off_account_and_position_slots():
    torch.manual_seed(0)
    model = SingleSymbolActorCritic(input_dim=STATE_DIM)
    assert _trainer(model, torch.randn(64, 4, STATE_DIM)).fit_feature_stats()

    mask = model.feat_mask
    assert mask[list(BRAIN_EXCLUDED_SLOTS)].sum() == 0
    market_active = [i for i in ACTIVE_SLOTS if i not in BRAIN_EXCLUDED_SLOTS and i not in MACRO_PENDING_SLOTS]
    assert bool((mask[market_active] == 1).all())


def test_macro_inputs_wait_for_enough_live_macro_data():
    torch.manual_seed(0)
    old = SingleSymbolActorCritic(input_dim=STATE_DIM)
    _trainer(old, torch.randn(64, 4, STATE_DIM), timestamp=MACRO_LIVE_SINCE - 86400).fit_feature_stats()
    assert old.feat_mask[list(MACRO_PENDING_SLOTS)].sum() == 0  # mostly pre-fix data: off

    new = SingleSymbolActorCritic(input_dim=STATE_DIM)
    _trainer(new, torch.randn(64, 4, STATE_DIM), timestamp=(MACRO_LIVE_SINCE + 86400) * 1000).fit_feature_stats()
    assert bool((new.feat_mask[list(MACRO_PENDING_SLOTS)] == 1).all())  # enough live data (ms timestamps): on


def test_brain_ignores_account_state():
    torch.manual_seed(0)
    model = SingleSymbolActorCritic(input_dim=STATE_DIM)
    _trainer(model, torch.randn(64, 4, STATE_DIM)).fit_feature_stats()
    model.eval()
    x = torch.randn(8, 4, STATE_DIM)
    flat_account = x.clone()
    flat_account[..., list(BRAIN_EXCLUDED_SLOTS)] = 5.0  # e.g. free margin jumps, no open position
    with torch.no_grad():
        a, v = model(x)
        b, w = model(flat_account)
    assert torch.allclose(a, b) and torch.allclose(v, w)


def test_direction_reset_is_neutral_and_keeps_the_rest():
    torch.manual_seed(0)
    model = SingleSymbolActorCritic(input_dim=STATE_DIM)
    with torch.no_grad():
        model.actor_fc.bias[0] = 5.0  # saturated "always LONG" habit
    t = PPOTrainer.__new__(PPOTrainer)
    t.model = model
    t.log_std = torch.nn.Parameter(torch.tensor([-1.6, -0.5, -0.5, -0.5]))
    t.optimizer = torch.optim.Adam(list(model.parameters()) + [t.log_std], lr=1e-3)
    other_rows = model.actor_fc.weight[1:].clone()
    critic = model.critic_fc.weight.clone()
    model.eval()
    x = torch.randn(16, 4, STATE_DIM)
    with torch.no_grad():
        before = model(x)[0][:, 0]
    assert float(before.mean()) > 0.9

    t.reset_direction_head()
    with torch.no_grad():
        after = model(x)[0][:, 0]
    assert float(after.abs().max()) < 0.05            # starts at "no opinion" (HOLD)
    assert torch.equal(model.actor_fc.weight[1:], other_rows)
    assert torch.equal(model.critic_fc.weight, critic)
    assert abs(float(t.log_std[0]) - DIR_LOG_STD_MAX) < 1e-6
