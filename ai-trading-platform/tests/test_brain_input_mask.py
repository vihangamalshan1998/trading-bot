from types import SimpleNamespace

import torch

from apps.research.model import SingleSymbolActorCritic
from apps.trainer.ppo import PPOTrainer, DIR_LOG_STD_MAX
from core.schemas.state_layout import (
    ACTIVE_SLOTS, BRAIN_EXCLUDED_SLOTS, FULL_INPUTS_SINCE, MIN_FULL_INPUT_ROWS, PENDING_INPUT_SLOTS, STATE_DIM,
)


class _Buffer:
    def __init__(self, states, timestamp=0, complete_rows=0):
        self.cache = [SimpleNamespace(timestamp=timestamp) for _ in range(64)]
        self._states = states
        self.complete_rows = complete_rows
        self.min_timestamp = 0.0
        self.reloads = 0

    def count_graded_since(self, ts):
        return self.complete_rows

    def load_cache_from_db(self, limit=0):
        self.reloads += 1

    def build_tensors(self, sample, seq_len=300):
        return self._states, None, None, None, None


def _trainer(model, states, timestamp=0, complete_rows=0):
    t = PPOTrainer.__new__(PPOTrainer)
    t.model = model
    t.limit = 100
    t.buffer = _Buffer(states, timestamp, complete_rows)
    return t


def test_fitting_switches_off_account_and_position_slots():
    torch.manual_seed(0)
    model = SingleSymbolActorCritic(input_dim=STATE_DIM)
    assert _trainer(model, torch.randn(64, 4, STATE_DIM)).fit_feature_stats()

    mask = model.feat_mask
    assert mask[list(BRAIN_EXCLUDED_SLOTS)].sum() == 0
    market_active = [i for i in ACTIVE_SLOTS if i not in BRAIN_EXCLUDED_SLOTS and i not in PENDING_INPUT_SLOTS]
    assert bool((mask[market_active] == 1).all())


def test_late_inputs_wait_for_enough_complete_records():
    torch.manual_seed(0)
    early = SingleSymbolActorCritic(input_dim=STATE_DIM)
    t = _trainer(early, torch.randn(64, 4, STATE_DIM), complete_rows=MIN_FULL_INPUT_ROWS - 1)
    t.fit_feature_stats()
    assert early.feat_mask[list(PENDING_INPUT_SLOTS)].sum() == 0      # not enough complete records: off
    assert t.buffer.min_timestamp == 0.0 and t.buffer.reloads == 0    # still trains on all records

    ready = SingleSymbolActorCritic(input_dim=STATE_DIM)
    t = _trainer(ready, torch.randn(64, 4, STATE_DIM), complete_rows=MIN_FULL_INPUT_ROWS)
    t.fit_feature_stats()
    active_pending = [i for i in PENDING_INPUT_SLOTS if i in ACTIVE_SLOTS]
    assert bool((ready.feat_mask[active_pending] == 1).all())          # enough: on
    assert t.buffer.min_timestamp == FULL_INPUTS_SINCE and t.buffer.reloads == 1  # ...and complete records only
    assert ready.feat_mask[list(BRAIN_EXCLUDED_SLOTS)].sum() == 0      # account / position stay off regardless


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
