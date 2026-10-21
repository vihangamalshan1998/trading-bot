import torch

from apps.research.model import SingleSymbolActorCritic
from apps.trainer.ppo import PPOTrainer
from core.schemas.state_layout import ACTIVE_SLOTS, BRAIN_EXCLUDED_SLOTS, STATE_DIM


class _Buffer:
    def __init__(self, states):
        self.cache = list(range(64))
        self._states = states

    def build_tensors(self, sample, seq_len=300):
        return self._states, None, None, None, None


def _trainer(model, states):
    t = PPOTrainer.__new__(PPOTrainer)
    t.model = model
    t.buffer = _Buffer(states)
    return t


def test_fitting_switches_off_account_and_position_slots():
    torch.manual_seed(0)
    model = SingleSymbolActorCritic(input_dim=STATE_DIM)
    states = torch.randn(64, 4, STATE_DIM)  # every slot varies
    assert _trainer(model, states).fit_feature_stats()

    mask = model.feat_mask
    assert mask[list(BRAIN_EXCLUDED_SLOTS)].sum() == 0
    market_active = [i for i in ACTIVE_SLOTS if i not in BRAIN_EXCLUDED_SLOTS]
    assert bool((mask[market_active] == 1).all())


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
