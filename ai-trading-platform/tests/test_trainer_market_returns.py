"""
Trainer correctness on TRUE market-return rewards:
- normalization is idempotent (no train/serve skew on already-normalized live frames)
- the replay buffer only emits rows graded with market returns
- the counterfactual PPO update learns a planted directional signal, a calibrated critic,
  and stays flat when no move can beat round-trip costs
"""
import os
import types
import torch
import pytest

os.environ.setdefault("LOCAL_MODE", "True")

from apps.trainer.ppo import PPOTrainer, normalize_state_tensor, ROUND_TRIP_COST_PCT
from apps.research.model import SingleSymbolActorCritic
from core.ai.replay_buffer import ReplayBuffer
from core.database.models.ai import GRADED_FLAG


def _planted_batch(n=64, seq=4, move=1.0, slot=50):
    """Signal in state slot `slot` (+1 / -1) decides the sign of the future move."""
    sign = torch.where(torch.rand(n) > 0.5, 1.0, -1.0)
    states = torch.randn(n, seq, 200) * 0.1
    states[:, :, slot] = sign[:, None]
    rewards = torch.stack([0.3 * move * sign, move * sign, 1.5 * move * sign], dim=1)
    actions = torch.zeros(n, 4)
    dones = torch.ones(n, 1)
    return states, actions, rewards, states, dones, sign


def test_normalization_is_idempotent_on_live_frames():
    frame = torch.zeros(1, 1, 200)
    frame[..., 41] = 0.135   # bid_qty already log-scaled by the live bot
    frame[..., 40] = 0.0001  # spread already bps/100
    frame[..., 30] = 488000.0  # legacy raw volume still gets scaled
    out = normalize_state_tensor(frame)
    assert torch.isclose(out[..., 41], torch.tensor(0.135)).all()
    assert torch.isclose(out[..., 40], torch.tensor(0.0001)).all()
    assert 0.0 < out[..., 30].item() < 2.0
    assert torch.equal(normalize_state_tensor(out), out)

    as_list = frame[0, 0].tolist()
    once = ReplayBuffer._normalize_frame(as_list)
    assert ReplayBuffer._normalize_frame(once) == once
    assert abs(once[41] - 0.135) < 1e-6


def test_build_tensors_uses_only_true_market_returns():
    buf = ReplayBuffer.__new__(ReplayBuffer)
    frame = [0.0] * 200
    graded = types.SimpleNamespace(market_state=[frame] * 3, reward_5m=0.2, reward_1h=-0.5, reward_4h=1.1,
                                   data_quality_flags=GRADED_FLAG, action="HOLD", next_state=None)
    legacy = types.SimpleNamespace(market_state=[frame] * 3, reward_5m=0.05, reward_1h=0.45, reward_4h=1.85,
                                   data_quality_flags=None, action="HOLD", next_state=None)
    ungraded = types.SimpleNamespace(market_state=[frame] * 3, reward_5m=None, reward_1h=None, reward_4h=None,
                                     data_quality_flags=None, action="OPEN_LONG", next_state=None)
    s, a, r, ns, d = buf.build_tensors([graded, legacy, ungraded], seq_len=5)
    assert s.shape == (1, 5, 200)
    # 3 critic targets + LONG / SHORT outcome (no path label -> blended return 0.1*0.2 + 0.6*-0.5 + 0.3*1.1)
    assert torch.allclose(r, torch.tensor([[0.2, -0.5, 1.1, 0.05, -0.05]]), atol=1e-6)
    assert d.item() == 1.0

    # With a path-aware label the outcome columns carry it (asymmetric: both sides can lose)
    graded._trade_outcome = (-0.45, -0.35)
    _, _, r, _, _ = buf.build_tensors([graded], seq_len=5)
    assert torch.allclose(r[0, 3:], torch.tensor([-0.45, -0.35]))


@pytest.fixture
def trainer():
    torch.manual_seed(0)
    t = PPOTrainer(model=SingleSymbolActorCritic(input_dim=200), limit=10)
    for g in t.optimizer.param_groups:
        g["lr"] = 2e-3
    return t


def test_counterfactual_ppo_learns_planted_direction_and_critic(trainer):
    for _ in range(120):
        s, a, r, ns, d, _ = _planted_batch()
        assert trainer.train_step(tensors=(s, a, r, ns, d), epochs=2) is True

    s, a, r, ns, d, sign = _planted_batch(n=256)
    trainer.model.eval()
    with torch.no_grad():
        acts, vals = trainer.model(normalize_state_tensor(s))
    pos = trainer.position_from_action(acts[:, 0])
    assert (pos == sign).float().mean() > 0.9, "policy should trade in the direction of the planted move"
    corr = torch.corrcoef(torch.stack([vals[:, 1], r[:, 1]]))[0, 1]
    assert corr > 0.9, "critic 1h head should predict the real market return"
    conf = (acts[:, 1] + 1) / 2
    assert conf.mean() > 0.7, "confidence should be high when calls are reliably profitable"


def test_policy_stays_flat_when_moves_cannot_beat_costs(trainer):
    tiny = ROUND_TRIP_COST_PCT * 0.3
    for _ in range(120):
        s, a, r, ns, d, _ = _planted_batch(move=tiny)
        trainer.train_step(tensors=(s, a, r, ns, d), epochs=2)

    s, a, r, ns, d, sign = _planted_batch(n=256, move=tiny)
    trainer.model.eval()
    with torch.no_grad():
        acts, _ = trainer.model(normalize_state_tensor(s))
    pos = trainer.position_from_action(acts[:, 0])
    assert (pos == 0).float().mean() > 0.8, "trading moves smaller than fees must be unlearned"
