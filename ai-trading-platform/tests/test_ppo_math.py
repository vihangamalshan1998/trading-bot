import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from apps.trainer.ppo import PPOTrainer
from core.ai.rollout_buffer import RolloutBuffer
from apps.research.model import MultiSymbolActorCritic

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(200, 4)
        self.critic = nn.Linear(200, 3)

    def forward(self, x):
        # x is (batch, seq_len, 200) or (batch, 200)
        if x.dim() == 3:
            last_step = x[:, -1, :]
        else:
            last_step = x
        return self.actor(last_step), self.critic(last_step)


@pytest.fixture(autouse=True)
def mock_replay_buffer_db():
    with patch("core.ai.replay_buffer.ReplayBuffer.load_cache_from_db", return_value=0):
        yield


def test_sample_action_exploration():
    model = MockModel()
    trainer = PPOTrainer(model=model)
    state = torch.randn(1, 300, 200)

    action, log_prob, value = trainer.sample_action(state, deterministic=False)
    assert action is not None
    assert action.shape == (1, 4)
    assert log_prob is not None
    assert log_prob.shape == (1, 1)
    assert torch.isfinite(log_prob).all()
    assert log_prob.item() != 0.0


def test_sample_action_deterministic_evaluation():
    model = MockModel()
    trainer = PPOTrainer(model=model)
    state = torch.randn(1, 300, 200)

    action, log_prob, value = trainer.sample_action(state, deterministic=True)
    assert action is not None
    assert action.shape == (1, 4)
    assert log_prob is None  # Strictly excluded from exploration ratio calculations


def test_rollout_buffer_filtering():
    buffer = RolloutBuffer(capacity=10)
    state = torch.randn(300, 200)
    next_state = torch.randn(300, 200)
    action = torch.randn(4)
    reward = torch.randn(3)
    done = 0.0

    # Add 4 exploration transitions
    for i in range(4):
        buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=torch.tensor([-1.5 - i * 0.1]),
            is_deterministic=False
        )

    # Add 2 deterministic evaluation transitions
    for i in range(2):
        buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=None,
            is_deterministic=True
        )

    assert len(buffer) == 6
    states, actions, rewards, next_states, dones, log_probs = buffer.build_tensors(filter_deterministic=True)
    assert len(states) == 4
    assert len(log_probs) == 4
    assert torch.isfinite(log_probs).all()


def test_train_step_rejects_dummy_zeros():
    model = MockModel()
    trainer = PPOTrainer(model=model)
    
    batch_size = 8
    states = torch.randn(batch_size, 300, 200)
    actions = torch.randn(batch_size, 4)
    rewards = torch.randn(batch_size, 3) * 5.0
    next_states = torch.randn(batch_size, 300, 200)
    dones = torch.zeros(batch_size, 1)
    dummy_log_probs = torch.zeros(batch_size, 1)  # Dummy zeros!

    success = trainer.train_step(
        tensors=(states, actions, rewards, next_states, dones),
        stored_old_log_probs=dummy_log_probs
    )
    assert success is False, "train_step must reject dummy zero log probabilities"


def test_train_step_rejects_arbitrary_constants():
    model = MockModel()
    trainer = PPOTrainer(model=model)
    
    batch_size = 8
    states = torch.randn(batch_size, 300, 200)
    actions = torch.randn(batch_size, 4)
    rewards = torch.randn(batch_size, 3) * 5.0
    next_states = torch.randn(batch_size, 300, 200)
    dones = torch.zeros(batch_size, 1)
    constant_log_probs = torch.full((batch_size, 1), -2.5)  # Identical constant values!

    success = trainer.train_step(
        tensors=(states, actions, rewards, next_states, dones),
        stored_old_log_probs=constant_log_probs
    )
    assert success is False, "train_step must reject arbitrary constant log probabilities"


def test_train_step_rejects_missing_behavior_policy():
    model = MockModel()
    trainer = PPOTrainer(model=model)
    
    batch_size = 8
    states = torch.randn(batch_size, 300, 200)
    actions = torch.randn(batch_size, 4)
    rewards = torch.randn(batch_size, 3) * 5.0
    next_states = torch.randn(batch_size, 300, 200)
    dones = torch.zeros(batch_size, 1)

    success = trainer.train_step(
        tensors=(states, actions, rewards, next_states, dones),
        stored_old_log_probs=None,
        require_stored_behavior_policy=True
    )
    assert success is False, "train_step must reject update when require_stored_behavior_policy=True but none provided"


def test_train_step_success_with_valid_rollout():
    model = MockModel()
    trainer = PPOTrainer(model=model)
    rollout = RolloutBuffer(capacity=50)

    for i in range(16):
        state = torch.randn(300, 200)
        action, log_prob, _ = trainer.sample_action(state, deterministic=False)
        next_state = torch.randn(300, 200)
        reward = torch.randn(3) * 10.0
        done = 0.0
        rollout.add(
            state=state,
            action=action.squeeze(0),
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob.squeeze(0),
            is_deterministic=False
        )

    success = trainer.train_step(rollout_buffer=rollout, epochs=2)
    assert success is True, "train_step must succeed with valid sampled rollout buffer"
