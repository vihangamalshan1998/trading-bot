import torch
from typing import List, Tuple, Optional
from core.logging.logger import logger

class RolloutBuffer:
    """
    High-Fidelity On-Policy Rollout Buffer for PPO.
    Preserves authentic behavior-policy distributions collected at action time:
    - Stores states, actions, rewards, next_states, dones, and action-time log_probs.
    - Strictly rejects dummy zeros (torch.zeros_like) or arbitrary constant log_probs.
    - Isolates deterministic/greedy evaluation actions from on-policy exploration samples.
    """
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.next_states: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.is_deterministic: List[bool] = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        log_prob: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        is_deterministic: bool = False
    ) -> bool:
        """
        Adds a single transition to the rollout buffer with strict data validation.
        """
        if not isinstance(done, torch.Tensor):
            done = torch.tensor([float(done)], dtype=torch.float32)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([float(reward)], dtype=torch.float32)

        # Validate tensors
        if not (torch.isfinite(state).all() and torch.isfinite(action).all() and torch.isfinite(reward).all()):
            logger.warning("RolloutBuffer: Corrupted state, action, or reward with NaN/Inf rejected.")
            return False

        if not (torch.isfinite(next_state).all() and torch.isfinite(done).all()):
            logger.warning("RolloutBuffer: Corrupted next_state or done with NaN/Inf rejected.")
            return False

        # Terminal done flag must be binary
        if not torch.all((done == 0.0) | (done == 1.0)):
            logger.error("RolloutBuffer: Non-binary terminal flag in done tensor rejected.")
            return False

        # Evaluation / greedy action handling
        if is_deterministic:
            # Deterministic actions cannot have exploration log probabilities for PPO ratio calculations
            self.states.append(state.detach().cpu())
            self.actions.append(action.detach().cpu())
            self.rewards.append(reward.detach().cpu())
            self.next_states.append(next_state.detach().cpu())
            self.dones.append(done.detach().cpu())
            self.log_probs.append(torch.tensor([float("nan")]))
            self.values.append(value.detach().cpu() if value is not None else torch.zeros(1))
            self.is_deterministic.append(True)
            return True

        # Stochastic exploration action handling
        if log_prob is None:
            logger.error("RolloutBuffer: Exploration sample missing action-time log probability. Rejected.")
            return False

        if not torch.isfinite(log_prob).all():
            logger.error("RolloutBuffer: Non-finite log probability rejected.")
            return False

        # Reject dummy zeros (torch.zeros_like)
        if torch.all(log_prob == 0.0):
            logger.error("RolloutBuffer: Dummy zeros detected in log probability (torch.zeros_like rejected).")
            return False

        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.next_states.append(next_state.detach().cpu())
        self.dones.append(done.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.values.append(value.detach().cpu() if value is not None else torch.zeros(1))
        self.is_deterministic.append(False)
        return True

    def build_tensors(
        self,
        filter_deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds stacked PyTorch tensors from the rollout buffer.
        When filter_deterministic=True, deterministic evaluation samples are excluded,
        guaranteeing a clean on-policy exploration distribution for PPO updates.
        Returns:
            states, actions, rewards, next_states, dones, old_log_probs
        """
        if len(self.states) == 0:
            return (
                torch.empty(0), torch.empty(0), torch.empty(0),
                torch.empty(0), torch.empty(0), torch.empty(0)
            )

        indices = []
        for i in range(len(self.states)):
            if filter_deterministic and self.is_deterministic[i]:
                continue
            indices.append(i)

        if not indices:
            return (
                torch.empty(0), torch.empty(0), torch.empty(0),
                torch.empty(0), torch.empty(0), torch.empty(0)
            )

        states = torch.stack([self.states[i] for i in indices])
        actions = torch.stack([self.actions[i] for i in indices])
        rewards = torch.stack([self.rewards[i] for i in indices])
        next_states = torch.stack([self.next_states[i] for i in indices])
        dones = torch.stack([self.dones[i] for i in indices])
        old_log_probs = torch.stack([self.log_probs[i] for i in indices])

        # Ensure correct dimensions
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        if old_log_probs.dim() == 1:
            old_log_probs = old_log_probs.unsqueeze(-1)

        # Validate that stacked old_log_probs are valid
        if not torch.isfinite(old_log_probs).all():
            raise ValueError("RolloutBuffer: Stacked old_log_probs contain non-finite values.")
        if (old_log_probs == 0.0).all():
            raise ValueError("RolloutBuffer: Stacked old_log_probs are dummy zeros.")
        if old_log_probs.shape[0] > 1 and torch.all(old_log_probs == old_log_probs[0]):
            raise ValueError("RolloutBuffer: Stacked old_log_probs are arbitrary constant values.")

        return states, actions, rewards, next_states, dones, old_log_probs

    def clear(self):
        """Resets the rollout buffer for the next trajectory."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.is_deterministic.clear()

    def __len__(self) -> int:
        return len(self.states)
