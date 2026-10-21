import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Tuple
import asyncio
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()

from core.db.redis import redis_manager
from core.ai.replay_buffer import ReplayBuffer
from core.ai.rollout_buffer import RolloutBuffer
from core.logging.logger import logger, set_log_file
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry
from core.ai import trainer_status
from core.config.settings import settings
from core.risk import exit_policy
from core.schemas.state_layout import LOG_VOLUME_SLOTS, SPREAD_SLOT, ACTIVE_SLOTS, STATE_DIM, WHALE_IMBALANCE_SLOT, WHALE_VOLUME_SLOT, ADDED_2026_10_SLOTS, BRAIN_EXCLUDED_SLOTS
from core.schemas.state_layout import MACRO_PENDING_SLOTS, MACRO_LIVE_SINCE, MACRO_MIN_LIVE_SHARE

# --- Trading economics used to score actions (all in % of price / notional) ---
# Live entry rule: |action| > 0.20 opens a position (see trading_bot.main side determination)
DIRECTION_DEADBAND = 0.20
# Round trip (maker entry + taker exit + slippage), shared with the live exit policy
ROUND_TRIP_COST_PCT = exit_policy.ROUND_TRIP_COST * 100.0
# Blend of 5m / 1h / 4h forward returns matching the 1-4h swing horizon the exits are built for
REWARD_HORIZON_WEIGHTS = exit_policy.HORIZON_WEIGHTS
# Live exit envelope (core/risk/exit_policy): widest stop and widest hard take-profit, in % of price
STOP_LOSS_PCT = exit_policy.MAX_STOP * 100.0
TAKE_PROFIT_PCT = exit_policy.MAX_STOP * exit_policy.HARD_TP_RR * 100.0
TRAINER_STATE_PATH = "models/candidates/trainer_state.pt"
# Direction exploration noise must stay small relative to the 0.20 deadband. With wide noise the
# STOCHASTIC policy's optimum drifts toward the "less costly losing side" instead of flat, and the
# DETERMINISTIC mean used live then trades into fees. sigma in [0.25, 0.30]: the floor was 0.08, the
# noise had shrunk to 0.20, and on 82-100% of states all K samples then landed on the same side
# (zero advantage, no direction learning).
DIR_LOG_STD_MIN = -1.4
DIR_LOG_STD_MAX = -1.2

def normalize_state_tensor(states: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a batch of state tensors (shape: [batch, seq_len, 200] or [batch, 200] or [seq_len, 200])
    so raw wallet balances ($5,000) and entry prices ($81,000) are converted to clean normalized ratios.
    This guarantees that the 90 market indicators are never drowned out in LayerNorm(200).
    """
    if states is None:
        return states
    s = states.clone()
    base_capital = 5000.0
    
    if s.shape[-1] != 200:
        return s
        
    # Slot 0: wallet_balance
    s[..., 0] = torch.where(s[..., 0] > 5.0, torch.clamp(s[..., 0] / base_capital, 0.0, 5.0), torch.clamp(s[..., 0], 0.0, 5.0))
    
    # Slot 1: equity
    s[..., 1] = torch.where(s[..., 1] > 5.0, torch.clamp(s[..., 1] / base_capital, 0.0, 5.0), torch.clamp(s[..., 1], 0.0, 5.0))
    
    # Slot 2: free_margin
    s[..., 2] = torch.where(s[..., 2] > 1.0, torch.clamp(s[..., 2] / base_capital, 0.0, 1.0), torch.clamp(s[..., 2], 0.0, 1.0))
    
    # Slot 4: exposure
    s[..., 4] = torch.where(s[..., 4] > 10.0, torch.clamp(s[..., 4] / base_capital, 0.0, 10.0), torch.clamp(s[..., 4], 0.0, 10.0))
    
    # Slot 10: position direction [-1.0, 1.0]
    s[..., 10] = torch.where(torch.abs(s[..., 10]) > 1.0, torch.sign(s[..., 10]), s[..., 10])
    
    # Slot 11: entry_price distance (if abs > 1.0, raw asset price like $81,000 BTC -> reset to 0.0)
    s[..., 11] = torch.where(torch.abs(s[..., 11]) > 1.0, torch.zeros_like(s[..., 11]), s[..., 11])
    
    # Volume / orderbook-quantity / liquidation slots (log_volume transform in core.schemas.state_layout).
    # Log-scaling compresses million-token quantities (e.g. 488,000 DOGE) to clean [0, 2.0] range.
    # IDEMPOTENT: the live bot and the replay buffer already scale these, so only touch values that are
    # still raw (> 5.0). Re-scaling scaled values created a train/serve skew (features ~10x smaller in training).
    for vol_idx in LOG_VOLUME_SLOTS:
        v = s[..., vol_idx]
        s[..., vol_idx] = torch.where(torch.abs(v) > 5.0, torch.sign(v) * torch.log1p(torch.abs(v)) / 10.0, v)

    # Whale flow: legacy records stored RAW dollar pressure (e.g. 109,359) where the bot now writes a
    # normalized imbalance [-1, 1] and log volume [0, 1]. Map legacy values onto the live definitions.
    wi, wv = s[..., WHALE_IMBALANCE_SLOT], s[..., WHALE_VOLUME_SLOT]
    s[..., WHALE_IMBALANCE_SLOT] = torch.where(torch.abs(wi) > 1.0, torch.zeros_like(wi), wi)
    s[..., WHALE_VOLUME_SLOT] = torch.where(wv > 1.0, torch.clamp(torch.log1p(wv) / 16.0, 0.0, 1.0), wv)

    # spread_bps (scale so 100 bps = 1.0) — only raw legacy values
    s[..., SPREAD_SLOT] = torch.where(s[..., SPREAD_SLOT] > 5.0, torch.clamp(s[..., SPREAD_SLOT] / 100.0, 0.0, 5.0), s[..., SPREAD_SLOT])
    
    # Universal clamp across all slots to protect LayerNorm from extreme outliers
    s = torch.clamp(s, -5.0, 5.0)
    
    return s


class PPOTrainer:
    """
    Phase 11: Proximal Policy Optimization (PPO) Training Loop.
    Samples from the MySQL ReplayBuffer or RolloutBuffer and updates the Actor-Critic model.
    """
    def __init__(self, model: SingleSymbolActorCritic, lr: float = 3e-4, gamma: float = 0.99, clip_epsilon: float = 0.2, limit: int = 300):
        self.model = model
        self.log_std = nn.Parameter(torch.tensor([-1.6, -0.5, -0.5, -0.5]))
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + [self.log_std],
            lr=lr,
            weight_decay=1e-5
        )
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.actions_per_state = 4  # K fresh policy samples per replay state (leave-one-out baseline)
        self.buffer = ReplayBuffer()
        self.rollout_buffer = RolloutBuffer()
        self.limit = limit
        
        # V3 Upgrade: The 300-frame sequence is massive (2MB per record). 
        # Cache limit math: 300 exp × 300 frames × 200 features × 4 bytes = ~70MB cache.
        self.buffer.load_cache_from_db(limit=self.limit)

        
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        gae_lambda: float = 0.95,
        is_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimation (GAE).
        For sequential trajectories (is_trajectory=True):
            Recursively accumulates backward over the temporal trajectory:
            delta_t = reward_t + gamma * next_value_t * (1 - done_t) - value_t
            advantage_t = delta_t + gamma * gae_lambda * (1 - done_t) * advantage_{t+1}
        For independent minibatch transitions (is_trajectory=False):
            Computes exact single-step TD error to avoid falsely chaining independent transitions:
            delta_i = reward_i + gamma * next_value_i * (1 - done_i) - value_i
        """
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)

        # Multi-step sequential trajectory handling (T, Batch, ...)
        if is_trajectory and rewards.dim() >= 2 and rewards.size(0) > 1:
            T = rewards.size(0)
            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros_like(rewards[-1])
            
            # Recursive backward iteration over the sequential trajectory
            for t in reversed(range(T)):
                non_terminal = 1.0 - dones[t]
                n_val = next_values[t] if t == T - 1 else values[t + 1]
                delta = rewards[t] + (self.gamma * n_val * non_terminal) - values[t]
                last_gae = delta + (self.gamma * gae_lambda * non_terminal * last_gae)
                advantages[t] = last_gae
                
            returns = advantages + values
            return advantages, returns
        else:
            # Independent transition handling
            non_terminal = 1.0 - dones
            delta = rewards + (self.gamma * next_values * non_terminal) - values
            advantages = delta.clone()
            returns = advantages + values
            return advantages, returns
        
    def _action_std(self) -> torch.Tensor:
        """Per-dimension policy std. Direction (dim 0) noise is kept narrow relative to the entry deadband."""
        dir_log_std = torch.clamp(self.log_std[0:1], min=DIR_LOG_STD_MIN, max=DIR_LOG_STD_MAX)
        rest_log_std = torch.clamp(self.log_std[1:], min=-2.0, max=0.5)
        return torch.exp(torch.cat([dir_log_std, rest_log_std]))

    def sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Samples an action from the parametric continuous policy distribution at action time.
        - If deterministic=False (exploration): samples action ~ Normal(mu, std),
          and computes the exact action-time log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True).
        - If deterministic=True (evaluation): greedy action = mu,
          log_prob = None (strictly excluded from on-policy exploration ratio calculations).
        """
        self.model.eval()
        with torch.no_grad():
            if state.dim() == 2:
                state = state.unsqueeze(0)
            state = normalize_state_tensor(state)
            action_mean, value = self.model(state)
            action_std = self._action_std()
            dist = torch.distributions.Normal(action_mean, action_std)
            if deterministic:
                action = action_mean
                log_prob = None
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob, value

    # ------------------------------------------------------------------
    # Trading economics shared by training and evaluation
    # ------------------------------------------------------------------
    def position_from_action(self, action_dir: torch.Tensor) -> torch.Tensor:
        """Mirror of the live bot's entry rule: |action| > 0.20 opens a full long / short, else flat."""
        return torch.where(action_dir > DIRECTION_DEADBAND, torch.ones_like(action_dir),
                           torch.where(action_dir < -DIRECTION_DEADBAND, -torch.ones_like(action_dir),
                                       torch.zeros_like(action_dir)))

    def trade_return(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Blended forward market return (% of price) a position opened now would capture, clipped to
        the bot's hard stop-loss / take-profit envelope. rewards: (batch, 3) -> (batch, 1)
        """
        w = torch.tensor(REWARD_HORIZON_WEIGHTS, dtype=rewards.dtype, device=rewards.device)
        blended = (rewards * w).sum(dim=-1, keepdim=True)
        return torch.clamp(blended, -STOP_LOSS_PCT, TAKE_PROFIT_PCT)

    def trade_outcomes(self, rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (long_pct, short_pct), each (batch, 1): gross % a LONG / SHORT opened at that state actually
        realized under the live exit policy (replay buffer columns 3 / 4, path-aware: a trade that hits
        its stop before the price recovers is a LOSS). 3-column rewards fall back to the blended return.
        """
        if rewards.shape[-1] >= 5:
            return rewards[:, 3:4], rewards[:, 4:5]
        blended = self.trade_return(rewards[:, :3])
        return blended, -blended

    def action_pnl(self, action_dir: torch.Tensor, long_ret: torch.Tensor, short_ret: torch.Tensor) -> torch.Tensor:
        """Net PnL (% of notional) of taking `action_dir`, from the realized LONG / SHORT outcome, after round-trip costs."""
        pos = self.position_from_action(action_dir)
        gross = torch.where(pos > 0, long_ret.expand_as(pos),
                            torch.where(pos < 0, short_ret.expand_as(pos), torch.zeros_like(pos)))
        return gross - ROUND_TRIP_COST_PCT * pos.abs()

    def train_step(
        self,
        batch_size: int = 64,
        epochs: int = 4,
        tensors: tuple = None,
        rollout_buffer: RolloutBuffer = None,
        stored_old_log_probs: torch.Tensor = None,
        require_stored_behavior_policy: bool = False,
        use_off_policy_correction: bool = False
    ):
        """
        One PPO update with strict data validation.

        rewards are TRUE forward market returns (%, direction-free) for 5m / 1h / 4h.

        Critic: direct regression of those returns (what the live bot reads as "critic_return %").
                No bootstrapping: samples are independent one-step decisions.
        Actor (direction head):
          - Replay data (no behavior log-probs): the market move is known, so the PnL of ANY action
            can be computed. We sample K fresh actions per state from the current policy, score them
            exactly, and use a leave-one-out baseline. This is genuinely on-policy PPO and learns from
            every stored state (HOLDs included), not just the handful of executed trades.
          - Rollout / stored behavior log-probs: classic PPO on the logged actions, scored the same way.
        Confidence head: calibrated probability that the current direction call is profitable after costs.
        Size head: follows confidence (bet more only when the call is more likely to win).
        """
        if rollout_buffer is not None:
            states, actions, rewards, next_states, dones, stored_old_log_probs = rollout_buffer.build_tensors(filter_deterministic=True)
        elif tensors is not None:
            states, actions, rewards, next_states, dones = tensors
        else:
            batch = self.buffer.sample(batch_size)
            if len(batch) < batch_size:
                logger.warning(f"Not enough samples in replay buffer to train. Got {len(batch)}, needed {batch_size}")
                return False
            states, actions, rewards, next_states, dones = self.buffer.build_tensors(batch, seq_len=300)

        if len(states) == 0:
            return False

        # Priority 2: Never silently reset model parameters if non-finite weights are found. Stop update & report.
        is_nan = any(not torch.isfinite(p).all() for p in self.model.parameters())
        if is_nan:
            logger.error("CRITICAL: Non-finite model weights detected. Aborting training step to preserve checkpoint integrity.")
            return False

        # Task 3 Data Validation: Validate dimensions
        if states.ndim != 3 or states.shape[-1] != 200:
            logger.error(f"PPO: Invalid state tensor dimension {states.shape}. Expected (batch, seq_len, 200).")
            return False
        if actions.ndim != 2 or actions.shape[-1] != 4:
            logger.error(f"PPO: Invalid action tensor dimension {actions.shape}. Expected (batch, 4).")
            return False
        if rewards.ndim != 2 or rewards.shape[-1] not in (3, 5):
            logger.error(f"PPO: Invalid reward tensor dimension {rewards.shape}. Expected (batch, 3) or (batch, 5).")
            return False
        if dones.ndim != 2 or dones.shape[-1] != 1:
            logger.error(f"PPO: Invalid dones tensor dimension {dones.shape}. Expected (batch, 1).")
            return False

        # Normalize state vectors (idempotent: already-normalized live frames pass through unchanged)
        states = normalize_state_tensor(states)

        # Detect and filter corrupted non-finite samples (Never silently convert with nan_to_num)
        finite_s = torch.isfinite(states).all(dim=1).all(dim=1)
        finite_a = torch.isfinite(actions).all(dim=1)
        finite_r = torch.isfinite(rewards).all(dim=1)
        finite_d = torch.isfinite(dones).all(dim=1)

        valid_mask = finite_s & finite_a & finite_r & finite_d
        corrupted_count = int((~valid_mask).sum().item())
        if corrupted_count > 0:
            logger.warning(f"PPO: Logged and rejected {corrupted_count} corrupted samples containing NaN/Inf values. Retaining only valid historical experiences.")
            states = states[valid_mask]
            actions = actions[valid_mask]
            rewards = rewards[valid_mask]
            dones = dones[valid_mask]
            if stored_old_log_probs is not None and len(stored_old_log_probs) == len(valid_mask):
                stored_old_log_probs = stored_old_log_probs[valid_mask]

        if len(states) < 4:
            logger.warning(f"PPO: Insufficient valid non-corrupted samples remaining ({len(states)} < 4). Aborting update.")
            return False

        # Verify terminal flags are strictly binary {0.0, 1.0}
        if not torch.all((dones == 0.0) | (dones == 1.0)):
            logger.error("PPO: Non-binary terminal flags detected in dones tensor. Aborting update.")
            return False

        # Verify action bounds
        if (torch.abs(actions) > 10.0).any():
            logger.error("PPO: Out-of-bounds action values detected (> 10.0). Aborting update.")
            return False

        # 1. Behavior Policy Old Log Probabilities
        if stored_old_log_probs is not None:
            if not torch.isfinite(stored_old_log_probs).all():
                logger.error("PPO: Stored old-policy log probabilities contain non-finite values. Aborting update.")
                return False
            if stored_old_log_probs.shape[0] != states.shape[0]:
                logger.error(f"PPO: Dimension mismatch for stored old log probs: {stored_old_log_probs.shape[0]} vs {states.shape[0]}.")
                return False
            # Reject dummy zeros (torch.zeros_like)
            if (stored_old_log_probs == 0.0).all():
                logger.error("PPO: Stored old-policy log probabilities are dummy zeros (torch.zeros_like rejected).")
                return False
            # Reject arbitrary constant values
            if stored_old_log_probs.shape[0] > 1 and torch.all(stored_old_log_probs == stored_old_log_probs[0]):
                logger.error("PPO: Stored old-policy log probabilities are arbitrary constant values. Rejected.")
                return False
        elif require_stored_behavior_policy:
            logger.error("PPO: Replay samples lack stored behavior policy log probabilities. Cannot execute on-policy PPO update without actual behavior policy.")
            return False

        safe_rewards = torch.clamp(rewards, min=-15.0, max=15.0)
        market_ret = safe_rewards[:, :3]                         # critic targets (5m / 1h / 4h)
        long_ret, short_ret = self.trade_outcomes(safe_rewards)  # actor targets, (B, 1) each
        use_logged_actions = stored_old_log_probs is not None
        K = 1 if use_logged_actions else self.actions_per_state

        self.model.train()
        with torch.no_grad():
            old_mean, _ = self.model(states)
            dir_std = self._action_std()[0]
            if use_logged_actions:
                sampled_dir = actions[:, 0:1]                                   # (B, 1)
                old_log_probs = stored_old_log_probs.reshape(-1, 1)
                pnl = self.action_pnl(sampled_dir, long_ret, short_ret)         # (B, 1)
                advantages = pnl - pnl.mean()
            else:
                dist_old = torch.distributions.Normal(old_mean[:, 0:1].expand(-1, K), dir_std)
                sampled_dir = dist_old.sample()                                 # (B, K)
                old_log_probs = dist_old.log_prob(sampled_dir)                  # (B, K)
                pnl = self.action_pnl(sampled_dir, long_ret, short_ret)         # (B, K)
                # Leave-one-out baseline: compare each sampled action to the others for the SAME state
                advantages = pnl - (pnl.sum(dim=1, keepdim=True) - pnl) / (K - 1)

            adv_std = advantages.std(unbiased=False)
            # Phase 1.A: Never manufacture random advantages.
            # - No reward variation at all -> nothing to learn, defer the whole update.
            # - Rewards vary but every sampled action scored the same (converged policy) -> no actor
            #   gradient this step, but the critic / confidence heads still learn.
            if advantages.numel() <= 1 or (adv_std <= 1e-6 and market_ret.std(unbiased=False) <= 1e-6):
                logger.info(f"PPO: Insufficient reward variation in batch (adv_std={adv_std.item():.6f} <= 1e-6). Deferring update.")
                return False
            if adv_std <= 1e-6:
                advantages = torch.zeros_like(advantages)
            else:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

            # Confidence target: would the current direction call win after costs (real trade outcome)?
            # (When the policy is flat, the call is judged as if it had picked the better side.)
            current_pos = self.position_from_action(old_mean[:, 0:1])
            call_ret = torch.where(current_pos > 0, long_ret,
                                   torch.where(current_pos < 0, short_ret, torch.maximum(long_ret, short_ret)))
            conf_target = ((call_ret - ROUND_TRIP_COST_PCT) > 0).float()

        # 2. PPO Epochs
        for _ in range(epochs):
            action_preds, values = self.model(states)

            if use_logged_actions:
                dist = torch.distributions.Normal(action_preds, self._action_std())
                log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1).mean()
            else:
                dist = torch.distributions.Normal(action_preds[:, 0:1].expand(-1, K), self._action_std()[0])
                log_probs = dist.log_prob(sampled_dir)
                entropy = dist.entropy()[:, 0].mean()

            # Numerically stable probability ratio
            diff = torch.clamp(log_probs - old_log_probs, min=-10.0, max=10.0)
            ratio = torch.exp(diff)

            # Off-policy correction: Importance sampling truncation (V-trace style)
            if use_off_policy_correction:
                ratio = torch.clamp(ratio, max=1.0)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic: robust regression of true multi-horizon market returns (heavy-tailed -> Huber)
            critic_loss = nn.functional.smooth_l1_loss(values, market_ret, beta=1.0)

            # Confidence head (tanh space -> probability) and size head anchored to it
            conf_prob = torch.clamp((action_preds[:, 1:2] + 1.0) / 2.0, 1e-4, 1.0 - 1e-4)
            conf_loss = nn.functional.binary_cross_entropy(conf_prob, conf_target)
            size_loss = nn.functional.mse_loss(action_preds[:, 2:3], (2.0 * conf_prob.detach() - 1.0))

            entropy_coef = 0.002
            loss = actor_loss + 0.5 * critic_loss + 0.2 * conf_loss + 0.05 * size_loss - entropy_coef * entropy

            if not torch.isfinite(loss):
                logger.error("PPO Loss is NaN or Inf! Skipping step.")
                break

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + [self.log_std], 0.5)
            self.optimizer.step()

        with torch.no_grad():
            batch_reward = float(rewards[:, 0].mean().item())  # Dashboard tracks 5m market return
            mean_pnl = float(pnl.mean().item())
            # Action distribution of the CURRENT policy (deterministic), not of the logged data
            cur_pos = self.position_from_action(action_preds[:, 0:1].detach())
            holds = float((cur_pos == 0).float().mean().item())
            buys = float((cur_pos > 0).float().mean().item())
            sells = float((cur_pos < 0).float().mean().item())

        logger.info(
            f"PPO Training Step Complete | Loss: {loss.item():.4f} | Actor: {actor_loss.item():.4f} | "
            f"Critic: {critic_loss.item():.4f} | Conf: {conf_loss.item():.4f} | Entropy: {entropy.item():.4f} | "
            f"Sampled PnL: {mean_pnl:+.4f}% | Policy L/S/H: {buys:.2f}/{sells:.2f}/{holds:.2f} | "
            f"MktRet5m: {batch_reward:+.4f}% | Critic5m: {values[:, 0].mean().item():+.4f}%"
        )

        # MEMORY FIX: Now safe to delete large tensors - all scalar values already extracted above
        del states, actions, rewards, next_states, dones
        del old_mean, old_log_probs, safe_rewards, market_ret, long_ret, short_ret, advantages
        # Note: Do NOT call gc.collect() here. It causes GIL deadlocks and freezes PyTorch.

        payload = {
            "timestamp": time.time(),
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "cumulative_reward": batch_reward,
            "action_dist": {
                "hold": holds,
                "buy": buys,
                "sell": sells
            }
        }

        if os.getenv("LOCAL_MODE") != "True":
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._publish_metrics(payload))
            except RuntimeError:
                asyncio.run(self._publish_metrics(payload))

        return True

    def save_trainer_state(self, path: str = TRAINER_STATE_PATH):
        """log_std and optimizer moments live outside model.state_dict(); persist them so restarts don't reset exploration."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({"log_std": self.log_std.detach().clone(), "optimizer": self.optimizer.state_dict()}, path)
        except Exception as e:
            logger.warning(f"Could not save trainer state: {e}")

    def load_trainer_state(self, path: str = TRAINER_STATE_PATH):
        if not os.path.exists(path):
            return
        try:
            state = torch.load(path, map_location="cpu")
            with torch.no_grad():
                self.log_std.copy_(state["log_std"])
            self.optimizer.load_state_dict(state["optimizer"])
            logger.info(f"Restored trainer state (log_std={self.log_std.detach().numpy().round(3).tolist()}).")
        except Exception as e:
            logger.warning(f"Could not restore trainer state ({e}); continuing with fresh exploration noise.")

    def reset_direction_head(self):
        """
        Neutral restart of the LONG / SHORT output only (actor row 0): the market understanding (LSTM),
        the critic forecasts and the confidence / size / offset outputs are kept. Used when training
        starts from a brain that learned direction from the account slots (saturated at +1 = LONG).
        """
        fc = self.model.actor_fc
        with torch.no_grad():
            fc.weight[0].normal_(0.0, 1e-3)
            fc.bias[0] = 0.0
            self.log_std[0] = DIR_LOG_STD_MAX
            for p in (fc.weight, fc.bias):
                st = self.optimizer.state.get(p) or {}
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in st:
                        st[k][0].zero_()
        logger.info("Direction output reset to neutral (old LONG habit removed); exploration noise 0.30.")

    def fit_feature_stats(self, n_experiences: int = 400) -> bool:
        """
        Fits per-slot mean / std on training frames and stores them in the model (they ship with the
        checkpoint). Only ACTIVE slots (core.schemas.state_layout) that actually vary in the data are
        enabled; RESERVED / DEPRECATED / dead slots are masked to 0 so they cannot add noise.
        Fit once, then keep fixed: the weights learn on top of these statistics.
        """
        pool = list(self.buffer.cache)
        if len(pool) < 30:
            logger.warning(f"Feature stats: only {len(pool)} training experiences available; keeping current statistics.")
            return False
        import random as _random
        sample = _random.sample(pool, min(n_experiences, len(pool)))
        states, _, _, _, _ = self.buffer.build_tensors(sample, seq_len=300)
        if len(states) == 0:
            return False
        frames = normalize_state_tensor(states).reshape(-1, STATE_DIM)
        frames = frames[torch.isfinite(frames).all(dim=1)]
        mean = frames.mean(dim=0)
        std = frames.std(dim=0, unbiased=False)
        mask = torch.zeros(STATE_DIM)
        mask[list(ACTIVE_SLOTS)] = 1.0
        mask = mask * (std > 1e-6).float()
        mask[list(BRAIN_EXCLUDED_SLOTS)] = 0.0  # account / position state: risk rules only, never the brain
        ts = [float(getattr(e, "timestamp", 0) or 0) for e in sample]
        live_share = sum((t / 1000.0 if t > 1e11 else t) >= MACRO_LIVE_SINCE for t in ts) / max(1, len(ts))
        macro_on = live_share >= MACRO_MIN_LIVE_SHARE
        if not macro_on:
            mask[list(MACRO_PENDING_SLOTS)] = 0.0
        self.model.set_feature_stats(mean, std, mask)
        dead = [i for i in ACTIVE_SLOTS if std[i] <= 1e-6 and i not in BRAIN_EXCLUDED_SLOTS]
        logger.info(
            f"Feature stats: macro momentum inputs {'ON' if macro_on else 'OFF'} "
            f"({live_share:.0%} of the sample has live macro data, need {MACRO_MIN_LIVE_SHARE:.0%})."
        )
        logger.info(
            f"Feature stats fitted on {frames.shape[0]} frames: {int(mask.sum().item())} slots enabled, "
            f"{len(dead)} ACTIVE slots carry no data yet (masked until refit): {dead}"
        )
        return True

    async def _publish_metrics(self, payload: dict):
        if os.getenv("LOCAL_MODE") == "True":
            return  # In local mode, metrics are saved directly to logs and tournament history
        try:
            if not redis_manager.redis:
                await redis_manager.connect()
            if redis_manager.redis:
                await redis_manager.redis.publish("training:metrics", json.dumps(payload))
        except Exception:
            pass # Suppress background metrics errors if Redis is down

    async def _publish_model_update(self):
        if os.getenv("LOCAL_MODE") == "True":
            return
        try:
            if not redis_manager.redis:
                await redis_manager.connect()
            if redis_manager.redis:
                payload = {"timestamp": time.time(), "event": "model_saved"}
                await redis_manager.redis.publish("training:model_update", json.dumps(payload))
        except Exception:
            pass

    def _score_policy(self, model: nn.Module, states: torch.Tensor, rewards: torch.Tensor) -> dict:
        """Deterministic (live-like) holdout scoring of one model."""
        model.eval()
        if not bool(getattr(model, "feat_fitted", False)):
            # Legacy (unfitted) models get 0 in the October 2026 inputs, exactly as the live bot feeds them
            states = states.clone()
            states[..., list(ADDED_2026_10_SLOTS)] = 0.0
        with torch.no_grad():
            acts, vals = model(states)
        long_ret, short_ret = self.trade_outcomes(rewards)
        market_ret = rewards[:, :3]
        dir_ = acts[:, 0:1]
        pos = self.position_from_action(dir_)
        pnl = self.action_pnl(dir_, long_ret, short_ret)
        n_trades = int((pos != 0).sum().item())
        wins = int(((pnl > 0) & (pos != 0)).sum().item())
        # Critic skill per horizon vs the naive "price won't move" forecast (1.0 = perfect, 0 = no skill, < 0 = harmful)
        mse = ((vals - market_ret) ** 2).mean(dim=0)
        naive = (market_ret ** 2).mean(dim=0).clamp_min(1e-8)
        skill = 1.0 - mse / naive
        return {
            "acts": acts, "vals": vals, "pnl": pnl, "pos": pos,
            "sim_pnl": float(pnl.mean().item()),               # mean net % per decision (0 = staying flat)
            "total_pnl": float(pnl.sum().item()),
            "n_trades": n_trades,
            "trade_rate": n_trades / max(1, len(pos)),
            "win_rate": wins / n_trades if n_trades else 0.0,
            "long_share": float((pos > 0).sum().item()) / n_trades if n_trades else 0.0,
            "critic_mse": float(mse.mean().item()),
            "critic_skill": [float(x) for x in skill.tolist()],
        }

    @staticmethod
    def _monotone_rates(rates: List[float], weights: List[int]) -> List[float]:
        """Pool-adjacent-violators: closest non-decreasing sequence (weighted)."""
        blocks = [[r, w, 1] for r, w in zip(rates, weights)]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                a, b = blocks[i], blocks[i + 1]
                w = a[1] + b[1]
                blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / max(w, 1), w, a[2] + b[2]]
                del blocks[i + 1]
                i = max(i - 1, 0)
            else:
                i += 1
        out = []
        for r, _, n in blocks:
            out.extend([r] * n)
        return out

    def derive_trade_policy(self, acts: torch.Tensor, pnl: torch.Tensor, pos: torch.Tensor, symbols: List[str]) -> dict:
        """
        Live trading policy learned on the holdout (shipped in version.json with the model):

        1. Honest confidence: the raw confidence head is mapped to the win rate it ACTUALLY achieved on
           unseen data (binned, forced monotone). The live bot logs / sizes with the calibrated value.
        2. Confidence threshold: the raw confidence above which holdout trades made the most money
           after costs (at least MIN_POLICY_TRADES trades), replacing the hand-set session hurdles.
        3. Per-coin filter: coins whose trades above that threshold lost money on the holdout (with at
           least MIN_COIN_TRADES trades of evidence) are blocked for this model.

        Selection only RESTRICTS trading; the approval decision itself is made on the unfiltered policy.
        """
        MIN_POLICY_TRADES, MIN_COIN_TRADES, N_BINS = 30, 20, 6
        conf = torch.clamp((acts[:, 1] + 1.0) / 2.0, 0.05, 0.99)   # same mapping as the live bot
        traded = (pos[:, 0] != 0)
        conf_t = conf[traded].tolist()
        pnl_t = pnl[traded, 0].tolist()
        sym_t = [s for s, k in zip(symbols, traded.tolist()) if k]
        policy = {"version": 1, "min_confidence": None, "calibration": None, "blocked_symbols": [], "coin_stats": {}}
        if len(conf_t) < MIN_POLICY_TRADES:
            policy["note"] = f"only {len(conf_t)} holdout trades: no filters derived"
            return policy

        order = sorted(range(len(conf_t)), key=lambda i: conf_t[i])
        # 1. Calibration bins (equal-count)
        size = max(1, len(order) // N_BINS)
        raw_pts, rates, weights = [], [], []
        for b in range(0, len(order), size):
            idx = order[b:b + size]
            if len(idx) < max(5, size // 2) and raw_pts:
                weights[-1] += len(idx)  # fold a small tail into the last bin
                continue
            raw_pts.append(sum(conf_t[i] for i in idx) / len(idx))
            rates.append(sum(1 for i in idx if pnl_t[i] > 0) / len(idx))
            weights.append(len(idx))
        policy["calibration"] = {"raw": [round(x, 4) for x in raw_pts],
                                 "win_rate": [round(x, 4) for x in self._monotone_rates(rates, weights)]}

        # 2. Best confidence threshold by total holdout PnL
        best_t, best_total = conf_t[order[0]], sum(pnl_t)
        for k in range(len(order) - MIN_POLICY_TRADES + 1):
            total = sum(pnl_t[i] for i in order[k:])
            if total > best_total:
                best_t, best_total = conf_t[order[k]], total
        policy["min_confidence"] = round(float(best_t), 4)

        # 3. Per-coin filter on the trades the live bot would take
        kept = [i for i in order if conf_t[i] >= best_t]
        stats = {}
        for i in kept:
            st = stats.setdefault(sym_t[i], {"n": 0, "pnl": 0.0, "wins": 0})
            st["n"] += 1
            st["pnl"] += pnl_t[i]
            st["wins"] += pnl_t[i] > 0
        policy["coin_stats"] = {s: {"n": v["n"], "mean_pnl": round(v["pnl"] / v["n"], 5), "win_rate": round(v["wins"] / v["n"], 3)}
                                for s, v in stats.items()}
        policy["blocked_symbols"] = sorted(s for s, v in stats.items() if v["n"] >= MIN_COIN_TRADES and v["pnl"] <= 0.0)
        allowed = [i for i in kept if sym_t[i] not in policy["blocked_symbols"]]
        policy["filtered_trades"] = len(allowed)
        policy["filtered_pnl_per_trade"] = round(sum(pnl_t[i] for i in allowed) / len(allowed), 5) if allowed else 0.0
        return policy

    def evaluate_candidate(self, sample_size: int = 1000) -> dict:
        """
        Champion vs Challenger tournament on the CHRONOLOGICAL HOLDOUT (newest data, never trained on).

        Both models act deterministically (as live), with the live entry rule (|a| > 0.2), scored by the
        realized market move minus round-trip costs. A challenger is approved only if it:
          1. does not lose money vs staying flat (strictly positive, unless the champion itself is losing),
          2. beats the champion's holdout PnL and composite score by a margin,
          3. has a critic that is not worse than the champion's (the bot gates trades on it).
        """
        # 1. Weights sanity check
        if any(torch.isnan(p).any() for p in self.model.parameters()):
            logger.error("Candidate evaluation FAILED: Challenger model weights contain NaN.")
            return {"approved": False, "reason": "NAN_WEIGHTS"}

        min_eval_samples = 30
        batch = self.buffer.sample_holdout(sample_size)
        if len(batch) < min_eval_samples:
            logger.warning(f"Candidate evaluation REJECTED: Holdout has insufficient samples ({len(batch)} < {min_eval_samples}).")
            return {"approved": False, "reason": "INSUFFICIENT_SAMPLE_SIZE", "critic_loss": 0.0, "entropy": 0.0}

        try:
            states, actions, rewards, next_states, dones = self.buffer.build_tensors(batch, seq_len=300)
            row_symbols = list(getattr(self.buffer, "last_symbols", []) or [])
            if len(row_symbols) != len(states):
                row_symbols = [""] * len(states)
            if len(states) < min_eval_samples:
                logger.warning(f"Candidate evaluation REJECTED: Insufficient valid sequence tensors ({len(states)} < {min_eval_samples}).")
                return {"approved": False, "reason": "INSUFFICIENT_TENSORS", "critic_loss": 0.0, "entropy": 0.0}

            states = normalize_state_tensor(states)
            states = torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)
            rewards = torch.clamp(torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0), min=-15.0, max=15.0)
            del actions, next_states, dones

            chall = self._score_policy(self.model, states, rewards)
            if torch.isnan(chall["acts"]).any() or torch.isnan(chall["vals"]).any():
                logger.error("Candidate evaluation FAILED: Challenger output contains NaN.")
                return {"approved": False, "reason": "NAN_OUTPUTS"}
            challenger_entropy = float(chall["acts"][:, 0].std().item())

            # A one-sided policy that trades a lot is almost certainly fitting the holdout's regime, not an edge
            if chall["trade_rate"] > 0.05 and (chall["long_share"] > 0.95 or chall["long_share"] < 0.05):
                logger.warning(f"Candidate evaluation FAILED: Action monoculture (trade rate {chall['trade_rate']:.1%}, long share {chall['long_share']:.1%}).")
                return {"approved": False, "reason": "ACTION_MONOCULTURE", "long_share": chall["long_share"], "trade_rate": chall["trade_rate"]}

            fee_hurdle = settings.taker_fee_pct * 2.0 * 100.0
            slippage_hurdle = ROUND_TRIP_COST_PCT - fee_hurdle

            # 2. Benchmark against Production Champion
            champion_path = "models/production/model_v1.pt"
            if not os.path.exists(champion_path):
                logger.error("Candidate evaluation FAILED: Production champion checkpoint (models/production/model_v1.pt) is missing. Failing closed to prevent unbenchmarked promotion.")
                return {
                    "approved": False,
                    "reason": "MISSING_CHAMPION_CHECKPOINT",
                    "challenger_score": None,
                    "champion_score": None,
                    "critic_loss": chall["critic_mse"],
                    "entropy": challenger_entropy,
                    "sample_size": len(states)
                }

            champion_model = SingleSymbolActorCritic(input_dim=200)
            try:
                champion_state = torch.load(champion_path, map_location="cpu")
                champion_model.load_state_dict(champion_state)
                champ = self._score_policy(champion_model, states, rewards)
                del champion_model
            except Exception as champ_e:
                logger.error(f"Candidate evaluation FAILED: Corrupt, incompatible, or unloadable champion model ({champ_e}). Failing closed without fabricating benchmark scores.")
                return {
                    "approved": False,
                    "reason": f"CHAMPION_LOAD_FAILURE: {champ_e}",
                    "challenger_score": None,
                    "champion_score": None,
                    "critic_loss": chall["critic_mse"],
                    "entropy": challenger_entropy,
                    "sample_size": len(states)
                }

            eval_sample_count = len(states)
            del states, rewards

            # Per-coin filter + honest confidence, derived on the same unseen holdout (restrict-only)
            trade_policy = self.derive_trade_policy(chall["acts"], chall["pnl"], chall["pos"], row_symbols)

            # Composite score: holdout trading PnL dominates; critic skill (1h horizon) is a tie-breaker.
            challenger_score = chall["sim_pnl"] + 0.01 * chall["critic_skill"][1]
            champion_score = champ["sim_pnl"] + 0.01 * champ["critic_skill"][1]

            SUPERIORITY_MARGIN = 0.0005  # % per decision
            score_beaten = challenger_score >= (champion_score + SUPERIORITY_MARGIN)
            pnl_preserved = chall["sim_pnl"] >= champ["sim_pnl"]
            beats_flat = chall["sim_pnl"] > 0.0 or (chall["sim_pnl"] >= 0.0 and champ["sim_pnl"] < 0.0)
            critic_not_degraded = chall["critic_skill"][1] >= min(champ["critic_skill"][1], 0.0) - 0.01
            is_superior = score_beaten and pnl_preserved and beats_flat and critic_not_degraded

            eval_timestamp = time.time()

            # Compute cryptographic hash of the current champion checkpoint
            import hashlib
            sha256 = hashlib.sha256()
            with open(champion_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)
            champ_hash = sha256.hexdigest()

            # Read champion version from version.json
            champ_version = "v1"
            version_file = "models/production/version.json"
            if os.path.exists(version_file):
                try:
                    with open(version_file, "r") as f:
                        v_data = json.load(f)
                        champ_version = v_data.get("version", "v1")
                except Exception:
                    pass

            metrics = {
                "challenger_score": challenger_score,
                "champion_score": champion_score,
                "challenger_critic_mse": chall["critic_mse"],
                "champion_critic_mse": champ["critic_mse"],
                "challenger_sim_pnl": chall["sim_pnl"],
                "champion_sim_pnl": champ["sim_pnl"],
                "challenger_total_pnl": chall["total_pnl"],
                "champion_total_pnl": champ["total_pnl"],
                "challenger_trades": chall["n_trades"],
                "champion_trades": champ["n_trades"],
                "challenger_win_rate": chall["win_rate"],
                "champion_win_rate": champ["win_rate"],
                "challenger_critic_skill": chall["critic_skill"],
                "champion_critic_skill": champ["critic_skill"],
                "critic_loss": chall["critic_mse"],
                "entropy": challenger_entropy,
                "sample_size": eval_sample_count,
                "eval_timestamp": eval_timestamp,
                "eval_type": "chronological_tournament",
                "data_split": "chronological_holdout",
                "holdout_start_ts": float(min(self.buffer.holdout_start_ts, 2**62)),
                "fee_hurdle": fee_hurdle,
                "slippage_hurdle": slippage_hurdle,
                "champion_checkpoint_hash": champ_hash,
                "champion_version": champ_version,
                "evaluator_version": "v3_holdout_counterfactual",
                "label_type": "path_aware_trade_outcome",
                "trade_policy": trade_policy,
            }

            summary = (
                f"Holdout PnL/decision: {chall['sim_pnl']:+.4f}% vs Champ {champ['sim_pnl']:+.4f}% | "
                f"Trades: {chall['n_trades']} (win {chall['win_rate']:.0%}) vs {champ['n_trades']} (win {champ['win_rate']:.0%}) | "
                f"Critic skill 5m/1h/4h: {[round(x, 3) for x in chall['critic_skill']]} vs {[round(x, 3) for x in champ['critic_skill']]} | "
                f"n={eval_sample_count} | Policy: min conf {trade_policy.get('min_confidence')}, "
                f"blocked {trade_policy.get('blocked_symbols')}, filtered PnL/trade {trade_policy.get('filtered_pnl_per_trade')}"
            )
            if is_superior:
                logger.info(f"🏆 TOURNAMENT WINNER: Challenger is SUPERIOR on holdout! {summary}")
                return {"approved": True, "reason": "SUPERIOR_TO_CHAMPION", "metrics": metrics, **metrics}
            if not (score_beaten and pnl_preserved):
                reason = "INFERIOR_TO_CHAMPION"
            elif not beats_flat:
                reason = "NO_EDGE_AFTER_COSTS"
            else:
                reason = "CRITIC_DEGRADED"
            logger.warning(f"⛔ TOURNAMENT DEFEAT ({reason}): {summary}. REJECTING PUSH.")
            return {"approved": False, "reason": reason, "metrics": metrics, **metrics}
        except Exception as e:
            logger.exception(f"Error during candidate tournament evaluation: {e!r}")
            return {"approved": False, "reason": f"EXCEPTION_{e!r}"}

async def run_training_loop():
    set_log_file("logs/ai_trainer.log")
    logger.info("Initializing PPO Training Engine...")
    # Dashboard heartbeat from launch on (loading the replay cache alone can take ~10 minutes)
    trainer_status.set_phase("starting")
    trainer_status.start_heartbeat()
    
    # We need to initialize the model first
    from core.config.settings import settings
    import torch
    # Restrict PyTorch to 1 CPU thread so it doesn't starve the Live Trading Bot on shared VPS
    torch.set_num_threads(1)
    
    # The ReplayBuffer currently returns 41-dim state vectors (single symbol + portfolio + position + macro)
    model = SingleSymbolActorCritic(input_dim=200)
    registry = ModelRegistry()
    
    try:
        model = registry.load_model(model)
        logger.info("Successfully loaded existing model weights.")
    except Exception as e:
        logger.warning(f"Could not load existing model, starting fresh: {e}")
    
    # Thermal Management & CPU Core Restrictions
    local_mode = os.getenv("LOCAL_MODE") == "True"
    
    if local_mode:
        total_cores = os.cpu_count() or 4
        # DYNAMIC THERMAL BALANCING: Do NOT lock strictly to Core 0 (Mask=3 locks both threads to a single physical core, causing severe localized hotspots!).
        # Instead, allow Windows scheduler to dynamically rotate threads across ALL physical cores.
        # Combined with torch.set_num_threads(2), this keeps total CPU load at ~25% while rotating heat across the entire die so no single core overheats.
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            all_cores_mask = (1 << total_cores) - 1
            ctypes.windll.kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            ctypes.windll.kernel32.SetProcessAffinityMask(handle, all_cores_mask)
            # Set Process Priority to BELOW_NORMAL (0x00004000)
            ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)
            logger.info(f"❄️ DYNAMIC THERMAL BALANCING ACTIVE: Workload rotated across all {total_cores} cores & BELOW_NORMAL priority.")
        except Exception as aff_e:
            logger.warning(f"Could not apply thermal settings: {aff_e}")

        # PyTorch thread cap: strictly 1 worker thread (Zero thread-lock overhead, ~12.5% single-core load)
        safe_threads = 1
        torch.set_num_threads(safe_threads)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        limit = 5000
        epochs = 2  # Balanced PPO passes per batch (cuts continuous computation time)
        sleep_time = 0.5 # 4.0s cooling breather after every step for complete heat dissipation
        logger.info(f"❄️ MAXIMUM THERMAL PROTECTION: PyTorch capped at {safe_threads} thread, {epochs} epochs, with {sleep_time}s heatpipe breathers.")
    else:
        logger.info("🐢 VPS MODE DETECTED: Running in STARVATION MODE (1 CPU Core, 300 RAM limit)")
        torch.set_num_threads(1)
        limit = 500
        epochs = 2
        sleep_time = 30.0
    
    # 1. AUTO-BACKUP: Save the current good brain before we risk overwriting it
    if local_mode:
        import shutil
        from datetime import datetime
        model_path = "models/production/model_v1.pt"
        if os.path.exists(model_path):
            os.makedirs("models/checkpoints", exist_ok=True)
            backup_name = f"models/checkpoints/model_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            try:
                shutil.copy2(model_path, backup_name)
                logger.info(f"💾 Created pre-training backup: {backup_name}")
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")

    # 2. PRIORITY #1: Load Cache into RAM FIRST (Unobstructed, zero DB lock contention)
    trainer_status.set_phase("loading training data")
    trainer = PPOTrainer(model=model, limit=limit)
    trainer.load_trainer_state()
    # Per-slot input scaling: fit once for models that predate it, or on request when new inputs come online
    import sys as _sys
    # A brain that still sees the account slots learned LONG / SHORT from them: restart that output neutral
    legacy_direction = (not bool(model.feat_fitted)) or bool(model.feat_mask[list(BRAIN_EXCLUDED_SLOTS)].any())
    if not bool(model.feat_fitted) or "--refit-feature-stats" in _sys.argv:
        trainer_status.set_phase("fitting input scaling")
        trainer.fit_feature_stats()
    elif bool(model.feat_mask[list(BRAIN_EXCLUDED_SLOTS)].any()):
        # Fitted before the account / position slots were excluded: switch them off, keep the rest
        with torch.no_grad():
            model.feat_mask[list(BRAIN_EXCLUDED_SLOTS)] = 0.0
        logger.info("Feature mask: account / position slots 0-19 switched off for the brain.")
    if legacy_direction:
        trainer.reset_direction_head()
    logger.info("Starting continuous PPO training on historical/live data...")
    trainer_status.set_phase(
        "training",
        step_fn=lambda: getattr(trainer, "step_count", 0),
        extra_fn=lambda: {"train_rows": len(trainer.buffer.cache), "holdout_rows": len(trainer.buffer.holdout_cache)},
    )

    # 3. PRIORITY #2: Start DB Auto-Syncer & Golden Extractor in background thread AFTER cache is in RAM
    if local_mode:
        try:
            import threading
            import time
            import sys
            import subprocess
            from scripts.sync_db import sync_experiences
            from apps.experience.reward_calculator import RewardCalculator, golden_table
            
            def background_syncer():
                logger.info("Background DB Syncer, Reward Grader & Golden Extractor started!")
                last_golden_time = 0.0
                grader = RewardCalculator()
                try:
                    golden = golden_table()
                except Exception as e:
                    golden = None
                    logger.warning(f"Golden table unavailable for grading: {e}")
                while True:
                    try:
                        sync_experiences()
                    except Exception as e:
                        logger.error(f"Sync failed: {e}")

                    try:
                        # Grade synced rows locally with TRUE market returns (public klines); the trainer
                        # only learns from rows graded this way.
                        graded = grader.process_batch()
                        if graded:
                            logger.info(f"Reward grader: graded {graded} experiences with true market returns.")
                        # Golden rows are copies; any copied before grading must be graded in place too.
                        graded_golden = grader.process_batch(table=golden) if golden is not None else 0
                        if graded_golden:
                            logger.info(f"Reward grader: graded {graded_golden} golden experiences.")
                        # Path-aware labels: what a LONG / SHORT opened at each record really earned
                        # under the live exits (stop before recovery = loss). Backfills the history too.
                        labelled = grader.process_outcomes()
                        if golden is not None:
                            labelled += grader.process_outcomes(table=golden, src="gold")
                        if labelled:
                            logger.info(f"Reward grader: added {labelled} real trade outcomes.")
                    except Exception as e:
                        logger.error(f"Reward grading failed: {e}")
                        
                    try:
                        # Extract golden trades every 5 minutes (300s) to keep DB fast
                        if time.time() - last_golden_time > 300:
                            last_golden_time = time.time()
                            result = subprocess.run([sys.executable, "scratch/extract_golden_batch.py"], capture_output=True, text=True)
                            if "New Golden Trades Extracted: " in result.stdout:
                                count_str = result.stdout.split("New Golden Trades Extracted: ")[1].split("\n")[0].strip()
                                if count_str != "0":
                                    logger.info(f"⛏️ GOLDEN EXTRACTOR: Successfully secured {count_str} highly profitable trades into permanent memory!")
                    except Exception as e:
                        pass
                        
                    time.sleep(30) # Check for new trades every 30 seconds

            threading.Thread(target=background_syncer, daemon=True).start()
        except Exception as e:
            logger.error(f"Could not start background syncer: {e}")
    
    step = 0
    # Initialize push_count from tournament_history.json so version tags increment seamlessly
    push_count = 0
    try:
        if os.path.exists("models/production/tournament_history.json"):
            with open("models/production/tournament_history.json", "r") as f:
                hist = json.load(f)
                for entry in hist:
                    ver = entry.get("version", "")
                    if "v1_push_" in ver:
                        try:
                            num = int(ver.replace("v1_push_", ""))
                            if num > push_count:
                                push_count = num
                        except ValueError:
                            pass
    except Exception:
        pass

    last_push_time = 0.0
    try:
        while True:
            # CPU FIX: Cut epochs to 2 to halve the math computation time.
            success = trainer.train_step(batch_size=64, epochs=epochs)
            
            if not success:
                # CRITICAL FIX: If train_step fails (usually because cache is empty),
                # we MUST try to reload the cache from the DB. Otherwise, we hit an infinite sleep loop.
                logger.warning("Train step failed (likely empty cache). Forcing cache reload...")
                trainer.buffer.load_cache_from_db(limit=limit)
                await asyncio.sleep(2.0)
                continue
                
            # SAFETY BREATHER: 5 second rest to prevent host from suspending the server
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            step += 1
            trainer.step_count = step
            if step % 100 == 0:
                logger.info(f"Completed {step} training steps. Evaluating candidate brain before deployment...")
                
                # Auto-save candidate weights to disk so training progress is never lost even if rejected
                cand_dir = "models/candidates"
                os.makedirs(cand_dir, exist_ok=True)
                cand_step_path = os.path.join(cand_dir, f"candidate_step_{step}.pt")
                torch.save(model.state_dict(), cand_step_path)
                torch.save(model.state_dict(), os.path.join(cand_dir, "candidate_latest.pt"))
                trainer.save_trainer_state()
                logger.info(f"💾 Saved candidate checkpoint: {cand_step_path}")
                
                trainer_status.set_phase("running tournament")
                eval_result = trainer.evaluate_candidate(sample_size=1000)
                trainer_status.set_phase("training")
                
                if not eval_result.get("approved", False):
                    logger.warning(f"Candidate brain REJECTED by evaluation gate: {eval_result.get('reason')}. Continuing training without resetting weights (Saved at {cand_step_path})...")
                    registry.record_tournament_result(eval_result, version_tag=f"rejected_step_{step}")
                else:
                    try:
                        push_count += 1
                        version_tag = f"v1_push_{push_count}"
                        cand_path = os.path.join(cand_dir, f"{version_tag}.pt")
                        torch.save(model.state_dict(), cand_path)
                        cand_hash = registry.compute_file_sha256(cand_path)
                        eval_result["candidate_checkpoint_hash"] = cand_hash
                        if "metrics" in eval_result and isinstance(eval_result["metrics"], dict):
                            eval_result["metrics"]["candidate_checkpoint_hash"] = cand_hash
                        promoted = registry.promote_candidate(cand_path, version_tag=version_tag, metrics=eval_result)
                        if not promoted:
                            logger.error(f"Promotion gate rejected candidate {version_tag}.")
                            continue
                        
                        if local_mode and (time.time() - last_push_time > 300):
                            logger.info(f"🚀 Pushing evaluated Brain ({version_tag}) to VPS in the background ({push_count}/150)...")
                            last_push_time = time.time()
                            import subprocess
                            subprocess.Popen(
                                ["scp", "models/production/model_v1.pt", "root@72.62.255.1:/var/www/trading-bot/ai-trading-platform/models/production/model_v1.pt"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            subprocess.Popen(
                                ["scp", "models/production/version.json", "root@72.62.255.1:/var/www/trading-bot/ai-trading-platform/models/production/version.json"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            subprocess.Popen(
                                ["scp", "models/production/tournament_history.json", "root@72.62.255.1:/var/www/trading-bot/ai-trading-platform/models/production/tournament_history.json"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            
                            # Auto-Shutdown after 150 successful pushes
                            if push_count >= 150:
                                logger.info("✅ Successfully completed 150 Brain pushes. Auto-shutting down Master Trainer!")
                                import sys
                                sys.exit(0)
                        
                        # Schedule model update broadcast
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(trainer._publish_model_update())
                        except RuntimeError:
                            asyncio.run(trainer._publish_model_update())
                    except Exception as e:
                        logger.error(f"Failed to save model: {e}")
                    
                # Refresh cache AFTER saving, so the GC freeze doesn't block the actual deployment
                logger.info("Refreshing RAM cache from DB (this may take a minute)...")
                trainer_status.set_phase("reloading training data")
                trainer.buffer.load_cache_from_db(limit=limit)
                trainer_status.set_phase("training")
                
                # Auto-delete data older than 30 days to save VPS disk space
                if step % 1000 == 0:
                    trainer.buffer.cleanup_old_data(days=30)
                    
    except asyncio.CancelledError:
        logger.info("PPO Training Engine shutting down...")
        
if __name__ == "__main__":
    try:
        asyncio.run(run_training_loop())
    except KeyboardInterrupt:
        pass
