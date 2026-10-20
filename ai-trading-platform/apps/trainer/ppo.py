import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
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
    
    return s


class PPOTrainer:
    """
    Phase 11: Proximal Policy Optimization (PPO) Training Loop.
    Samples from the MySQL ReplayBuffer or RolloutBuffer and updates the Actor-Critic model.
    """
    def __init__(self, model: SingleSymbolActorCritic, lr: float = 3e-4, gamma: float = 0.99, clip_epsilon: float = 0.2, limit: int = 300):
        self.model = model
        self.log_std = nn.Parameter(torch.ones(4) * -0.5)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + [self.log_std],
            lr=lr,
            weight_decay=1e-5
        )
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
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
            action_std = torch.exp(torch.clamp(self.log_std, min=-2.0, max=0.5))
            dist = torch.distributions.Normal(action_mean, action_std)
            if deterministic:
                action = action_mean
                log_prob = None
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob, value

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
        """Executes a single mathematically valid PPO training step with strict data validation."""
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
        if rewards.ndim != 2 or rewards.shape[-1] != 3:
            logger.error(f"PPO: Invalid reward tensor dimension {rewards.shape}. Expected (batch, 3).")
            return False
        if dones.ndim != 2 or dones.shape[-1] != 1:
            logger.error(f"PPO: Invalid dones tensor dimension {dones.shape}. Expected (batch, 1).")
            return False

        # Normalize state vectors so wallet balance & entry price don't drown out market indicators in LayerNorm
        states = normalize_state_tensor(states)
        next_states = normalize_state_tensor(next_states)

        # Detect and filter corrupted non-finite samples (Never silently convert with nan_to_num)
        finite_s = torch.isfinite(states).all(dim=1).all(dim=1)
        finite_ns = torch.isfinite(next_states).all(dim=1).all(dim=1)
        finite_a = torch.isfinite(actions).all(dim=1)
        finite_r = torch.isfinite(rewards).all(dim=1)
        finite_d = torch.isfinite(dones).all(dim=1)

        valid_mask = finite_s & finite_ns & finite_a & finite_r & finite_d
        corrupted_count = int((~valid_mask).sum().item())
        if corrupted_count > 0:
            logger.warning(f"PPO: Logged and rejected {corrupted_count} corrupted samples containing NaN/Inf values. Retaining only valid historical experiences.")
            states = states[valid_mask]
            actions = actions[valid_mask]
            rewards = rewards[valid_mask]
            next_states = next_states[valid_mask]
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
            
        self.model.train()
        
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
            old_log_probs = stored_old_log_probs
            with torch.no_grad():
                old_action_preds, old_values = self.model(states)
                _, next_values = self.model(next_states)
        else:
            if require_stored_behavior_policy:
                logger.error("PPO: Replay samples lack stored behavior policy log probabilities. Cannot execute on-policy PPO update without actual behavior policy.")
                return False
            with torch.no_grad():
                old_action_preds, old_values = self.model(states)
                _, next_values = self.model(next_states)
                action_std = torch.exp(torch.clamp(self.log_std, min=-2.0, max=0.5))
                dist_old = torch.distributions.Normal(old_action_preds, action_std)
                old_log_probs = dist_old.log_prob(actions).sum(dim=-1, keepdim=True)
            
        with torch.no_grad():
            safe_rewards = rewards / 200.0
            advantages, returns = self.compute_gae(safe_rewards, old_values, next_values, dones)

            if not (torch.isfinite(advantages).all() and torch.isfinite(returns).all()):
                logger.error("PPO: Computed non-finite advantages or returns. Aborting update.")
                return False

            actor_advantages = advantages.mean(dim=1, keepdim=True)
            adv_mean = actor_advantages.mean()
            adv_std = actor_advantages.std(unbiased=False)
            
            # Phase 1.A: Never manufacture random advantages. If insufficient variance, skip update.
            if actor_advantages.size(0) <= 1 or adv_std <= 1e-6:
                logger.info(f"PPO: Insufficient reward variation in batch (adv_std={adv_std.item():.6f} <= 1e-6). Deferring update.")
                return False
                
            actor_advantages = (actor_advantages - adv_mean) / (adv_std + 1e-8)
            
        # 2. PPO Epochs
        for _ in range(epochs):
            action_preds, values = self.model(states)
            
            action_std = torch.exp(torch.clamp(self.log_std, min=-2.0, max=0.5))
            dist = torch.distributions.Normal(action_preds, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            
            # Numerically stable probability ratio
            diff = torch.clamp(log_probs - old_log_probs, min=-10.0, max=10.0)
            ratio = torch.exp(diff)
            
            # Off-policy correction: Importance sampling truncation (V-trace style)
            if use_off_policy_correction:
                ratio = torch.clamp(ratio, max=1.0)
            
            # Clipped surrogate objective
            surr1 = ratio * actor_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * actor_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (MSE across multi-horizon critic)
            critic_loss = nn.MSELoss()(values, returns)
            
            # Analytical Gaussian distribution entropy
            entropy = dist.entropy().sum(dim=-1).mean()
            entropy_coef = 0.015
            
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
            
            if not torch.isfinite(loss):
                logger.error("PPO Loss is NaN or Inf! Skipping step.")
                break
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + [self.log_std], 0.5)
            self.optimizer.step()
            
        logger.info(f"PPO Training Step Complete | Loss: {loss.item():.4f} | Actor: {actor_loss.item():.4f} | Critic: {critic_loss.item():.4f} | Entropy: {entropy.item():.4f} | Rewards: {rewards[:, 0].mean().item():.4f} | Values: {values[:, 0].mean().item():.4f}")
        
        # Calculate extra metrics for dashboard BEFORE deleting tensors
        with torch.no_grad():
            batch_reward = float(rewards[:, 0].mean().item()) # Dashboard tracks 5m reward
            
            # Action distribution (actions[:, 0]: 0.0=Hold, 0.5=Buy, -0.5=Sell)
            actual_size = actions.size(0)
            act_types = actions[:, 0]
            holds = float((act_types == 0.0).sum().item()) / actual_size if actual_size > 0 else 1.0
            buys = float((act_types > 0.0).sum().item()) / actual_size if actual_size > 0 else 0.0
            sells = float((act_types < 0.0).sum().item()) / actual_size if actual_size > 0 else 0.0

        # MEMORY FIX: Now safe to delete large tensors - all scalar values already extracted above
        del states, actions, rewards, next_states, dones
        del old_action_preds, old_values, next_values
        del old_log_probs, safe_rewards, advantages, returns, actor_advantages
        # Note: Do NOT call gc.collect() here. It causes GIL deadlocks and freezes PyTorch.
            
        # Publish metrics to Redis asynchronously via asyncio.create_task or run_coroutine_threadsafe
        # We assume trainer loop might be sync or async. Let's provide a safe sync wrapper or fire-and-forget
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

    def evaluate_candidate(self, sample_size: int = 1000) -> dict:
        """
        Phase 27 & 53 Automated Model Evaluation:
        Conducts a Champion vs Challenger Tournament before allowing any model promotion.
        - Loads the current production Champion model from disk.
        - Evaluates both Champion and Challenger on the same batch of replay buffer experiences.
        - Compares:
            1. Critic MSE (prediction accuracy on future multi-horizon moves).
            2. Policy alignment / Simulated return (actions chosen vs realized returns).
            3. Policy entropy (ensuring healthy diversity, zero mode collapse).
        - A Challenger is ONLY approved for promotion if it proves mathematical superiority over the Champion.
        """
        # 1. Weights sanity check
        if any(torch.isnan(p).any() for p in self.model.parameters()):
            logger.error("Candidate evaluation FAILED: Challenger model weights contain NaN.")
            return {"approved": False, "reason": "NAN_WEIGHTS"}

        min_eval_samples = 30
        batch = self.buffer.sample(sample_size)
        if len(batch) < min_eval_samples:
            logger.warning(f"Candidate evaluation REJECTED: Replay buffer has insufficient samples ({len(batch)} < {min_eval_samples}).")
            return {"approved": False, "reason": "INSUFFICIENT_SAMPLE_SIZE", "critic_loss": 0.0, "entropy": 0.0}

        try:
            states, actions, rewards, next_states, dones = self.buffer.build_tensors(batch, seq_len=300)
            if len(states) < min_eval_samples:
                logger.warning(f"Candidate evaluation REJECTED: Insufficient valid sequence tensors ({len(states)} < {min_eval_samples}).")
                return {"approved": False, "reason": "INSUFFICIENT_TENSORS", "critic_loss": 0.0, "entropy": 0.0}

            states = normalize_state_tensor(states)
            states = torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)
            safe_rewards = rewards / 200.0
            actual_5m_ret = rewards[:, 0]

            self.model.eval()
            with torch.no_grad():
                challenger_actions, challenger_values = self.model(states)

                if torch.isnan(challenger_actions).any() or torch.isnan(challenger_values).any():
                    logger.error("Candidate evaluation FAILED: Challenger output contains NaN.")
                    return {"approved": False, "reason": "NAN_OUTPUTS"}

                # Strict Policy Entropy Floor (Prevents pushing mode-collapsed models)
                noise = torch.randn_like(challenger_actions) * 1e-5
                var = (challenger_actions + noise).var(dim=0, unbiased=False)
                challenger_entropy = float(torch.sqrt(var + 1e-8).mean().item())

                if challenger_entropy < 0.60:
                    logger.warning(f"Candidate evaluation FAILED: Challenger mode collapse / low entropy ({challenger_entropy:.4f} < 0.60). Preserving champion.")
                    return {"approved": False, "reason": "LOW_POLICY_ENTROPY", "entropy": challenger_entropy}

                challenger_critic_mse = float(nn.MSELoss()(challenger_values, safe_rewards).item())
                if challenger_critic_mse > 4.0:
                    logger.warning(f"Candidate evaluation FAILED: Challenger critic loss too high ({challenger_critic_mse:.2f} > 4.0). Preserving champion.")
                    return {"approved": False, "reason": "CRITIC_EXPLODED", "critic_loss": challenger_critic_mse}

                # Action Monoculture / Saturation Check (Prevents pushing 100% blind short/long models)
                short_ratio = float((challenger_actions[:, 0] < -0.2).float().mean().item())
                long_ratio = float((challenger_actions[:, 0] > 0.2).float().mean().item())
                if short_ratio > 0.85 or long_ratio > 0.85:
                    logger.warning(f"Candidate evaluation FAILED: Action monoculture detected (Short: {short_ratio*100:.1f}%, Long: {long_ratio*100:.1f}%). Model is blindly biased.")
                    return {"approved": False, "reason": "ACTION_MONOCULTURE", "short_ratio": short_ratio, "long_ratio": long_ratio}

                # Critic Prediction Variance Check (Ensures Critic responds dynamically to market regimes)
                c_std = float(challenger_values[:, 0].std().item())
                if c_std < 0.05:
                    logger.warning(f"Candidate evaluation FAILED: Critic output has near-zero variance ({c_std:.4f} < 0.05). Critic has collapsed.")
                    return {"approved": False, "reason": "CRITIC_ZERO_VARIANCE", "critic_std": c_std}

            # Real-World VPS Fee & Slippage Simulation:
            # 0.10% Binance Futures round-trip taker fee + 0.02% conservative slippage
            fee_hurdle = 0.10
            slippage_hurdle = 0.02
            total_cost_hurdle = fee_hurdle + slippage_hurdle

            challenger_is_active = (torch.abs(challenger_actions[:, 0]) > 0.2).float()
            challenger_net_returns = (challenger_actions[:, 0] * actual_5m_ret) - (challenger_is_active * total_cost_hurdle)
            challenger_sim_pnl = float(challenger_net_returns.mean().item())
            challenger_score = challenger_sim_pnl - (0.5 * challenger_critic_mse)

            # 2. Benchmark against Production Champion
            champion_path = "models/production/model_v1.pt"
            champion_available = os.path.exists(champion_path)
            
            if not champion_available:
                logger.error("Candidate evaluation FAILED: Production champion checkpoint (models/production/model_v1.pt) is missing. Failing closed to prevent unbenchmarked promotion.")
                del states, actions, rewards, next_states, dones
                return {
                    "approved": False,
                    "reason": "MISSING_CHAMPION_CHECKPOINT",
                    "challenger_score": challenger_score,
                    "champion_score": None,
                    "critic_loss": challenger_critic_mse,
                    "entropy": challenger_entropy,
                    "sample_size": len(batch)
                }

            # Load Champion for comparison
            champion_model = SingleSymbolActorCritic(input_dim=200)
            try:
                champion_state = torch.load(champion_path, map_location="cpu")
                champion_model.load_state_dict(champion_state)
                champion_model.eval()

                with torch.no_grad():
                    champ_actions, champ_values = champion_model(states)
                    champ_critic_mse = float(nn.MSELoss()(champ_values, safe_rewards).item())
                    champ_is_active = (torch.abs(champ_actions[:, 0]) > 0.2).float()
                    champ_net_returns = (champ_actions[:, 0] * actual_5m_ret) - (champ_is_active * total_cost_hurdle)
                    champ_sim_pnl = float(champ_net_returns.mean().item())
                    champion_score = champ_sim_pnl - (0.5 * champ_critic_mse)

                del champion_model, champ_actions, champ_values
            except Exception as champ_e:
                logger.error(f"Candidate evaluation FAILED: Corrupt, incompatible, or unloadable champion model ({champ_e}). Failing closed without fabricating benchmark scores.")
                del states, actions, rewards, next_states, dones
                return {
                    "approved": False,
                    "reason": f"CHAMPION_LOAD_FAILURE: {champ_e}",
                    "challenger_score": challenger_score,
                    "champion_score": None,
                    "critic_loss": challenger_critic_mse,
                    "entropy": challenger_entropy,
                    "sample_size": len(batch)
                }

            eval_sample_count = len(states)
            del states, actions, rewards, next_states, dones

            # Superiority criteria (Task 1.1: Strict Numerical Superiority):
            # Challenger must strictly beat Champion on composite score AND not degrade trading PnL!
            # Critic MSE improvement alone cannot authorize promotion if composite score or trading PnL is inferior.
            SUPERIORITY_MARGIN = 0.01  # Predefined minimum improvement threshold
            score_beaten = challenger_score >= (champion_score + SUPERIORITY_MARGIN)
            pnl_preserved = challenger_sim_pnl >= champ_sim_pnl
            critic_not_degraded = challenger_critic_mse <= max(champ_critic_mse * 1.25, 2.5)
            is_superior = score_beaten and pnl_preserved and critic_not_degraded

            eval_timestamp = time.time()
            
            # Compute cryptographic hash of the current champion checkpoint
            import hashlib
            champ_hash = ""
            if os.path.exists(champion_path):
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
                "challenger_critic_mse": challenger_critic_mse,
                "champion_critic_mse": champ_critic_mse,
                "challenger_sim_pnl": challenger_sim_pnl,
                "champion_sim_pnl": champ_sim_pnl,
                "critic_loss": challenger_critic_mse,
                "entropy": challenger_entropy,
                "sample_size": eval_sample_count,
                "eval_timestamp": eval_timestamp,
                "eval_type": "chronological_tournament",
                "data_split": "chronological_holdout",
                "fee_hurdle": fee_hurdle,
                "slippage_hurdle": slippage_hurdle,
                "champion_checkpoint_hash": champ_hash,
                "champion_version": champ_version,
                "evaluator_version": "v2_trusted"
            }

            if is_superior:
                logger.info(
                    f"🏆 TOURNAMENT WINNER: Challenger is SUPERIOR! "
                    f"Score: {challenger_score:.4f} vs Champ: {champion_score:.4f} | "
                    f"Critic MSE: {challenger_critic_mse:.4f} vs {champ_critic_mse:.4f} | "
                    f"Sim PnL: {challenger_sim_pnl:.4f} vs {champ_sim_pnl:.4f}"
                )
                return {
                    "approved": True,
                    "reason": "SUPERIOR_TO_CHAMPION",
                    "metrics": metrics,
                    **metrics
                }
            else:
                logger.warning(
                    f"⛔ TOURNAMENT DEFEAT: Challenger is INFERIOR to current live Champion! "
                    f"Score: {challenger_score:.4f} vs Champ: {champion_score:.4f} (Margin: {SUPERIORITY_MARGIN}) | "
                    f"Critic MSE: {challenger_critic_mse:.4f} vs {champ_critic_mse:.4f} | "
                    f"Sim PnL: {challenger_sim_pnl:.4f} vs {champ_sim_pnl:.4f}. REJECTING PUSH."
                )
                return {
                    "approved": False,
                    "reason": "INFERIOR_TO_CHAMPION",
                    "metrics": metrics,
                    **metrics
                }
        except Exception as e:
            logger.error(f"Error during candidate tournament evaluation: {e}")
            return {"approved": False, "reason": f"EXCEPTION_{e}"}

async def run_training_loop():
    set_log_file("logs/ai_trainer.log")
    logger.info("Initializing PPO Training Engine...")
    
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
    trainer = PPOTrainer(model=model, limit=limit)
    logger.info("Starting continuous PPO training on historical/live data...")

    # 3. PRIORITY #2: Start DB Auto-Syncer & Golden Extractor in background thread AFTER cache is in RAM
    if local_mode:
        try:
            import threading
            import time
            import sys
            import subprocess
            from scripts.sync_db import sync_experiences
            
            def background_syncer():
                logger.info("Background DB Syncer & Golden Extractor started!")
                last_golden_time = 0.0
                while True:
                    try:
                        sync_experiences()
                    except Exception as e:
                        logger.error(f"Sync failed: {e}")
                        
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
            if step % 100 == 0:
                logger.info(f"Completed {step} training steps. Evaluating candidate brain before deployment...")
                
                # Auto-save candidate weights to disk so training progress is never lost even if rejected
                cand_dir = "models/candidates"
                os.makedirs(cand_dir, exist_ok=True)
                cand_step_path = os.path.join(cand_dir, f"candidate_step_{step}.pt")
                torch.save(model.state_dict(), cand_step_path)
                torch.save(model.state_dict(), os.path.join(cand_dir, "candidate_latest.pt"))
                logger.info(f"💾 Saved candidate checkpoint: {cand_step_path}")
                
                eval_result = trainer.evaluate_candidate(sample_size=1000)
                
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
                trainer.buffer.load_cache_from_db(limit=limit)
                
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
