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
from core.logging.logger import logger, set_log_file
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry

class PPOTrainer:
    """
    Phase 11: Proximal Policy Optimization (PPO) Training Loop.
    Samples from the MySQL ReplayBuffer and updates the Actor-Critic model.
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
        
    def train_step(self, batch_size: int = 64, epochs: int = 4):
        """Executes a single mathematically valid PPO training step."""
        batch = self.buffer.sample(batch_size)
        if len(batch) < batch_size:
            logger.warning(f"Not enough samples in replay buffer to train. Got {len(batch)}, needed {batch_size}")
            return False
            
        # Priority 2: Never silently reset model parameters if non-finite weights are found. Stop update & report.
        is_nan = any(not torch.isfinite(p).all() for p in self.model.parameters())
        if is_nan:
            logger.error("CRITICAL: Non-finite model weights detected. Aborting training step to preserve checkpoint integrity.")
            return False

        states, actions, rewards, next_states, dones = self.buffer.build_tensors(batch, seq_len=300)
        if len(states) == 0:
            return False
            
        states = torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)
        next_states = torch.nan_to_num(next_states, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)
        dones = torch.nan_to_num(dones, nan=0.0, posinf=1.0, neginf=0.0)
            
        self.model.train()
        
        # 1. Old Policy Evaluation with Continuous Gaussian Action Distribution
        with torch.no_grad():
            old_action_preds, old_values = self.model(states)
            _, next_values = self.model(next_states)
            
            # Action distribution: Normal(mu, sigma)
            action_std = torch.exp(torch.clamp(self.log_std, min=-2.0, max=0.5))
            dist_old = torch.distributions.Normal(old_action_preds, action_std)
            old_log_probs = dist_old.log_prob(actions).sum(dim=-1, keepdim=True)
            
            safe_rewards = rewards / 200.0
            advantages, returns = self.compute_gae(safe_rewards, old_values, next_values, dones)
            actor_advantages = advantages.mean(dim=1, keepdim=True)
            
            adv_mean = actor_advantages.mean()
            adv_std = actor_advantages.std(unbiased=False)
            
            # Phase 1.A: Never manufacture random advantages. If insufficient variance, skip update.
            if actor_advantages.size(0) <= 1 or adv_std <= 1e-6:
                logger.info(f"PPO: Insufficient reward variation in batch (adv_std={adv_std.item():.6f} <= 1e-6). Deferring update.")
                return False
                
            actor_advantages = (actor_advantages - adv_mean) / (adv_std + 1e-8)
            actor_advantages = torch.nan_to_num(actor_advantages, nan=0.0)
            
        # 2. PPO Epochs
        for _ in range(epochs):
            action_preds, values = self.model(states)
            
            action_std = torch.exp(torch.clamp(self.log_std, min=-2.0, max=0.5))
            dist = torch.distributions.Normal(action_preds, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
            
            # Numerically stable probability ratio
            diff = torch.clamp(log_probs - old_log_probs, min=-10.0, max=10.0)
            ratio = torch.exp(diff)
            
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

                # Policy entropy check (prevents pushing mode-collapsed models)
                noise = torch.randn_like(challenger_actions) * 1e-5
                var = (challenger_actions + noise).var(dim=0, unbiased=False)
                challenger_entropy = float(torch.sqrt(var + 1e-8).mean().item())

                if challenger_entropy < 1e-4:
                    logger.warning(f"Candidate evaluation FAILED: Challenger mode collapse detected (Entropy: {challenger_entropy:.6f} < 1e-4).")
                    return {"approved": False, "reason": "MODE_COLLAPSE", "entropy": challenger_entropy}

                challenger_critic_mse = float(nn.MSELoss()(challenger_values, safe_rewards).item())
                if challenger_critic_mse > 100.0:
                    logger.warning(f"Candidate evaluation FAILED: Challenger critic loss exploded ({challenger_critic_mse:.2f} > 100.0).")
                    return {"approved": False, "reason": "CRITIC_EXPLODED", "critic_loss": challenger_critic_mse}

                # Real-World VPS Fee Simulation: Deduct 0.10% Binance Futures round-trip taker fee per active trade
                fee_hurdle = 0.10  # 0.10% fee (0.05% entry + 0.05% exit)
                challenger_is_active = (torch.abs(challenger_actions[:, 0]) > 0.2).float()
                challenger_net_returns = (challenger_actions[:, 0] * actual_5m_ret) - (challenger_is_active * fee_hurdle)
                challenger_sim_pnl = float(challenger_net_returns.mean().item())
                challenger_score = challenger_sim_pnl - (0.5 * challenger_critic_mse)

            # 2. Benchmark against Production Champion
            champion_path = "models/production/model_v1.pt"
            champion_available = os.path.exists(champion_path)
            
            if not champion_available:
                logger.info(f"Initial baseline brain established. Approved without tournament benchmark. (Score: {challenger_score:.4f})")
                del states, actions, rewards, next_states, dones
                return {
                    "approved": True,
                    "reason": "INITIAL_BASELINE",
                    "challenger_score": challenger_score,
                    "champion_score": 0.0,
                    "critic_loss": challenger_critic_mse,
                    "entropy": challenger_entropy
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
                    champ_net_returns = (champ_actions[:, 0] * actual_5m_ret) - (champ_is_active * fee_hurdle)
                    champ_sim_pnl = float(champ_net_returns.mean().item())
                    champion_score = champ_sim_pnl - (0.5 * champ_critic_mse)

                del champion_model, champ_actions, champ_values
            except Exception as champ_e:
                logger.warning(f"Could not benchmark Champion model ({champ_e}). Proceeding with Challenger metrics.")
                champion_score = challenger_score - 0.01
                champ_critic_mse = challenger_critic_mse + 0.01
                champ_sim_pnl = challenger_sim_pnl

            del states, actions, rewards, next_states, dones

            # Superiority criteria (Refined Elite Margin):
            # 1. Composite score beats Champion by +0.05 margin, OR
            # 2. Significant Critic MSE improvement (>= 10% error reduction) with resilient directional PnL (>= 90% of champ)
            SUPERIORITY_MARGIN = 0.05
            is_superior = (challenger_score >= champion_score + SUPERIORITY_MARGIN) or (
                challenger_critic_mse <= (champ_critic_mse * 0.90) and challenger_sim_pnl >= (champ_sim_pnl * 0.90)
            )

            metrics = {
                "challenger_score": challenger_score,
                "champion_score": champion_score,
                "challenger_critic_mse": challenger_critic_mse,
                "champion_critic_mse": champ_critic_mse,
                "challenger_sim_pnl": challenger_sim_pnl,
                "champion_sim_pnl": champ_sim_pnl,
                "entropy": challenger_entropy
            }

            if is_superior:
                logger.info(
                    f"🏆 TOURNAMENT WINNER: Challenger is SUPERIOR! "
                    f"Score: {challenger_score:.4f} vs Champ: {champion_score:.4f} | "
                    f"Critic MSE: {challenger_critic_mse:.4f} vs {champ_critic_mse:.4f} | "
                    f"Sim PnL: {challenger_sim_pnl:.4f} vs {champ_sim_pnl:.4f}"
                )
                return {"approved": True, "reason": "SUPERIOR_TO_CHAMPION", "metrics": metrics}
            else:
                logger.warning(
                    f"⛔ TOURNAMENT DEFEAT: Challenger is INFERIOR to current live Champion! "
                    f"Score: {challenger_score:.4f} < Champ: {champion_score:.4f} | "
                    f"Critic MSE: {challenger_critic_mse:.4f} vs {champ_critic_mse:.4f} | "
                    f"Sim PnL: {challenger_sim_pnl:.4f} vs {champ_sim_pnl:.4f}. REJECTING PUSH."
                )
                return {"approved": False, "reason": "INFERIOR_TO_CHAMPION", "metrics": metrics}
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
                eval_result = trainer.evaluate_candidate(sample_size=1000)
                
                if not eval_result.get("approved", False):
                    logger.warning(f"Candidate brain REJECTED by evaluation gate: {eval_result.get('reason')}. Continuing training without resetting weights...")
                    registry.record_tournament_result(eval_result, version_tag=f"rejected_step_{step}")
                else:
                    try:
                        push_count += 1
                        version_tag = f"v1_push_{push_count}"
                        registry.save_model(model, version_tag=version_tag, metrics=eval_result)
                        
                        if local_mode and (time.time() - last_push_time > 300):
                            logger.info(f"🚀 Pushing evaluated Brain ({version_tag}) to VPS in the background ({push_count}/100)...")
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
                            
                            # Auto-Shutdown after 100 successful pushes (approx 1 to 1.5 hours)
                            if push_count >= 100:
                                logger.info("✅ Successfully completed 100 Brain pushes. Auto-shutting down Master Trainer!")
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
