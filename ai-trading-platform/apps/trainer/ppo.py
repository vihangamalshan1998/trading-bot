import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import asyncio
import json
import time
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
    def __init__(self, model: SingleSymbolActorCritic, lr: float = 3e-4, gamma: float = 0.99, clip_epsilon: float = 0.2):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.buffer = ReplayBuffer()
        
        # V3 Upgrade: The 300-frame sequence is massive (2MB per record). 
        # Cache limit math: 500 exp × 300 frames × 200 features × 4 bytes = ~117MB cache.
        # With gc.collect() after every step, total trainer RAM stays ~500MB (safe for VPS).
        # 500 is the sweet spot: enough diversity for sampler without risking OOM.
        self.buffer.load_cache_from_db(limit=500)

        
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Generalized Advantage Estimation (GAE) for Multi-Horizon."""
        # Both rewards and values are shape (batch_size, 3)
        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values
        advantages = deltas
        returns = advantages + values
        return advantages, returns
        
    def train_step(self, batch_size: int = 64, epochs: int = 4):
        """Executes a single PPO training step."""
        batch = self.buffer.sample(batch_size)
        if len(batch) < batch_size:
            logger.warning(f"Not enough samples in replay buffer to train. Got {len(batch)}, needed {batch_size}")
            return
            
        # Check if model weights are NaN
        is_nan = any(torch.isnan(p).any() for p in self.model.parameters())
        if is_nan:
            logger.error("MODEL WEIGHTS ARE NAN!")
            # Re-initialize model to recover
            self.model.apply(lambda m: hasattr(m, 'reset_parameters') and m.reset_parameters())
            logger.info("Model weights have been re-initialized.")

            
        states, actions, rewards, next_states, dones = self.buffer.build_tensors(batch, seq_len=300)
        if len(states) == 0:
            return
            
        # Absolute safety net: Clean any NaN values pulled from the MySQL database
        states = torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)
        next_states = torch.nan_to_num(next_states, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)
        dones = torch.nan_to_num(dones, nan=0.0, posinf=1.0, neginf=0.0)
            
        self.model.train()
        
        # 1. Get old log probabilities and values
        with torch.no_grad():
            old_action_preds, old_values = self.model(states)
            _, next_values = self.model(next_states)
            
            # Use MSE between action and predicted action as proxy for probability
            old_log_probs = -((actions - old_action_preds) ** 2).mean(dim=-1, keepdim=True)
            
            # Scale rewards to prevent gradient explosion.
            # Multi-horizon rewards are summed over hours (can reach 100+), so we need a larger divisor.
            safe_rewards = rewards / 200.0
            
            advantages, returns = self.compute_gae(safe_rewards, old_values, next_values, dones)
            
            # Average advantages across the 3 horizons for the Actor
            actor_advantages = advantages.mean(dim=1, keepdim=True)
            
            adv_mean = actor_advantages.mean()
            adv_std = actor_advantages.std(unbiased=False)
            
            # CRITICAL FIX: If all rewards are identical (e.g., all -0.02% fees),
            # standard deviation is 0. Normalizing it zeroes out the advantages entirely,
            # which deletes the Actor's ability to learn that the action was bad!
            if actor_advantages.size(0) > 1 and adv_std > 1e-5:
                actor_advantages = (actor_advantages - adv_mean) / (adv_std + 1e-8)
            else:
                # CRITICAL FIX: If std is zero, the actor gets no gradient and mode-collapses.
                # Inject noise to force exploration!
                actor_advantages = torch.randn_like(actor_advantages) * 0.1
                
            # Final Safety Net against NaN values
            actor_advantages = torch.nan_to_num(actor_advantages, nan=0.0)
            
        # 2. PPO Epochs
        for _ in range(epochs):
            action_preds, values = self.model(states)
            
            # Simple Gaussian log prob proxy
            log_probs = -((actions - action_preds) ** 2).mean(dim=-1, keepdim=True)
            
            # CRITICAL FIX: Clamp the diff before exp() to prevent ratio from becoming Inf/NaN
            diff = torch.clamp(log_probs - old_log_probs, min=-20.0, max=20.0)
            ratio = torch.exp(diff)
            
            # Clipped surrogate objective
            surr1 = ratio * actor_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * actor_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic must learn all 3 horizons (MSE handles the shape automatically)
            # Normalize returns to prevent Critic MSE from exploding when rewards are large
            returns_normalized = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
            values_normalized = (values - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
            critic_loss = nn.MSELoss()(values_normalized, returns_normalized)
            
            # Entropy Bonus (prevents Mode Collapse / 100% BUY situations)
            if action_preds.size(0) > 1:
                # CRITICAL FIX: PyTorch's std() has a derivative of NaN when the variance is exactly 0
                # (division by zero in the square root). We MUST add epsilon BEFORE the square root.
                # Also, add tiny noise to prevent d(var)/d(x) = 0 during absolute mode collapse
                noise = torch.randn_like(action_preds) * 1e-5
                var = (action_preds + noise).var(dim=0, unbiased=False)
                entropy = torch.sqrt(var + 1e-8).mean()
            else:
                entropy = torch.tensor(0.0, device=action_preds.device)
                
            entropy = torch.nan_to_num(entropy, nan=0.0) # Absolute safety net
            entropy_coef = 0.01 # Standard PPO value. 0.15 was way too high and fought against the Actor learning a real policy.
            
            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
            
            # Check for NaN in loss before backward
            if torch.isnan(loss).any():
                logger.error("Loss is NaN! Skipping step.")
                break
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
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
        import gc
        del states, next_states, old_action_preds, old_values, next_values
        del old_log_probs, safe_rewards, advantages, returns, actor_advantages
        gc.collect()
            
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

        
        # Since trainer might run synchronously, we can dispatch it via the running loop or a new one
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_metrics(payload))
        except RuntimeError:
            asyncio.run(self._publish_metrics(payload))
            
    async def _publish_metrics(self, payload: dict):
        try:
            if not redis_manager.redis:
                await redis_manager.connect()
            if redis_manager.redis:
                await redis_manager.redis.publish("training:metrics", json.dumps(payload))
        except Exception:
            pass # Suppress background metrics errors if Redis is down

    async def _publish_model_update(self):
        try:
            if not redis_manager.redis:
                await redis_manager.connect()
            if redis_manager.redis:
                payload = {"timestamp": time.time(), "event": "model_saved"}
                await redis_manager.redis.publish("training:model_update", json.dumps(payload))
        except Exception:
            pass

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
    
    trainer = PPOTrainer(model=model)
    logger.info("Starting continuous PPO training on historical/live data...")
    
    step = 0
    try:
        while True:
            trainer.train_step(batch_size=64, epochs=4)
            await asyncio.sleep(1.0) # Prevent 100% CPU usage
            step += 1
            if step % 100 == 0:
                logger.info(f"Completed {step} training steps. Refreshing cache and saving model...")
                # Clear memory before pulling new data to prevent RAM spikes
                import gc
                trainer.buffer.cache.clear()
                gc.collect()
                
                # Refresh cache from DB to prevent Mode Collapse (keep at 500 to match init limit)
                trainer.buffer.load_cache_from_db(limit=500)

                
                # Auto-delete data older than 30 days to save VPS disk space
                if step % 1000 == 0:
                    trainer.buffer.cleanup_old_data(days=30)
                    
                try:
                    registry.save_model(model)
                    
                    # Schedule model update broadcast
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(trainer._publish_model_update())
                    except RuntimeError:
                        asyncio.run(trainer._publish_model_update())
                except Exception as e:
                    logger.error(f"Failed to save model: {e}")
                    
    except asyncio.CancelledError:
        logger.info("PPO Training Engine shutting down...")
        
if __name__ == "__main__":
    try:
        asyncio.run(run_training_loop())
    except KeyboardInterrupt:
        pass
