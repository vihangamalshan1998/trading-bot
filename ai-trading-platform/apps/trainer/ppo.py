import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import asyncio
import json
import time
from core.db.redis import redis_manager
from core.ai.replay_buffer import ReplayBuffer
from core.logging.logger import logger
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry

class PPOTrainer:
    """
    Phase 11: Proximal Policy Optimization (PPO) Training Loop.
    Samples from the MySQL ReplayBuffer and updates the Actor-Critic model.
    """
    def __init__(self, model: SingleSymbolActorCritic, lr: float = 3e-4, gamma: float = 0.99, clip_epsilon: float = 0.2):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.buffer = ReplayBuffer()
        
        # Load recent cache from DB
        self.buffer.load_cache_from_db(limit=1000)
        
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Generalized Advantage Estimation (GAE)."""
        # Simplified advantage calculation for independent samples
        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values
        advantages = deltas # In a real sequence we'd accumulate with lambda
        returns = advantages + values
        return advantages, returns
        
    def train_step(self, batch_size: int = 64, epochs: int = 4):
        """Executes a single PPO training step."""
        batch = self.buffer.sample(batch_size)
        if len(batch) < batch_size:
            logger.warning(f"Not enough samples in replay buffer to train. Got {len(batch)}, needed {batch_size}")
            return
            
        states, actions, rewards, next_states, dones = self.buffer.build_tensors(batch)
        if len(states) == 0:
            return
            
        self.model.train()
        
        # 1. Get old log probabilities and values
        with torch.no_grad():
            old_action_preds, old_values = self.model(states)
            _, next_values = self.model(next_states)
            
            # Use MSE between action and predicted action as proxy for probability
            old_log_probs = -((actions - old_action_preds) ** 2).mean(dim=-1, keepdim=True)
            
            advantages, returns = self.compute_gae(rewards, old_values, next_values, dones)
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # 2. PPO Epochs
        for _ in range(epochs):
            action_preds, values = self.model(states)
            
            # Simple Gaussian log prob proxy
            log_probs = -((actions - action_preds) ** 2).mean(dim=-1, keepdim=True)
            
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(values, returns)
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
        logger.info(f"PPO Training Step Complete | Loss: {loss.item():.4f}")
        
        # Publish metrics to Redis asynchronously via asyncio.create_task or run_coroutine_threadsafe
        # We assume trainer loop might be sync or async. Let's provide a safe sync wrapper or fire-and-forget
        payload = {
            "timestamp": time.time(),
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item())
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
    logger.info("Initializing PPO Training Engine...")
    
    # We need to initialize the model first
    from core.config.settings import settings
    # The ReplayBuffer currently returns 29-dim state vectors (single symbol + portfolio + position)
    model = SingleSymbolActorCritic(input_dim=29)
    registry = ModelRegistry()
    
    trainer = PPOTrainer(model=model)
    logger.info("Starting continuous PPO training on historical/live data...")
    
    step = 0
    try:
        while True:
            trainer.train_step(batch_size=64, epochs=4)
            await asyncio.sleep(1.0) # Prevent 100% CPU usage
            step += 1
            if step % 100 == 0:
                logger.info(f"Completed {step} training steps. Saving model...")
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
