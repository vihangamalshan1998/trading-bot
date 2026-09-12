import asyncio
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import os

from core.logging.logger import logger
from core.ai.replay_buffer import ReplayBuffer
from core.ai.registry import ModelRegistry
from apps.research.model import MultiSymbolActorCritic
from core.config.settings import settings

class OnlineTrainer:
    """
    Phase 4: Real Online Trainer without Dummy Losses.
    Continuously samples from the ReplayBuffer (MySQL) and updates a CANDIDATE model.
    Never mutates the active production weights directly.
    """
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.replay_buffer = ReplayBuffer(capacity_cache=100000)
        self.registry = ModelRegistry()
        
        # We start by copying the current production weights into our Candidate
        self.candidate_model = MultiSymbolActorCritic(num_symbols=self.num_symbols, macro_dim=8)
        try:
            self.candidate_model = self.registry.load_model(self.candidate_model) # Loads active
            logger.info("Loaded Active production model as the base for Candidate Training.")
        except Exception:
            logger.warning("No active production model found. Starting Candidate from random initialization.")
            
        self.candidate_model.train()
        self.optimizer = optim.Adam(self.candidate_model.parameters(), lr=1e-4)
        
        self.running = False
        self.batch_size = 64
        self.gamma = 0.99
        self.sync_interval_steps = 1000
        self.current_step = 0
        
    async def training_loop(self):
        logger.info("Starting real online Candidate training loop...")
        
        # Pre-fill cache from DB
        self.replay_buffer.load_cache_from_db(limit=10000)
        
        while self.running:
            if len(self.replay_buffer.cache) < self.batch_size:
                await asyncio.sleep(10)
                continue
                
            try:
                # 1. Sample intelligent batch (recent, rare, historical)
                batch = self.replay_buffer.sample(
                    batch_size=self.batch_size, 
                    recent_pct=0.4, historical_pct=0.3, rare_pct=0.2, random_pct=0.1
                )
                
                # 2. Build Tensors
                states, actions, rewards, next_states, dones = self.replay_buffer.build_tensors(batch)
                
                if states.numel() == 0: # Empty/failed dimension build
                    await asyncio.sleep(5)
                    continue
                    
                # 3. Real Training Pipeline (TD Error / Actor-Critic)
                # Forward pass current states
                action_logits, values = self.candidate_model(states)
                
                with torch.no_grad():
                    # Target values for next states
                    _, next_values = self.candidate_model(next_states)
                    target_values = rewards + self.gamma * next_values * (1 - dones)
                    
                # Critic Loss (MSE of Value)
                critic_loss = F.mse_loss(values, target_values)
                
                # Actor Loss (Simplified DPG/Advantage)
                advantage = (target_values - values).detach()
                actor_loss = -(action_logits.mean(dim=2) * advantage).mean()
                
                # Total Loss
                loss = critic_loss + actor_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.candidate_model.parameters(), 1.0)
                self.optimizer.step()
                
                self.current_step += 1
                
                if self.current_step % 100 == 0:
                    logger.info(f"Trainer Step {self.current_step} | Loss: {loss.item():.4f} (Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f})")
                    
                # Periodically save a candidate checkpoint
                if self.current_step % self.sync_interval_steps == 0:
                    version_id = f"candidate_v_{int(time.time())}"
                    metrics = {"loss": float(loss.item()), "steps": self.current_step}
                    
                    self.registry.save_model(
                        model=self.candidate_model, 
                        version_id=version_id, 
                        architecture=self.candidate_model.__class__.__name__, 
                        metrics=metrics
                    )
                    logger.info(f"Saved Candidate Checkpoint: {version_id}. Awaiting evaluation/promotion.")
                    
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                
            # Prevent 100% CPU lock in async loop
            await asyncio.sleep(0.1)

    async def start(self):
        self.running = True
        await self.training_loop()
        
    def stop(self):
        self.running = False
        
if __name__ == "__main__":
    trainer = OnlineTrainer(symbols=["BTCUSDT", "ETHUSDT"])
    try:
        asyncio.run(trainer.start())
    except KeyboardInterrupt:
        trainer.stop()
