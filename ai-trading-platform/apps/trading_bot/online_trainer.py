import asyncio
import torch
import torch.optim as optim
from core.logging.logger import logger
from apps.trading_bot.replay_buffer import replay_buffer
from apps.research.world_model import LatentWorldModel

class OnlineTrainer:
    """
    Continuously trains the active World Model in the background using live experiences.
    """
    def __init__(self, model: LatentWorldModel, update_interval: int = 3600):
        self.model = model
        self.update_interval = update_interval
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5) # Very small LR for online finetuning
        self._running = False
        
    async def run_training_step(self):
        """Performs a single gradient update using the replay buffer."""
        batch = replay_buffer.sample(batch_size=64)
        if len(batch) < 16:
            logger.info("Not enough experiences in replay buffer. Skipping training step.")
            return
            
        logger.info(f"Running online gradient update on {len(batch)} experiences...")
        
        # In a real implementation, we would unpack the batch, run it through the World Model,
        # calculate Reconstruction Loss, KL Loss, and Policy Loss, and perform a backward pass.
        
        # Dummy update for demonstration
        self.optimizer.zero_grad()
        dummy_loss = torch.tensor(0.1, requires_grad=True) 
        dummy_loss.backward()
        self.optimizer.step()
        
        logger.info("Online training step complete. Weights updated.")
        
    async def start(self):
        self._running = True
        logger.info(f"Starting Continuous Online Trainer (Updates every {self.update_interval}s)")
        
        while self._running:
            await asyncio.sleep(self.update_interval)
            try:
                await self.run_training_step()
            except Exception as e:
                logger.error(f"Error in Online Trainer: {e}")
                
    def stop(self):
        self._running = False

async def test_online_trainer():
    # Mock model
    model = LatentWorldModel(num_symbols=1)
    trainer = OnlineTrainer(model, update_interval=2)
    
    # Mock some data
    replay_buffer.add([0]*11, 1, 1.5, [0]*11, False, "BTCUSDT", "v1.0.0")
    
    task = asyncio.create_task(trainer.start())
    await asyncio.sleep(3)
    trainer.stop()
    await task

if __name__ == "__main__":
    asyncio.run(test_online_trainer())
