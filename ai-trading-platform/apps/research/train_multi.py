import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time

from apps.simulator.env import MultiAssetFuturesEnv
from apps.research.model import MultiSymbolTradingNet
from core.ai.registry import ModelRegistry
from core.logging.logger import logger

def generate_multi_asset_mock_data(steps: int = 1000):
    symbols = ["BTCUSDT", "ETHUSDT"]
    data = {sym: [] for sym in symbols}
    
    for i in range(steps):
        # BTC sine wave
        btc_price = 50000.0 + (1000.0 * np.sin(i / 20.0))
        data["BTCUSDT"].append({
            "mid_price": btc_price,
            "market_features": np.random.randn(25).tolist() # Mock 25-dim feature vector
        })
        
        # ETH correlated with BTC but lagged
        eth_price = 3000.0 + (100.0 * np.sin((i-5) / 20.0))
        data["ETHUSDT"].append({
            "mid_price": eth_price,
            "market_features": np.random.randn(25).tolist() # Mock 25-dim feature vector
        })
        
    # Mock macro data
    macro_data = []
    for i in range(steps):
        macro_data.append({
            "sentiment_score": np.sin(i / 100.0), # Slowly shifting macro sentiment
            "volatility_expectation": 0.5,
            "regime": 1.0 if np.sin(i / 200.0) > 0 else -1.0 # 1.0 (Bull) or -1.0 (Bear)
        })
        
    return data, symbols, macro_data

def train_multi_symbol_reinforce(episodes: int = 200, learning_rate: float = 5e-4, gamma: float = 0.99):
    logger.info("Initializing Multi-Symbol REINFORCE training loop...")
    
    # Setup Env
    data, symbols, macro_data = generate_multi_asset_mock_data(steps=300)
    env = MultiAssetFuturesEnv(
        historical_data=data, 
        symbols=symbols, 
        macro_data=macro_data,
        initial_balance=10000.0,
        leverage=5
    )
    
    # Setup Model
    model = MultiSymbolTradingNet(num_symbols=len(symbols), macro_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    registry = ModelRegistry()
    best_reward = -float("inf")
    
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # Forward pass: shape (1, num_symbols, 3)
            actions = model(state_tensor)
            
            # Continuous actions, add exploration noise
            if model.training:
                # Add Gaussian noise for exploration
                noise = torch.randn_like(actions) * 0.2
                noisy_actions = torch.clamp(actions + noise, -1.0, 1.0)
                
                # Treat noisy_actions as sampled from a normal distribution around `actions`
                # Calculate log_prob of the sample under the Gaussian N(actions, 0.2)
                dist = torch.distributions.Normal(actions, torch.tensor([0.2]))
                log_prob = dist.log_prob(noisy_actions).sum() # Sum across action dimensions
                log_probs.append(log_prob)
                
                env_actions = noisy_actions[0].detach().numpy()
            else:
                env_actions = actions[0].detach().numpy()
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(env_actions)
            rewards.append(reward)
            
            if terminated or truncated:
                break
                
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        if returns_tensor.numel() > 1 and returns_tensor.std() > 0:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            
        policy_loss = []
        for log_prob, R in zip(log_probs, returns_tensor):
            policy_loss.append(-log_prob * R)
            
        if len(policy_loss) > 0:
            policy_loss = torch.stack(policy_loss).sum()
            
            optimizer.zero_grad()
            policy_loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_val = policy_loss.item()
        else:
            loss_val = 0.0
        
        total_reward = sum(rewards)
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode:03d} | Total Reward: {total_reward:8.2f} | Loss: {loss_val:.4f}")
            
        # Checkpointing
        if total_reward > best_reward:
            best_reward = total_reward
            if episode > 0: # Avoid saving initial random policy noise
                version_id = f"v{int(time.time())}_ep{episode}"
                metrics = {"best_reward": float(best_reward), "episode": episode}
                registry.save_model(model, version_id, "MultiSymbolTradingNet_v2", metrics)
                logger.info(f"*** New High Score! Saved version {version_id} ***")

    logger.info("Training complete.")

if __name__ == "__main__":
    # Ensure working directory is correct for imports if running directly
    import sys
    if "." not in sys.path:
        sys.path.append(".")
    train_multi_symbol_reinforce()
