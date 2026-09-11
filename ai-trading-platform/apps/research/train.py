import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Any

from apps.research.environment import FuturesTradingEnv
from apps.research.model import TradingNet
from core.logging.logger import logger

def generate_mock_data(steps: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate synthetic sine wave data for basic training verification.
    """
    data = []
    for i in range(steps):
        # A simple sine wave pattern
        price = 50000.0 + (1000.0 * np.sin(i / 10.0))
        data.append({
            "best_bid": price - 1.0,
            "best_ask": price + 1.0,
            "mid_price": price,
            "spread_bps": 2.0 / price * 10000,
            "imbalance": np.cos(i / 10.0), # Correlated with price momentum
            "vwap_recent": price
        })
    return data

def train_reinforce(episodes: int = 500, learning_rate: float = 1e-3, gamma: float = 0.99):
    logger.info("Initializing REINFORCE training loop...")
    
    # 1. Setup Environment
    data = generate_mock_data(steps=200)
    env = FuturesTradingEnv(historical_data=data, initial_balance=1000.0)
    
    # 2. Setup Model
    model = TradingNet(input_dim=9, hidden_dim=64, output_dim=5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Checkpoint dir
    os.makedirs("models", exist_ok=True)
    best_reward = -float("inf")
    
    logger.info("Starting training episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        while True:
            # Add batch dimension to state
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # Forward pass to get action logits
            logits = model(state_tensor)
            
            # Policy gradient formulation requires probabilities
            # (We use softmax over the output layer)
            probs = F.softmax(logits, dim=1)
            m = Categorical(probs)
            
            # Sample an action
            action = m.sample()
            log_probs.append(m.log_prob(action))
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(action.item())
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
        
        # Normalize returns for stability
        if returns_tensor.numel() > 1 and returns_tensor.std() > 0:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
            
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns_tensor):
            policy_loss.append(-log_prob * R)
            
        policy_loss = torch.cat(policy_loss).sum()
        
        # Backpropagate
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode} | Total Reward: {total_reward:.2f} | Loss: {policy_loss.item():.4f}")
            
        # Checkpointing
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(model.state_dict(), "models/best_model.pth")
            if episode % 10 != 0:
                logger.info(f"New best model saved at Episode {episode} (Reward: {best_reward:.2f})")

    logger.info("Training complete.")

if __name__ == "__main__":
    train_reinforce()
