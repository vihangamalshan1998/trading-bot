import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

from apps.research.environment import MultiAssetFuturesEnv
from apps.research.world_model import LatentWorldModel
from apps.research.train_multi import generate_multi_asset_mock_data
from core.logging.logger import logger

def train_world_model(episodes: int = 50, learning_rate: float = 3e-4, gamma: float = 0.99):
    logger.info("Initializing World Model (Dreamer) training loop...")
    
    # 1. Setup Env
    data, symbols, macro_data = generate_multi_asset_mock_data(steps=300)
    env = MultiAssetFuturesEnv(
        historical_data=data, 
        symbols=symbols, 
        macro_data=macro_data,
        initial_balance=10000.0,
        leverage=5
    )
    
    # 2. Setup World Model
    num_symbols = len(symbols)
    macro_dim = 8
    actions_per_symbol = 3 # (action_type, confidence, target_size)
    
    model = LatentWorldModel(
        num_symbols=num_symbols, 
        macro_dim=macro_dim,
        embed_dim=128,
        hidden_dim=128,
        state_dim=32
    )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. Training Loop
    for episode in range(episodes):
        state, _ = env.reset()
        
        # Initialize memory states
        batch_size = 1
        hidden = torch.zeros(batch_size, model.rssM.hidden_dim)
        prev_action = torch.zeros(batch_size, num_symbols * actions_per_symbol)
        
        total_reconstruction_loss = 0.0
        total_kl_loss = 0.0
        total_reward_loss = 0.0
        total_policy_loss = 0.0
        total_reward = 0.0
        
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # Forward pass through World Model
            out = model(state_tensor, prev_action, hidden)
            
            # Extract outputs
            hidden = out["hidden"]
            actions = out["actions"] # (batch, num_symbols, 3)
            
            # Continuous actions, add exploration noise
            if model.training:
                noise = torch.randn_like(actions) * 0.1
                actions = torch.clamp(actions + noise, -1.0, 1.0)
            
            # Extract action for environment
            env_actions = actions[0].detach().numpy() # (num_symbols, 3)
            prev_action = actions.view(1, -1) # Flatten for next step
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(env_actions)
            total_reward += reward
            
            # --- CALCULATE LOSSES ---
            
            # 1. Reconstruction Loss (MSE between actual obs and reconstructed obs)
            recon_loss = F.mse_loss(out["reconstructed_obs"], state_tensor)
            
            # 2. Reward Loss (MSE between actual reward and predicted reward)
            reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
            reward_loss = F.mse_loss(out["pred_reward"], reward_tensor)
            
            # 3. KL Divergence Loss (Force prior to match posterior)
            post_dist = Normal(out["post_mean"], out["post_std"])
            prior_dist = Normal(out["prior_mean"].detach(), out["prior_std"].detach())
            kl_loss = torch.distributions.kl_divergence(post_dist, prior_dist).mean()
            
            # 4. Policy Loss (Simplified Actor-Critic TD Error for Continuous Actions)
            with torch.no_grad():
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                next_out = model(next_state_tensor, prev_action, hidden)
                target_value = reward_tensor + gamma * next_out["value"] * (1 - int(terminated))
                
            advantage = (target_value - out["value"]).detach()
            value_loss = F.mse_loss(out["value"], target_value)
            
            # Deterministic Policy Gradient (DPG) style surrogate: maximize Q(s, \mu(s))
            # Since we just have an advantage, we multiply the output of the actor by the advantage 
            # to push actions in the direction of positive advantage.
            # (In a real DPG, we'd pass actor actions through the critic directly).
            # For simplicity here, we do pseudo-policy-gradient:
            policy_loss = -(actions.mean() * advantage).mean()
            
            # Total Loss
            loss = recon_loss + reward_loss + kl_loss + value_loss + policy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Detach hidden state to prevent inplace autograd graph errors across loop steps
            hidden = hidden.detach()
            
            total_reconstruction_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_reward_loss += reward_loss.item()
            total_policy_loss += policy_loss.item()
            
            state = next_state
            
            if terminated or truncated:
                break
                
        if episode % 5 == 0:
            logger.info(f"Ep {episode:03d} | Reward: {total_reward:6.2f} | ReconL: {total_reconstruction_loss:6.2f} | KLL: {total_kl_loss:6.2f} | PolL: {total_policy_loss:6.2f}")

    logger.info("World Model training complete.")

if __name__ == "__main__":
    import sys
    if "." not in sys.path:
        sys.path.append(".")
    train_world_model()
