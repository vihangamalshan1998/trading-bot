import torch
import numpy as np
from typing import List, Dict, Any

from apps.research.environment import SpotTradingEnv
from apps.research.model import TradingNet
from apps.research.train import generate_mock_data
from core.logging.logger import logger

def evaluate_model(model_path: str):
    logger.info(f"Evaluating model: {model_path}")
    
    # Generate distinct validation data
    data = generate_mock_data(steps=500)
    env = SpotTradingEnv(historical_data=data, initial_balance=1000.0)
    
    model = TradingNet(input_dim=6, hidden_dim=64, output_dim=3)
    
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval() # Set to evaluation mode
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return
        
    state, _ = env.reset()
    total_reward = 0.0
    trades_taken = 0
    max_portfolio = 1000.0
    min_portfolio = 1000.0
    
    with torch.no_grad():
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = model(state_tensor)
            
            # Deterministic action selection (argmax)
            action = torch.argmax(logits, dim=1).item()
            
            if action in [1, 2]:
                trades_taken += 1
                
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            pv = info.get("portfolio_value", 1000.0)
            max_portfolio = max(max_portfolio, pv)
            min_portfolio = min(min_portfolio, pv)
            
            if terminated or truncated:
                break
                
    final_pnl = info.get("portfolio_value", 1000.0) - 1000.0
    max_drawdown = max_portfolio - min_portfolio
    
    logger.info("Evaluation Complete")
    logger.info(f"Total Trades Taken: {trades_taken}")
    logger.info(f"Cumulative PnL: ${final_pnl:.2f}")
    logger.info(f"Max Drawdown: ${max_drawdown:.2f}")
    logger.info(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    evaluate_model("models/best_model.pth")
