import numpy as np
from core.logging.logger import logger
from apps.research.environment import FuturesTradingEnv
from apps.research.baselines import MomentumStrategy, MeanReversionStrategy
from apps.research.train import generate_mock_data

def evaluate_strategy(env, strategy, name="Strategy"):
    obs, _ = env.reset()
    done = False
    
    total_reward = 0.0
    liquidation_count = 0
    max_balance = env.initial_balance
    min_balance = env.initial_balance
    
    while not done:
        action = strategy.step(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        current_margin = info.get("margin_balance", env.wallet_balance)
        
        if current_margin > max_balance:
            max_balance = current_margin
        if current_margin < min_balance:
            min_balance = current_margin
            
        if info.get("liquidation", False):
            liquidation_count += 1
            
        done = terminated or truncated
        
    final_margin = env.wallet_balance
    if env.position_size != 0:
        # force close at end
        env._close_position(obs[6])
        final_margin = env.wallet_balance
        
    pnl = final_margin - env.initial_balance
    drawdown = (max_balance - min_balance) / max_balance * 100 if max_balance > 0 else 0
    
    logger.info(f"--- {name} Results ---")
    logger.info(f"Final PnL: ${pnl:.2f}")
    logger.info(f"Max Drawdown: {drawdown:.2f}%")
    logger.info(f"Liquidations: {liquidation_count}")
    return pnl

if __name__ == "__main__":
    # Generate a longer mock dataset (e.g. 2000 steps with trends)
    logger.info("Generating synthetic market data...")
    data = []
    price = 50000.0
    for i in range(2000):
        # A mix of trend and mean-reversion noise
        price += (10.0 * np.sin(i / 50.0)) + np.random.normal(0, 5.0)
        data.append({
            "best_bid": price - 1.0,
            "best_ask": price + 1.0,
            "mid_price": price,
            "spread_bps": 2.0 / price * 10000,
            "imbalance": 0.0,
            "vwap_recent": price
        })
        
    logger.info("Evaluating Baselines in FuturesTradingEnv (10x Leverage)")
    
    env_momentum = FuturesTradingEnv(historical_data=data, leverage=10, initial_balance=1000.0)
    momentum = MomentumStrategy(window=20, threshold=0.002)
    evaluate_strategy(env_momentum, momentum, "Momentum")
    
    env_mr = FuturesTradingEnv(historical_data=data, leverage=10, initial_balance=1000.0)
    mr = MeanReversionStrategy(window=40, deviation_threshold=0.005)
    evaluate_strategy(env_mr, mr, "Mean Reversion")
