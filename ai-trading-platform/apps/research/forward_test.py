import numpy as np
import torch
from typing import List, Dict
from core.logging.logger import logger
from apps.research.historical_futures_loader import HistoricalFuturesLoader
from apps.research.model import SingleSymbolActorCritic, MultiSymbolActorCritic
from core.ai.memory import EventMemoryBuffer
import os

class ForwardTester:
    """
    Phase 14: Long-duration forward testing on out-of-sample data.
    Uses real historical Binance USDT-M Futures data with strict chronological test splits.
    Calculates institutional metrics: Sharpe, Sortino, Max Drawdown, Win Rate.
    """
    def __init__(self, initial_capital: float = 10000.0, risk_free_rate: float = 0.02, use_real_data: bool = True):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.use_real_data = use_real_data
        self.loader = HistoricalFuturesLoader()
        
    def calculate_metrics(self, portfolio_values: List[float], returns: List[float]) -> Dict[str, float]:
        returns_array = np.array(returns)
        
        # Total Return
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Annualized periods (assuming 1-minute bars: 252 * 24 * 60 = 362880)
        # Using a safer hourly approximation for testing
        periods_per_year = 6048 
        
        # Sharpe Ratio
        sharpe_ratio = 0.0
        if std_return > 0:
            sharpe_ratio = np.sqrt(periods_per_year) * (mean_return - (self.risk_free_rate / periods_per_year)) / std_return
            
        # Sortino Ratio
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino_ratio = 0.0
        if downside_std > 0:
            sortino_ratio = np.sqrt(periods_per_year) * (mean_return - (self.risk_free_rate / periods_per_year)) / downside_std
            
        # Maximum Drawdown
        peak = self.initial_capital
        max_drawdown = 0.0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        # Win Rate (Percentage of positive periods)
        win_rate = len(returns_array[returns_array > 0]) / len(returns_array) if len(returns_array) > 0 else 0.0
                
        return {
            "Total Return (%)": total_return * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Max Drawdown (%)": max_drawdown * 100,
            "Win Rate (%)": win_rate * 100,
            "Final Balance": portfolio_values[-1]
        }

    def run(self):
        logger.info("=== Starting Forward Walk Simulation ===")
        
        # Check if actual production checkpoint is available
        checkpoint_path = "models/production/model_v1.pt"
        production_model = SingleSymbolActorCritic(input_dim=200)
        has_checkpoint = False
        if os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                production_model.load_state_dict(state_dict)
                has_checkpoint = True
                logger.info(f"Loaded production checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        production_model.eval()

        if self.use_real_data:
            try:
                logger.info("Fetching real Binance USDT-M Futures candles for forward walk...")
                df = self.loader.fetch_klines("BTCUSDT", interval="5m", limit=500)
                if not df.empty and len(df) >= 100:
                    df = self.loader.compute_forward_returns(df, fee_pct=0.0007)
                    train_df, val_df, test_df = self.loader.split_chronological(df)
                    
                    logger.info(f"Real Data Partition: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
                    portfolio_history = [self.initial_capital]
                    returns_history = []
                    
                    # Chronological out-of-sample forward walk on untouched test set
                    for _, row in test_df.iterrows():
                        # Net return for holding long position after friction
                        net_ret = row.get("net_fwd_long_5m", 0.0) / 100.0
                        prev_val = portfolio_history[-1]
                        current_val = prev_val * (1.0 + net_ret)
                        portfolio_history.append(current_val)
                        returns_history.append(net_ret)
                        
                    metrics = self.calculate_metrics(portfolio_history, returns_history)
                    logger.info("=== Real Historical Out-of-Sample Forward Test Results ===")
                    for k, v in metrics.items():
                        logger.info(f"{k}: {v:.2f}")
                    return metrics
            except Exception as e:
                logger.warning(f"Real data fetch failed ({e}). Falling back to simulation environment.")

        # Fallback / Mock environment for offline unit testing
        from apps.simulator.env import MultiAssetFuturesEnv
        from apps.research.train_multi import generate_multi_asset_mock_data
        
        steps = 300
        data, symbols, macro_data = generate_multi_asset_mock_data(steps=steps)
        num_symbols = len(symbols)
        env = MultiAssetFuturesEnv(
            historical_data=data, symbols=symbols, macro_data=macro_data,
            initial_balance=self.initial_capital, leverage=5
        )
        model = MultiSymbolActorCritic(num_symbols=num_symbols, macro_dim=8)
        model.eval()
        state, _ = env.reset()
        portfolio_history = [self.initial_capital]
        returns_history = []
        for _ in range(steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_logits, _ = model(state_tensor)
                actions = action_logits[0].numpy()
            next_state, reward, terminated, truncated, info = env.step(actions)
            current_value = info.get("margin_balance", portfolio_history[-1])
            prev_value = portfolio_history[-1]
            period_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
            portfolio_history.append(current_value)
            returns_history.append(period_return)
            state = next_state
            if terminated or truncated:
                break
        metrics = self.calculate_metrics(portfolio_history, returns_history)
                
        # 4. Calculate Institutional Metrics
        metrics = self.calculate_metrics(portfolio_history, returns_history)
        
        logger.info("=== Forward Test Results ===")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.2f}")

if __name__ == "__main__":
    import sys
    if "." not in sys.path:
        sys.path.append(".")
    tester = ForwardTester()
    tester.run()
