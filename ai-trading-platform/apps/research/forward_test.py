import sys
import os
if "." not in sys.path:
    sys.path.insert(0, ".")

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
        # Priority 1: Require validated production checkpoint. Stop immediately if missing.
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"CRITICAL: Production model checkpoint not found at '{checkpoint_path}'. Aborting forward test to prevent unvalidated random evaluation.")

        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            production_model.load_state_dict(state_dict)
            logger.info(f"Loaded validated production checkpoint: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to load production checkpoint: {e}")
        production_model.eval()

        if self.use_real_data:
            logger.info("Fetching real Binance USDT-M Futures candles for leak-free forward walk...")
            df = self.loader.fetch_klines("BTCUSDT", interval="5m", limit=500)
            if df.empty or len(df) < 100:
                raise ValueError("Insufficient real Binance historical candle data for forward testing.")
                
            # Chronological partitioning: 70% Train, 15% Val, 15% Untouched Test
            train_df, val_df, test_df = self.loader.split_chronological(df)
            logger.info(f"Chronological Real Data Partition: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            # Leak-Free Model-Driven Walk-Forward Backtest
            portfolio_equity = float(self.initial_capital)
            portfolio_history = [portfolio_equity]
            returns_history = []
            
            position = 0.0          # Current position (-1.0 to +1.0)
            entry_price = 0.0       # Entry fill price
            trade_notional = 0.0    # Position notional
            leverage = 5            # 5x leverage
            
            # Realistic Execution Costs
            maker_fee_pct = 0.0002  # 0.02% maker entry
            taker_fee_pct = 0.0005  # 0.05% taker exit
            spread_pct = 0.00015    # 0.015% half-spread
            slippage_pct = 0.00010  # 0.010% slippage
            total_fees_paid = 0.0
            
            trade_pnls = []
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            
            # Sliding window of past prices for leak-free feature construction
            past_closes = []
            past_volumes = []
            
            for idx, row in test_df.iterrows():
                cur_price = float(row["close"])
                cur_vol = float(row.get("volume", 1000.0))
                past_closes.append(cur_price)
                past_volumes.append(cur_vol)
                
                # Priority 1: Zero Future Leakage.
                # Compute features strictly on PAST prices (t <= current bar)
                feat = torch.zeros((1, 200), dtype=torch.float32)
                
                if len(past_closes) >= 2:
                    past_ret_1bar = (past_closes[-1] - past_closes[-2]) / past_closes[-2]
                    feat[0, 0] = float(np.clip(past_ret_1bar * 100.0, -5.0, 5.0))
                if len(past_closes) >= 4:
                    past_ret_3bar = (past_closes[-1] - past_closes[-4]) / past_closes[-4]
                    feat[0, 1] = float(np.clip(past_ret_3bar * 100.0, -10.0, 10.0))
                if len(past_volumes) >= 5:
                    mean_vol = np.mean(past_volumes[-5:])
                    vol_ratio = cur_vol / max(mean_vol, 1.0)
                    feat[0, 2] = float(np.clip(vol_ratio, 0.0, 5.0))
                    
                # Position awareness in feature vector
                feat[0, 10] = position
                feat[0, 11] = (cur_price - entry_price) / entry_price if entry_price > 0 else 0.0

                # Model inference using production SingleSymbolActorCritic
                with torch.no_grad():
                    action_preds, critics = production_model(feat)
                    raw_act = float(action_preds[0, 0].item())
                    conf = float(1.0 / (1.0 + np.exp(-action_preds[0, 1].item())))
                    pred_edge = float(critics[0, 0].item()) if critics is not None and critics.numel() > 0 else 0.0

                # Production Action Contract:
                # Requires directional action conviction (|raw_act| > 0.20) and confidence (>= 0.65)
                desired_pos = 0.0
                if raw_act > 0.20 and conf >= 0.65:
                    desired_pos = 1.0   # LONG
                elif raw_act < -0.20 and conf >= 0.65:
                    desired_pos = -1.0  # SHORT

                # Calculate bar-by-bar unrealized PnL if holding a position
                bar_ret = 0.0
                if position != 0.0 and entry_price > 0.0 and len(past_closes) >= 2:
                    prev_bar_price = past_closes[-2]
                    bar_price_delta = (cur_price - prev_bar_price) / prev_bar_price
                    bar_ret = position * bar_price_delta * leverage

                # Order Execution & State Transitions
                if desired_pos != position:
                    # 1. Close existing position if any
                    if position != 0.0:
                        # Full round-trip trade PnL
                        gross_trade_ret = position * ((cur_price - entry_price) / entry_price)
                        # Exit friction: taker fee + half spread + slippage
                        exit_friction = taker_fee_pct + spread_pct + slippage_pct
                        net_trade_ret = (gross_trade_ret * leverage) - (exit_friction * leverage)
                        
                        trade_dollar_pnl = trade_notional * (gross_trade_ret - exit_friction)
                        portfolio_equity += trade_dollar_pnl
                        total_fees_paid += trade_notional * exit_friction
                        
                        trade_pnls.append(net_trade_ret)
                        total_trades += 1
                        if net_trade_ret > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1

                    # 2. Open new position if desired_pos != 0
                    if desired_pos != 0.0:
                        # Allocate 20% margin with leverage
                        margin_budget = portfolio_equity * 0.20
                        trade_notional = margin_budget * leverage
                        entry_friction = maker_fee_pct + spread_pct
                        
                        # Deduct entry friction immediately from equity
                        entry_fee_dollars = trade_notional * entry_friction
                        portfolio_equity -= entry_fee_dollars
                        total_fees_paid += entry_fee_dollars
                        
                        entry_price = cur_price
                        position = desired_pos
                    else:
                        position = 0.0
                        entry_price = 0.0
                        trade_notional = 0.0

                prev_val = portfolio_history[-1]
                current_val = max(100.0, portfolio_equity) # Floor to prevent negative balance
                period_return = (current_val - prev_val) / prev_val if prev_val > 0 else 0.0
                portfolio_history.append(current_val)
                returns_history.append(period_return)

            metrics = self.calculate_metrics(portfolio_history, returns_history)
            metrics["Total Strategy Trades"] = total_trades
            metrics["Winning Trades"] = winning_trades
            metrics["Losing Trades"] = losing_trades
            metrics["Win Rate (%)"] = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
            metrics["Total Fees & Friction Paid ($)"] = total_fees_paid
            
            gross_profit = sum(p for p in trade_pnls if p > 0)
            gross_loss = abs(sum(p for p in trade_pnls if p < 0))
            metrics["Profit Factor"] = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)
            metrics["Net Expectancy Per Trade (%)"] = np.mean(trade_pnls) * 100.0 if trade_pnls else 0.0

            logger.info("=== Leak-Free Real Historical Out-of-Sample Forward Test Results ===")
            for k, v in metrics.items():
                logger.info(f"{k}: {v:.2f}")
            return metrics

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
