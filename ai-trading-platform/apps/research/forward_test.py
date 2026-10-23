import sys
import os
if "." not in sys.path:
    sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from core.logging.logger import logger
from apps.research.historical_futures_loader import HistoricalFuturesLoader
from apps.research.model import SingleSymbolActorCritic, MultiSymbolActorCritic
from core.ai.memory import EventMemoryBuffer

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
        logger.info("=== Starting Production-Equivalent Forward Walk Simulation ===")
        
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
            from apps.feature_engine.engine import FeatureEngine
            
            logger.info("Fetching real Binance USDT-M Futures candles for leak-free forward walk...")
            df = self.loader.fetch_klines("BTCUSDT", interval="5m", limit=1000)
            if df.empty or len(df) < 300:
                raise ValueError("Insufficient real Binance historical candle data for forward testing.")
                
            # Chronological partitioning: 70% Train, 15% Val, 15% Untouched Test
            train_df, val_df, test_df = self.loader.split_chronological(df)
            logger.info(f"Chronological Real Data Partition: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            # Initialize production FeatureEngine for BTCUSDT
            feature_engine = FeatureEngine(symbol="BTCUSDT")
            
            # Warm up feature engine with train + validation bars chronologically so indicators (EMA, RSI, MACD) are fully settled
            logger.info("Warming up FeatureEngine across historical train/val bars...")
            warmup_df = pd.concat([train_df, val_df], ignore_index=True)
            for _, w_row in warmup_df.iterrows():
                p = float(w_row["close"])
                v = float(w_row.get("volume", 100.0))
                feature_engine.add_tick({
                    "mid_price": p,
                    "volume": v,
                    "buy_volume": float(w_row.get("taker_buy_base", v * 0.5)),
                    "trade_count": float(w_row.get("trades", 50.0)),
                    "best_bid": p * (1.0 - 0.0001),
                    "best_ask": p * (1.0 + 0.0001),
                    "bid_qty": 5.0,
                    "ask_qty": 5.0,
                    "funding_rate": 0.0001,
                    "open_interest": 10000.0,
                    "price_change_24h": 0.0,
                    "volume_24h": 10000.0,
                    "mark_price_premium": 0.0
                })
            
            # Financial Portfolio & Margin Accounting State
            wallet_balance = float(self.initial_capital)
            equity = wallet_balance
            peak_equity = wallet_balance
            
            position = 0.0          # Current position direction (-1.0 to +1.0)
            entry_price = 0.0       # Entry fill price
            trade_notional = 0.0    # Position notional ($)
            used_margin = 0.0       # Margin allocated ($)
            leverage = 5            # 5x leverage
            maintenance_margin_rate = 0.05 # 5% maintenance margin
            
            # Institutional Execution Frictions (Binance VIP-0 / Standard Futures)
            maker_fee_pct = 0.0002   # 0.02% maker fee
            taker_fee_pct = 0.0005   # 0.05% taker fee
            half_spread_pct = 0.00015 # 0.015% half-spread
            slippage_pct = 0.00010   # 0.010% slippage
            funding_rate_8h = 0.0001 # 0.01% standard 8-hour funding rate
            
            total_fees_paid = 0.0
            trade_pnls = []
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            
            portfolio_history = [wallet_balance]
            returns_history = []
            
            # Pending Order from previous bar's decision to execute at subsequent attainable price (next bar's open)
            pending_order = None # {"desired_pos": float, "decision_bar": int}
            
            test_rows = test_df.reset_index(drop=True)
            num_test_bars = len(test_rows)
            
            for i in range(num_test_bars):
                row = test_rows.iloc[i]
                cur_open = float(row["open"])
                cur_high = float(row["high"])
                cur_low = float(row["low"])
                cur_close = float(row["close"])
                cur_vol = float(row.get("volume", 100.0))
                
                # -------------------------------------------------------------
                # 1. Execute Pending Order at Subsequent Attainable Price (Bar Open)
                # -------------------------------------------------------------
                if pending_order is not None:
                    desired_pos = pending_order["desired_pos"]
                    
                    if desired_pos != position:
                        # A. Close existing position if any
                        if position != 0.0 and trade_notional > 0.0:
                            # Taker exit execution price including spread and slippage
                            if position > 0:
                                exit_exec_px = cur_open * (1.0 - half_spread_pct - slippage_pct)
                            else:
                                exit_exec_px = cur_open * (1.0 + half_spread_pct + slippage_pct)
                                
                            gross_trade_pnl = position * ((exit_exec_px - entry_price) / entry_price) * trade_notional
                            exit_fee = trade_notional * taker_fee_pct
                            net_trade_pnl = gross_trade_pnl - exit_fee
                            
                            wallet_balance += net_trade_pnl
                            total_fees_paid += exit_fee
                            trade_pnls.append(net_trade_pnl / used_margin if used_margin > 0 else 0.0)
                            total_trades += 1
                            if net_trade_pnl > 0:
                                winning_trades += 1
                            else:
                                losing_trades += 1
                                
                            position = 0.0
                            entry_price = 0.0
                            trade_notional = 0.0
                            used_margin = 0.0
                            
                        # B. Open new position if desired_pos != 0
                        if desired_pos != 0.0 and wallet_balance > 100.0:
                            # Maker entry execution price
                            if desired_pos > 0:
                                entry_exec_px = cur_open * (1.0 + half_spread_pct)
                            else:
                                entry_exec_px = cur_open * (1.0 - half_spread_pct)
                                
                            used_margin = wallet_balance * 0.20 # 20% margin budget
                            trade_notional = used_margin * leverage
                            entry_fee = trade_notional * maker_fee_pct
                            
                            wallet_balance -= entry_fee
                            total_fees_paid += entry_fee
                            
                            position = desired_pos
                            entry_price = entry_exec_px
                            
                    pending_order = None # Order filled
                    
                # -------------------------------------------------------------
                # 2. Mark Open Position to Market on Every Bar (Close Price)
                # -------------------------------------------------------------
                unrealized_pnl = 0.0
                if position != 0.0 and entry_price > 0.0 and trade_notional > 0.0:
                    unrealized_pnl = position * ((cur_close - entry_price) / entry_price) * trade_notional
                    
                    # 8-Hour Funding Rate Deduction (every 96 5m bars)
                    if (i + 1) % 96 == 0:
                        funding_cost = trade_notional * funding_rate_8h
                        wallet_balance -= funding_cost
                        total_fees_paid += funding_cost
                        
                    # Maintenance Margin / Liquidation Check
                    maintenance_margin = trade_notional * maintenance_margin_rate
                    current_equity = wallet_balance + unrealized_pnl
                    if current_equity <= maintenance_margin:
                        # Liquidation triggered!
                        logger.warning(f"Bar {i}: LIQUIDATION TRIGGERED on {position} position! Equity ${current_equity:.2f} <= Maintenance ${maintenance_margin:.2f}")
                        wallet_balance = max(10.0, current_equity - (trade_notional * 0.01)) # Liquidation penalty
                        position = 0.0
                        entry_price = 0.0
                        trade_notional = 0.0
                        used_margin = 0.0
                        unrealized_pnl = 0.0
                        total_trades += 1
                        losing_trades += 1
                        trade_pnls.append(-1.0)
                
                # Marked-to-market equity update
                equity = wallet_balance + unrealized_pnl
                if equity > peak_equity:
                    peak_equity = equity
                    
                prev_hist_val = portfolio_history[-1]
                portfolio_history.append(equity)
                ret_val = (equity - prev_hist_val) / prev_hist_val if prev_hist_val > 0 else 0.0
                returns_history.append(ret_val)
                
                # -------------------------------------------------------------
                # 3. Production Feature Builder: Feed Completed Bar into FeatureEngine
                # -------------------------------------------------------------
                feature_engine.add_tick({
                    "mid_price": cur_close,
                    "volume": cur_vol,
                    "buy_volume": float(row.get("taker_buy_base", cur_vol * 0.5)),
                    "trade_count": float(row.get("trades", 50.0)),
                    "best_bid": cur_close * (1.0 - 0.0001),
                    "best_ask": cur_close * (1.0 + 0.0001),
                    "bid_qty": 5.0,
                    "ask_qty": 5.0,
                    "funding_rate": 0.0001,
                    "open_interest": 10000.0,
                    "price_change_24h": 0.0,
                    "volume_24h": 10000.0,
                    "mark_price_premium": 0.0
                })
                market_features = feature_engine.compute_features() # Exactly 90 slots!
                
                # Construct Full Production 200-Dim State Vector
                # A. Portfolio & Risk (10 slots) - Normalized against base capital $5,000
                base_capital = 5000.0
                norm_wallet = float(np.clip(wallet_balance / base_capital, 0.0, 5.0))
                norm_equity = float(np.clip(equity / base_capital, 0.0, 5.0))
                free_margin = equity - used_margin  # was undefined (NameError on the first simulated step)
                norm_free_margin = float(np.clip(free_margin / equity if equity > 0 else 1.0, 0.0, 1.0))
                margin_util = used_margin / equity if equity > 0 else 0.0
                norm_exposure = float(np.clip(trade_notional / equity if equity > 0 else 0.0, 0.0, 10.0))
                account_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                win_rate_val = (winning_trades / total_trades) if total_trades > 0 else 0.0
                
                portfolio_block = [
                    norm_wallet, norm_equity, norm_free_margin, margin_util,
                    norm_exposure, 1.0 if position != 0 else 0.0,
                    win_rate_val, account_dd, 0.0, 0.0
                ]
                
                # B. Position Awareness (10 slots)
                current_pnl_pct = (unrealized_pnl / used_margin) if used_margin > 0 else 0.0
                dist_to_liq = 0.80 if position != 0 else 0.0
                entry_dist = (entry_price - cur_close) / cur_close if cur_close > 0 and entry_price > 0 else 0.0
                pos_signed_dir = 1.0 if position > 0 else (-1.0 if position < 0 else 0.0)
                position_block = [
                    pos_signed_dir, float(np.clip(entry_dist, -1.0, 1.0)), leverage / 50.0, current_pnl_pct,
                    dist_to_liq, max(0.0, current_pnl_pct), min(0.0, current_pnl_pct),
                    0.0, 0.0, 0.0
                ]
                
                # C. Macro & Cyclical Time (20 slots)
                bar_time = row["open_time"]
                minute_of_day = getattr(bar_time, "hour", 12) * 60 + getattr(bar_time, "minute", 0)
                time_sin = np.sin(2 * np.pi * minute_of_day / 1440.0)
                time_cos = np.cos(2 * np.pi * minute_of_day / 1440.0)
                day_w = getattr(bar_time, "dayofweek", 2)
                day_sin = np.sin(2 * np.pi * day_w / 7.0)
                day_cos = np.cos(2 * np.pi * day_w / 7.0)
                macro_block = [time_sin, time_cos, day_sin, day_cos] + [0.0] * 16
                
                # D. Whale & StatArb (70 slots)
                whale_statarb_block = [0.0] * 70
                
                full_state = portfolio_block + position_block + list(market_features) + macro_block + whale_statarb_block
                state_tensor = torch.tensor(full_state, dtype=torch.float32).unsqueeze(0)
                
                if state_tensor.shape[1] != 200:
                    raise ValueError(f"State dimension mismatch! Expected 200, got {state_tensor.shape[1]}")
                    
                # -------------------------------------------------------------
                # 4. Production Model Inference (Decision made at Bar Close)
                # -------------------------------------------------------------
                with torch.no_grad():
                    action_preds, critics = production_model(state_tensor)
                    raw_act = float(action_preds[0, 0].item())
                    conf = float(np.clip((action_preds[0, 1].item() + 1.0) / 2.0, 0.05, 0.99))
                    pred_edge = float(critics[0, 0].item()) if critics is not None and critics.numel() > 0 else 0.0

                # Production Action Contract: Conviction (|act| > 0.20) and Confidence (>= 0.65)
                next_desired_pos = 0.0
                if raw_act > 0.20 and conf >= 0.65:
                    next_desired_pos = 1.0   # LONG
                elif raw_act < -0.20 and conf >= 0.65:
                    next_desired_pos = -1.0  # SHORT
                else:
                    # In hysteresis, if already in a trade, hold unless exit condition or reversal
                    next_desired_pos = position

                # Schedule execution for subsequent attainable price (bar i+1 open)
                if i + 1 < num_test_bars and next_desired_pos != position:
                    pending_order = {"desired_pos": next_desired_pos, "decision_bar": i}

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

            logger.info("=== Leak-Free Production-Equivalent Historical Out-of-Sample Results ===")
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
