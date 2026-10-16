import asyncio
import json
import uuid
import torch
import numpy as np
import time
import os
import random
import collections
import datetime
import msgpack  # Phase 1: Binary compression for experience messages
from typing import Dict, Any

from core.db.redis import redis_manager
from core.ai.registry import ModelRegistry
from apps.research.model import SingleSymbolActorCritic
from core.risk.risk_manager import RiskManager
from core.exchange.binance_client import BinanceFuturesAdapter
from core.exchange.symbol_registry import registry
from core.logging.logger import logger, set_log_file
from core.ai.memory import EventMemoryBuffer
from core.schemas.state_schema import MarketState, PortfolioState, PositionState, MacroState, OrderRequest
from core.config.settings import settings

class ProductionTradingBot:
    """
    Phase 1: Safe Execution Engine with Canonical Schemas.
    Live execution engine for the fully upgraded RL model.
    """
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.num_symbols = len(symbols)
        
        if self.symbols != settings.symbol_universe:
            raise RuntimeError(f"Symbols mismatch: {self.symbols} != {settings.symbol_universe}")
            
        self.redis = redis_manager
        self.risk_manager = RiskManager()
        self.registry = ModelRegistry()
        self.binance = BinanceFuturesAdapter()
        
        # Phase 1: Complete Safety Gates Configured in settings
        self.dry_run = settings.dry_run
        self.trading_enabled = settings.trading_enabled
        
        if not self.dry_run and not settings.allow_live_trading and not settings.allow_testnet:
            logger.error("SAFETY GATE: Live/Testnet trading disabled by settings.")
            self.dry_run = True
            
        if settings.emergency_stop:
            logger.critical("SAFETY GATE: EMERGENCY STOP ACTIVE")
            self.trading_enabled = False
            self.dry_run = True
            
        logger.info(f"Trading Mode: {'DRY_RUN' if self.dry_run else 'LIVE_OR_TESTNET'} | Enabled: {self.trading_enabled}")
        
        # 4. Strict Model Checkpoint Loading (graceful if no checkpoint yet)
        self.has_valid_model = False
        self.model = SingleSymbolActorCritic(input_dim=200)
        
        # VPS CPU FIX: Restrict PyTorch inference to 1 core so the trading bot doesn't hog the host CPU
        torch.set_num_threads(1)
        
        try:
            # This calls the strictly validated loader that checks architecture and active status
            self.model = self.registry.load_model(self.model)
            self.model.eval()
            self.has_valid_model = True
            logger.info(f"Loaded rigorously validated SingleSymbolActorCritic for {self.num_symbols} symbols.")
        except Exception as e:
            logger.warning(f"No trained model checkpoint found yet. Proceeding with UNTRAINED model for exploration to bootstrap experiences. Error: {e}")
            self.model.eval()
            self.has_valid_model = True # Treat the randomly initialized model as valid so we can explore!
        
        self.running = False
        self.emergency_stop = False  # BUG FIX: was missing, caused AttributeError in monitor_kill_switch
        self.event_memory = EventMemoryBuffer()
        
        # Tracking live states using Canonical Schemas
        # SIMULATION ONLY: Safe zero account state
        self.portfolio_state = PortfolioState(
            wallet_balance=0.0, equity=0.0, used_margin=0.0,
            free_margin=0.0, total_unrealized_pnl=0.0, total_exposure=0.0,
            positions={sym: PositionState(symbol=sym) for sym in self.symbols}
        )
        self.macro_state = MacroState(timestamp=time.time())
        self.market_states: Dict[str, MarketState] = {}
        self.whale_states: Dict[str, dict] = {}
        self.statarb_states: Dict[str, dict] = {}
        
        # Phase 12: LSTM Sliding Window Sequence (300 steps = 5 minutes)
        self.sequence_length = 300
        self.state_history = collections.defaultdict(lambda: collections.deque(maxlen=self.sequence_length))
        
        # LSTM True Memory State Tracking
        self.hidden_states = {sym: None for sym in self.symbols}
        
        # Limit Order Tracking
        self.active_orders = {} # symbol -> {'order_id': str, 'timestamp': float}
        
        # Self-Awareness Tracking
        self.session_trades = 0
        self.session_wins = 0
        self.session_win_rate = 0.0
        
        self.max_equity = 0.0
        self.last_win_time = time.time()
        self.equity_history = collections.deque(maxlen=60) # Last 60 ticks (5 mins)
        self.last_close_time = {sym: 0.0 for sym in self.symbols} # Cooldown tracking per symbol
        self.position_entry_times = {sym: 0.0 for sym in self.symbols} # Minimum holding tracking per symbol
        self.peak_pnl_pct = {sym: 0.0 for sym in self.symbols} # Peak leveraged PnL tracking for trailing profit locks
        self.pnl_history = {sym: collections.deque(maxlen=30) for sym in self.symbols} # (timestamp, pnl_pct_leveraged) for velocity shock detection
        self.smoothed_action_history = {sym: collections.deque(maxlen=3) for sym in self.symbols} # Institutional signal smoothing (3-tick window)
        self.smoothed_confidence_history = {sym: collections.deque(maxlen=3) for sym in self.symbols}
        
        # Institutional Scale-Out & Free-Roll Runner Tracking
        self.partial_tp_triggered = {sym: False for sym in self.symbols}
        
        # Institutional Circuit Breaker (5% Daily Loss Kill Switch) & Profit Ratchet
        self.daily_start_equity = 0.0
        self.daily_peak_equity = 0.0
        self.daily_equity_reset_time = time.time()
        self.circuit_breaker_active_until = 0.0
        self.profit_ratchet_active = False
        
    async def listen_macro(self):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe("macro:state:global")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    self.macro_state.sentiment_score = payload.get("sentiment_score", 0.0)
                    self.macro_state.volatility_expectation = payload.get("volatility_expectation", 0.5)
                    self.macro_state.regime = payload.get("regime", 0.0)
                    self.macro_state.sp500_momentum = payload.get("sp500_momentum", 0.0)
                    self.macro_state.dxy_momentum = payload.get("dxy_momentum", 0.0)
                    self.macro_state.vix_momentum = payload.get("vix_momentum", 0.0)
                    self.macro_state.gold_momentum = payload.get("gold_momentum", 0.0)
                    self.macro_state.treasury_yield_momentum = payload.get("treasury_yield_momentum", 0.0)
                    self.macro_state.ndx_momentum = payload.get("ndx_momentum", 0.0)
                    self.macro_state.defi_tvl_momentum = payload.get("defi_tvl_momentum", 0.0)
                    self.macro_state.fear_greed_index = payload.get("fear_greed_index", 0.5)
                    self.macro_state.timestamp = time.time()
                except Exception as e:
                    logger.error(f"Error parsing macro: {e}")
                    
    async def listen_market(self, symbol: str):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe(f"market:state:{symbol}")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    if float(payload.get("mid_price", 0.0)) <= 0.0:
                        continue # Wait until market collector has real price data
                    # Parse into Canonical MarketState
                    self.market_states[symbol] = MarketState(**payload)
                except Exception as e:
                    logger.error(f"Error parsing market state for {symbol}: {e}")
                    
    async def listen_whale(self, symbol: str):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe(f"whale:state:{symbol}")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    self.whale_states[symbol] = payload
                except Exception as e:
                    logger.error(f"Error parsing whale state for {symbol}: {e}")

    async def listen_statarb(self, symbol: str):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe(f"statarb:state:{symbol}")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    self.statarb_states[symbol] = payload
                except Exception as e:
                    logger.error(f"Error parsing statarb state for {symbol}: {e}")

                    
    def _build_state_vector(self, symbol: str) -> tuple[torch.Tensor, list]:
        obs = []
        
        # 1. Portfolio & Risk (10 dims)
        equity = self.portfolio_state.equity
        free_margin = self.portfolio_state.free_margin
        margin_utilization = (equity - free_margin) / equity if equity > 0 else 0.0
        
        # Advanced Risk: Portfolio Heat
        active_positions = sum(1 for p in self.portfolio_state.positions.values() if p.quantity != 0)
        portfolio_heat = active_positions / self.num_symbols if self.num_symbols > 0 else 0.0
        
        # Meta-Learning & StatArb Internal Metrics
        if equity > self.max_equity:
            self.max_equity = equity
        account_drawdown = (self.max_equity - equity) / self.max_equity if self.max_equity > 0 else 0.0
        
        self.equity_history.append(equity)
        import numpy as np
        portfolio_variance = float(np.var(self.equity_history) / equity) if equity > 0 and len(self.equity_history) > 1 else 0.0
        time_since_win_hrs = (time.time() - self.last_win_time) / 3600.0 if self.last_win_time > 0 else 0.0
        
        obs.extend([
            self.portfolio_state.wallet_balance, 
            equity,
            free_margin,
            margin_utilization,
            self.portfolio_state.total_exposure,
            float(portfolio_heat),
            float(self.session_win_rate),
            float(account_drawdown), 
            float(portfolio_variance), 
            float(time_since_win_hrs)
        ])
        
        # 2. Position Awareness (10 dims)
        pos = self.portfolio_state.positions[symbol]
        market = self.market_states.get(symbol)
        
        time_held_hours = 0.0
        entry_dist_vwap = 0.0
        current_pnl_pct = 0.0
        dist_to_liq = 0.0
        
        if pos.quantity != 0 and pos.entry_price > 0 and market:
            raw_pnl = (market.mid_price - pos.entry_price) / pos.entry_price
            current_pnl_pct = raw_pnl if pos.quantity > 0 else -raw_pnl
            
            # High & Low Water Mark tracking
            if current_pnl_pct > pos.max_unrealized_pnl:
                pos.max_unrealized_pnl = current_pnl_pct
            if current_pnl_pct < pos.max_drawdown_pnl:
                pos.max_drawdown_pnl = current_pnl_pct
                
            if pos.liquidation_price > 0:
                dist_to_liq = abs(market.mid_price - pos.liquidation_price) / market.mid_price
                
            time_held_hours = (time.time() - pos.entry_time) / 3600.0 if pos.entry_time > 0 else 0.0
            if market.vwap > 0:
                entry_dist_vwap = (pos.entry_price - market.vwap) / market.vwap
        else:
            # Reset High Water Mark when flat
            pos.max_unrealized_pnl = 0.0
            pos.max_drawdown_pnl = 0.0
            pos.entry_time = 0.0
                
        obs.extend([
            float(pos.quantity),
            float(pos.entry_price),
            float(pos.leverage) / 50.0, # normalized
            current_pnl_pct,
            dist_to_liq,
            # Fully utilizing the 10-slot Position Block
            float(pos.max_unrealized_pnl), 
            float(pos.max_drawdown_pnl),
            float(time_held_hours),
            float(entry_dist_vwap),
            float(pos.accumulated_funding)
        ])
        
        # 3. Market Features (90 dims) from engine.py
        if market and len(market.features) == 90:
            obs.extend(market.features)
        else:
            obs.extend([0.0] * 90)
            
        # 4. Macro & Cross-Asset (20 dims)
        current_hour = time.localtime().tm_hour
        current_min = time.localtime().tm_min
        minute_of_day = current_hour * 60 + current_min
        time_sin = np.sin(2 * np.pi * minute_of_day / 1440.0)
        time_cos = np.cos(2 * np.pi * minute_of_day / 1440.0)
        
        day_w = time.localtime().tm_wday
        day_sin = np.sin(2 * np.pi * day_w / 7.0)
        day_cos = np.cos(2 * np.pi * day_w / 7.0)
        
        btc_market = self.market_states.get("BTCUSDT")
        btc_mom_1m = btc_market.features[2] if btc_market and len(btc_market.features) == 90 else 0.0
        btc_mom_15m = btc_market.features[4] if btc_market and len(btc_market.features) == 90 else 0.0
        
        # Cross-Coin Beta (Relative Strength to BTC)
        sym_mom_1m = market.features[2] if market and len(market.features) == 90 else 0.0
        cross_coin_beta = sym_mom_1m / btc_mom_1m if abs(btc_mom_1m) > 1e-6 else 0.0
        
        macro_features = [
            float(time_sin), float(time_cos), float(day_sin), float(day_cos), float(cross_coin_beta), 0.0,
            float(self.macro_state.sentiment_score),
            float(self.macro_state.volatility_expectation),
            float(self.macro_state.regime),
            btc_mom_1m, btc_mom_15m, 
            float(self.macro_state.sp500_momentum), 
            float(self.macro_state.dxy_momentum), 
            float(self.macro_state.fear_greed_index),
            float(self.macro_state.vix_momentum),
            float(self.macro_state.gold_momentum),
            float(self.macro_state.treasury_yield_momentum),
            float(self.macro_state.ndx_momentum),
            float(self.macro_state.defi_tvl_momentum), 
            0.0
        ]
        obs.extend(macro_features)
        
        # 5. Future-Proofing Canvas: Whale & StatArb Injection (70 dims)
        whale = self.whale_states.get(symbol, {})
        statarb = self.statarb_states.get(symbol, {})
        
        # Institutional Whale Order Flow Normalization:
        # Prevent injecting raw 8-figure dollar values (e.g. $124M) into the LSTM neural network!
        w_buy_raw = float(whale.get("whale_buy_pressure", 0.0))
        w_sell_raw = float(whale.get("whale_sell_pressure", 0.0))
        total_whale_vol = w_buy_raw + w_sell_raw
        whale_imbalance = (w_buy_raw - w_sell_raw) / (total_whale_vol + 1e-5) if total_whale_vol > 0 else 0.0
        whale_vol_norm = float(np.clip(np.log1p(total_whale_vol) / 16.0, 0.0, 1.0))
        
        obs.extend([
            # Whale Tracking (Slots 130 - 139)
            float(whale.get("buy_wall_distance", 0.0)),
            float(whale.get("sell_wall_distance", 0.0)),
            float(whale_imbalance), # Normalized [-1.0, 1.0] Order Flow Imbalance
            float(whale_vol_norm),  # Normalized [0.0, 1.0] Log Volume
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            
            # StatArb Tracking (Slots 140 - 149)
            float(statarb.get("okx_premium", 0.0)),
            float(statarb.get("bybit_premium", 0.0)),
            float(statarb.get("okx_momentum_lead", 0.0)),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        
        # Fill remaining 50 slots with zeros (150-199)
        obs.extend([0.0] * 50)
        
        if len(obs) != 200:
            raise ValueError(f"State vector dimension mismatch. Expected 200, got {len(obs)}")
            
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0), macro_features
                
    async def sync_dashboard(self):
        """Periodically fetches live positions from Binance and publishes to dashboard."""
        while self.running:
            try:
                if self.trading_enabled and not self.dry_run:
                    # Update live equity
                    usdt = await self.binance.get_account_details()
                    self.portfolio_state.wallet_balance = float(usdt.get("balance", 0.0))
                    self.portfolio_state.equity = float(usdt.get("crossWalletBalance", self.portfolio_state.wallet_balance)) + float(usdt.get("crossUnPnl", 0.0))
                    self.portfolio_state.free_margin = float(usdt.get("availableBalance", self.portfolio_state.wallet_balance))
                    
                    # Update live positions
                    live_positions = await self.binance.get_positions()
                    formatted_positions = []
                    # Update live positions without zeroing race conditions
                    active_synced_symbols = set()
                    for p in live_positions:
                        sym = p.get("symbol")
                        amt = float(p.get("positionAmt", 0))
                        
                        # Sync back to internal portfolio state
                        if sym in self.portfolio_state.positions:
                            active_synced_symbols.add(sym)
                            self.portfolio_state.positions[sym].quantity = amt
                            self.portfolio_state.positions[sym].leverage = int(p.get("leverage", 10))
                            if amt != 0:
                                self.portfolio_state.positions[sym].entry_price = float(p.get("entryPrice", 0))
                                # MEMORY RESILIENCE: If position is open but entry time was not tracked (e.g. after PM2 restart),
                                # recover the exact entry timestamp from Binance updateTime so the 600s breathing room stays active!
                                if self.position_entry_times.get(sym, 0.0) == 0.0:
                                    pos_update_ms = float(p.get("updateTime", 0.0))
                                    if pos_update_ms > 0:
                                        self.position_entry_times[sym] = pos_update_ms / 1000.0
                                        self.portfolio_state.positions[sym].entry_time = pos_update_ms / 1000.0
                                        logger.info(f"[{sym}] 🔄 RECOVERED ENTRY TIME from Binance: {time.time() - (pos_update_ms / 1000.0):.1f}s ago")
                                    else:
                                        self.position_entry_times[sym] = time.time()
                                        self.portfolio_state.positions[sym].entry_time = time.time()
                            else:
                                self.peak_pnl_pct[sym] = 0.0
                                self.pnl_history[sym].clear()
                                self.position_entry_times[sym] = 0.0
                                self.portfolio_state.positions[sym].entry_price = 0.0
                                self.partial_tp_triggered[sym] = False
                                
                        # CRITICAL: Only send active positions to the dashboard (once per active position)
                        if abs(amt) > 0:
                            pnl = float(p.get("unRealizedProfit", 0))
                            formatted_positions.append({
                                "symbol": sym,
                                "side": "LONG" if amt > 0 else "SHORT",
                                "quantity": abs(amt),
                                "pnl": pnl,
                                "entryPrice": float(p.get("entryPrice", 0)),
                                "markPrice": float(p.get("markPrice", 0)),
                                "leverage": int(p.get("leverage", 1)),
                                "liquidationPrice": float(p.get("liquidationPrice", 0)),
                                "marginType": p.get("marginType", "cross")
                            })

                    # Zero out any symbols absent from Binance response
                    for s in self.symbols:
                        if s in self.portfolio_state.positions and s not in active_synced_symbols:
                            self.portfolio_state.positions[s].quantity = 0.0
                            self.portfolio_state.positions[s].entry_price = 0.0
                            self.partial_tp_triggered[s] = False
                    
                    if self.redis.redis:
                        payload = {
                            "equity": self.portfolio_state.equity,
                            "positions": formatted_positions
                        }
                        await self.redis.redis.set("dashboard:portfolio", json.dumps(payload))
                        
            except Exception as e:
                logger.error(f"Dashboard sync failed: {e}")
            await asyncio.sleep(5.0)

    async def inference_loop(self):
        logger.info("Starting production inference loop...")
        while self.running:
            try:
                # 0. Fast Authoritative Redis Kill Switch & Emergency Stop Check
                ks_active = False
                try:
                    if self.redis.redis:
                        ks = await self.redis.redis.get("system:kill_switch:active")
                        val = (ks.decode("utf-8") if isinstance(ks, bytes) else str(ks or "")).strip().lower()
                        if val in ["true", "1", "yes"]:
                            ks_active = True
                except Exception as ks_err:
                    logger.error(f"Failed to check Redis kill switch in inference loop: {ks_err}")

                if ks_active or self.emergency_stop or settings.emergency_stop:
                    if not self.emergency_stop or not settings.emergency_stop:
                        logger.critical("🚨 EMERGENCY KILL SWITCH DETECTED IN INFERENCE LOOP: Halting trading immediately!")
                        self.emergency_stop = True
                        self.trading_enabled = False
                        settings.emergency_stop = True
                        await self.panic_liquidate()
                    await asyncio.sleep(3.0)
                    continue

                # 0. Hot Reload Model
                if self.registry.check_for_updates():
                    logger.info("New model version detected. Hot reloading...")
                    self.model = self.registry.load_model(self.model)
                    self.model.eval()
                    logger.info("Hot reload complete.")

                # 24-Hour Rolling Circuit Breaker Tracking (5% Daily Loss Kill Switch) & Profit Ratchet
                now_t = time.time()
                current_eq = float(self.portfolio_state.equity)
                if (now_t - self.daily_equity_reset_time) >= 86400.0 or self.daily_start_equity <= 0.0:
                    if current_eq > 0:
                        self.daily_start_equity = current_eq
                        self.daily_peak_equity = current_eq
                        self.daily_equity_reset_time = now_t
                        self.profit_ratchet_active = False
                        logger.info(f"📅 24-Hour Equity Benchmark Reset: Starting Equity = ${self.daily_start_equity:.2f}")

                if self.daily_start_equity > 0 and current_eq > 0:
                    # Update rolling daily high-water peak equity
                    if current_eq > self.daily_peak_equity:
                        self.daily_peak_equity = current_eq

                    # 1. 5% Daily Loss Circuit Breaker Kill Switch
                    daily_dd = (self.daily_start_equity - current_eq) / self.daily_start_equity
                    if daily_dd >= 0.05 and now_t > self.circuit_breaker_active_until:
                        self.circuit_breaker_active_until = now_t + (6 * 3600.0) # 6 hours standby
                        logger.critical(f"🚨 5% DAILY LOSS CIRCUIT BREAKER TRIPPED! Daily drawdown is {daily_dd*100:.2f}% >= 5.00%. Pausing new entries for 6 hours!")

                    # 2. Daily High-Water Profit Ratchet (Session Profit Lock)
                    # If daily peak profit reached >= +4.0% (e.g. +$180 on $4,500), protect accumulated daily profits!
                    # If current profit retraces by >= 50% from daily peak profit, arm Capital Preservation Mode.
                    peak_profit = self.daily_peak_equity - self.daily_start_equity
                    current_profit = current_eq - self.daily_start_equity
                    if peak_profit >= (0.040 * self.daily_start_equity):
                        if current_profit < (0.50 * peak_profit) and not self.profit_ratchet_active:
                            self.profit_ratchet_active = True
                            logger.warning(
                                f"🛡️ DAILY PROFIT RATCHET ARMED (Capital Preservation Mode): Daily peak profit was +${peak_profit:.2f} "
                                f"({(peak_profit/self.daily_start_equity)*100:.2f}%), but current profit retraced to +${current_profit:.2f}. "
                                f"Halving base margin to 5.0% and elevating confidence hurdle to 74% to permanently preserve winning day!"
                            )
                        elif current_profit >= (0.75 * peak_profit) and self.profit_ratchet_active:
                            self.profit_ratchet_active = False
                            logger.info(f"🛡️ Daily Profit Ratchet disarmed: Equity recovered to +${current_profit:.2f} (>= 75% of peak). Normal sizing restored.")

                # Institutional Liquidity Clock & Session Volatility Regimes
                utc_now = datetime.datetime.now(datetime.timezone.utc)
                is_weekend = (utc_now.weekday() >= 5) # Saturday (5) or Sunday (6)
                utc_hour = utc_now.hour + (utc_now.minute / 60.0)
                
                if is_weekend:
                    session_name = "WEEKEND_LOW_LIQUIDITY"
                    session_conf_hurdle = 0.72
                    session_max_spread = 0.00045 # Tight 0.045% limit in low liquidity
                    session_critic_hurdle = 0.42
                elif 13.5 <= utc_hour < 21.0: # US New York Peak Trend & Volatility (13:30 - 21:00 UTC)
                    session_name = "US_NEW_YORK_PEAK"
                    session_conf_hurdle = 0.65
                    session_max_spread = 0.00060 # Standard 0.060%
                    session_critic_hurdle = 0.35
                elif 8.0 <= utc_hour < 13.5: # London Morning & Pre-NY (08:00 - 13:30 UTC)
                    session_name = "LONDON_TREND"
                    session_conf_hurdle = 0.65
                    session_max_spread = 0.00060
                    session_critic_hurdle = 0.35
                else: # Asia Off-Peak (00:00 - 08:00 UTC) Mean-Reversion Chop
                    session_name = "ASIA_OFF_PEAK_CHOP"
                    session_conf_hurdle = 0.70
                    session_max_spread = 0.00050 # 0.050%
                    session_critic_hurdle = 0.42

                # Phase 2: Batched Inference Pre-Pass
                # Collect ALL valid symbol state vectors and run ONE model call instead of 6.
                # SAFETY SHIELD: Symbols with wrong shape are silently skipped (fall back to individual inference).
                precomputed_actions = {}  # {sym: action_logits_np_array}
                precomputed_critics = {}  # {sym: critic_pred_np_array}
                precomputed_seqs = {}     # {sym: (state_vector_tensor, seq_list)} — cache to avoid double build
                
                batch_tensors = []
                batch_syms = []
                
                for sym in self.symbols:
                    if sym not in self.market_states:
                        continue
                    if sym in self.active_orders:
                        continue  # Handled inside main loop
                    try:
                        sv, mf = self._build_state_vector(sym)
                        sv_list = sv.squeeze(0).tolist()
                        # Build seq using current history + new frame (peek-ahead)
                        seq = list(self.state_history[sym]) + [sv_list]
                        while len(seq) < self.sequence_length:
                            seq.insert(0, seq[0])
                        seq = seq[-self.sequence_length:]
                        
                        st = torch.tensor(seq, dtype=torch.float32)
                        # SAFETY SHIELD: Only batch if shape is exactly right
                        if st.shape == (self.sequence_length, 200):
                            batch_tensors.append(st)
                            batch_syms.append(sym)
                            precomputed_seqs[sym] = (sv, seq)
                        else:
                            logger.warning(f"[{sym}] Skipping batch inference: shape {st.shape}")
                    except Exception as e:
                        logger.warning(f"[{sym}] Batch pre-pass failed: {e}")
                
                if batch_tensors:
                    try:
                        final_batch = torch.stack(batch_tensors)  # [N, seq_len, 200]
                        final_batch = torch.nan_to_num(final_batch, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # Build batched hidden state
                        h_list, c_list = [], []
                        has_hidden = False
                        hidden_dim = getattr(self.model.lstm, 'hidden_size', 256) if hasattr(self.model, 'lstm') else 256
                        for sym in batch_syms:
                            h = self.hidden_states.get(sym)
                            if h is not None:
                                h_list.append(h[0])
                                c_list.append(h[1])
                                has_hidden = True
                            else:
                                h_list.append(torch.zeros(1, 1, hidden_dim, device=final_batch.device))
                                c_list.append(torch.zeros(1, 1, hidden_dim, device=final_batch.device))
                        
                        batch_hidden = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1)) if has_hidden else None

                        with torch.no_grad():
                            batch_logits, batch_critic, new_hidden_batch = self.model(final_batch, hidden_state=batch_hidden, return_hidden=True)
                        for i, sym in enumerate(batch_syms):
                            precomputed_actions[sym] = batch_logits[i].numpy()
                            precomputed_critics[sym] = batch_critic[i].numpy()
                            # Save independent hidden state for next tick
                            self.hidden_states[sym] = (new_hidden_batch[0][:, i:i+1, :].detach(), new_hidden_batch[1][:, i:i+1, :].detach())
                    except Exception as e:
                        logger.warning(f"Batch inference failed, falling back to individual: {e}")
                        precomputed_actions = {}  # Force fallback
                        precomputed_critics = {}
                
                for sym in self.symbols:
                    if sym not in self.market_states:
                        continue

                        
                    # Check for pending Limit Order
                    if sym in self.active_orders:
                        time_elapsed = time.time() - self.active_orders[sym]['timestamp']
                        if time_elapsed > 30:
                            logger.info(f"[{sym}] POST-ONLY MAKER TIMEOUT (> 30s). Cancelling unfilled order...")
                            try:
                                await self.binance.cancel_order(sym, self.active_orders[sym]['order_id'])
                            except Exception as e:
                                logger.warning(f"[{sym}] Failed to cancel (maybe already filled/cancelled): {e}")
                            # Revert optimistic portfolio update since order timed out and didn't fill
                            if 'trade_record' in self.active_orders[sym]:
                                side = self.active_orders[sym]['trade_record']['side']
                                qty = float(self.active_orders[sym]['trade_record']['quantity'])
                                price = float(self.active_orders[sym]['trade_record']['price'])
                                margin_used = (qty * price) / settings.max_leverage
                                
                                if "OPEN_LONG" in side or "CLOSE_SHORT" in side:
                                    self.portfolio_state.positions[sym].quantity -= qty
                                elif "CLOSE_LONG" in side or "OPEN_SHORT" in side:
                                    self.portfolio_state.positions[sym].quantity += qty
                                    
                                if abs(self.portfolio_state.positions[sym].quantity) < 1e-5:
                                    self.portfolio_state.positions[sym].quantity = 0.0
                                    self.portfolio_state.positions[sym].entry_price = 0.0
                                    self.position_entry_times[sym] = 0.0
                                    
                                if "OPEN" in side:
                                    self.portfolio_state.free_margin += margin_used
                                elif "CLOSE" in side:
                                    self.portfolio_state.free_margin -= margin_used
                            del self.active_orders[sym]

                            continue
                        else:
                            # Order is still active. Verify if it's already filled via Binance API
                            try:
                                open_orders = await self.binance.get_open_orders(sym)
                                # orderId from binance could be int or str, we check both string casts
                                if not any(str(o.get('orderId')) == str(self.active_orders[sym]['order_id']) or o.get('clientOrderId') == self.active_orders[sym]['order_id'] for o in open_orders):
                                    logger.info(f"[{sym}] LIMIT ORDER FILLED/NO LONGER OPEN! Removing from tracker.")
                                    
                                    # CRITICAL FIX: Publish to dashboard history ONLY WHEN FILLED!
                                    if 'trade_record' in self.active_orders[sym] and self.redis.redis:
                                        self.active_orders[sym]['trade_record']['timestamp'] = time.time()
                                        await self.redis.redis.lpush("dashboard:trade_history", json.dumps(self.active_orders[sym]['trade_record']))
                                        await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)
                                        
                                    # CRITICAL FIX: Publish experience to Redis ONLY WHEN CONFIRMED FILLED!
                                    # Prevents phantom experiences from entering training buffer if cancelled!
                                    if 'experience_payload' in self.active_orders[sym]:
                                        await redis_manager.publish_binary(
                                            "experience:completed",
                                            self.active_orders[sym]['experience_payload'],
                                        )
                                        
                                    del self.active_orders[sym]
                                    
                                    # No need to sync position here. We optimistically updated it when the order was placed.
                                    # Binance REST API has eventual consistency delays, so querying it now might return stale data 
                                    # and accidentally resurrect a closed position.

                                    continue
                                else:
                                    logger.debug(f"[{sym}] Limit order still pending. Skipping tick.")
                                    continue
                            except Exception as e:
                                logger.warning(f"[{sym}] Error checking open orders: {e}")
                                continue
                        
                    # Phase 2: Use precomputed batched result if available
                    if sym in precomputed_actions and sym in precomputed_seqs:
                        # Use results from the batch pre-pass (no model call needed!)
                        state_vector, seq = precomputed_seqs[sym]
                        self.state_history[sym].append(state_vector.squeeze(0).tolist())
                        action_logits = precomputed_actions[sym]
                    else:
                        # Fallback: individual inference (for symbols skipped in pre-pass)
                        state_vector, macro_features = self._build_state_vector(sym)
                        self.state_history[sym].append(state_vector.squeeze(0).tolist())
                        seq = list(self.state_history[sym])
                        while len(seq) < self.sequence_length:
                            seq.insert(0, seq[0] if len(seq) > 0 else state_vector.squeeze(0).tolist())
                        state_tensor = torch.tensor([seq], dtype=torch.float32)
                        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                        with torch.no_grad():
                            action_logits_t, expected_return, new_h = self.model(state_tensor, hidden_state=self.hidden_states.get(sym), return_hidden=True)
                            action_logits = action_logits_t[0].numpy()
                            precomputed_critics[sym] = expected_return[0].numpy()
                            self.hidden_states[sym] = new_h

                        
                    market = self.market_states[sym]
                    
                    # Update portfolio state with current price for RiskManager correlated exposure checks
                    if sym in self.portfolio_state.positions:
                        self.portfolio_state.positions[sym].current_price = market.mid_price
                    # Calculate Real-Time PNL Percentage, Adaptive Trailing Stop, Sudden Shock Detector, and Hard Stop Loss
                    hard_stop_triggered = False
                    hard_take_profit_triggered = False
                    sudden_shock_triggered = False
                    trailing_stop_triggered = False
                    stale_trade_triggered = False
                    partial_tp_triggered = False
                    freeroll_stop_triggered = False
                    pnl_pct_leveraged = 0.0

                    if sym in self.portfolio_state.positions:
                        pos_check = self.portfolio_state.positions[sym]
                        if pos_check.quantity != 0 and pos_check.entry_price > 0:
                            now_t = time.time()
                            # PNL % = (Current - Entry) / Entry  (* -1 if short)
                            pnl_pct = (market.mid_price - pos_check.entry_price) / pos_check.entry_price
                            if pos_check.quantity < 0:
                                pnl_pct = -pnl_pct
                            
                            # Leverage amplifies PNL %
                            pnl_pct_leveraged = pnl_pct * pos_check.leverage
                            
                            # Record PnL time-series history for sudden shock / velocity reversal detection
                            if sym not in self.pnl_history:
                                self.pnl_history[sym] = collections.deque(maxlen=30)
                            self.pnl_history[sym].append((now_t, pnl_pct_leveraged))
                            
                            # Track Peak PnL for Dynamic Trailing
                            prev_peak = self.peak_pnl_pct.get(sym, 0.0)
                            if pnl_pct_leveraged > prev_peak:
                                self.peak_pnl_pct[sym] = pnl_pct_leveraged
                                prev_peak = pnl_pct_leveraged

                            is_in_freeroll = self.partial_tp_triggered.get(sym, False)
                            
                            # 0. FREE-ROLL BREAKEVEN STOP LOSS (Active ONLY after 50% Partial TP has been secured)
                            # Protects the remaining 50% free-roll runner at Breakeven +0.25% (covering all round-trip fees)
                            if is_in_freeroll and pnl_pct_leveraged <= 0.0025:
                                logger.info(f"[{sym}] 🔒 FREE-ROLL BREAKEVEN HIT: Price retraced to +{pnl_pct_leveraged*100:.2f}% (<= +0.25% fee floor). Closing remaining 50% with zero loss!")
                                freeroll_stop_triggered = True
                            
                            # 1. HARD STOP LOSS (-15.0% leveraged = -0.75% spot move at 20x leverage)
                            # Provides complete wick breathing room to survive liquidity sweeps and ride the swing!
                            elif pnl_pct_leveraged <= -0.15:
                                logger.warning(f"[{sym}] 🛑 HARD STOP LOSS TRIGGERED: Position is down {pnl_pct_leveraged*100:.2f}%. Overriding AI to protect capital.")
                                hard_stop_triggered = True
                                
                            # 1.5. HARD TAKE-PROFIT CEILING (+35.0% leveraged = +1.75% price surge at 20x leverage)
                            # Instantly locks in windfall gains at the candle peak before any pullback!
                            elif pnl_pct_leveraged >= 0.35:
                                logger.info(f"[{sym}] 🎯 HARD TAKE-PROFIT TRIGGERED: Massive surge to +{pnl_pct_leveraged*100:.2f}% leveraged! Snatching windfall profit immediately at wick peak.")
                                hard_take_profit_triggered = True

                            # 1.8. INSTITUTIONAL DYNAMIC 50% SCALE-OUT (TP1 at +12.0% leveraged gain)
                            # Takes profit on 50% of the position, banking guaranteed cash and arming the Free-Roll!
                            elif not is_in_freeroll and pnl_pct_leveraged >= 0.120:
                                logger.info(f"[{sym}] 💰 DYNAMIC 50% SCALE-OUT (TP1): Surged to +{pnl_pct_leveraged*100:.2f}% leveraged! Banking 50% profit and arming Free-Roll runner.")
                                partial_tp_triggered = True
                            
                            # 2. SUDDEN VELOCITY SHOCK DETECTOR (Catches flash spikes / wicks against winning trades)
                            elif prev_peak >= 0.04:
                                # Inspect PnL over the last 15 seconds
                                recent_window = [p for t, p in self.pnl_history[sym] if (now_t - t) <= 15.0]
                                if recent_window:
                                    recent_peak = max(recent_window)
                                    # If PnL dropped by >= 3.5% leveraged in <15s and lost over 30% of recent peak
                                    if (recent_peak - pnl_pct_leveraged) >= 0.035 and pnl_pct_leveraged < (recent_peak * 0.70):
                                        logger.warning(f"[{sym}] ⚡ SUDDEN VELOCITY SHOCK DETECTED: Rapid drop from +{recent_peak*100:.2f}% to +{pnl_pct_leveraged*100:.2f}% in <15s! Locking in profit immediately.")
                                        sudden_shock_triggered = True

                            # 3. ATR-ADAPTIVE VOLATILITY RATCHETING TRAILING STOP (True Micro-Swing Protection)
                            if not (hard_stop_triggered or hard_take_profit_triggered or sudden_shock_triggered or partial_tp_triggered or freeroll_stop_triggered):
                                coin_vol = float(market.volatility) if market and hasattr(market, 'volatility') and market.volatility > 0 else 0.015
                                # vol_factor: baseline 0.015 (1.5% volatility)
                                # Low vol (BTC/ETH ~0.8%) -> tighter retention (+3% to +5% retention) to bank cash fast
                                # High vol (NEAR/SOL/ADA ~3.0%) -> wider breathing room (-5% to -7% retention) to ride runners
                                vol_factor = float(np.clip((coin_vol - 0.015) * 4.0, -0.07, +0.07))
                                t1_retention = float(np.clip(0.70 - vol_factor, 0.60, 0.78))
                                t2_retention = float(np.clip(0.75 - vol_factor, 0.65, 0.82))
                                t3_retention = float(np.clip(0.85 - (vol_factor * 0.5), 0.80, 0.90))

                                # Tier 1: Micro-Swing Impulse Protection (Requires >= +8.0% peak to ensure fees & slippage are cleanly beaten)
                                if 0.08 <= prev_peak < 0.15:
                                    trailing_floor = max(0.04, prev_peak * t1_retention) # Guarantee at least +4.0% net gain above fees
                                    if pnl_pct_leveraged < trailing_floor:
                                        logger.info(f"[{sym}] 🔒 ATR-ADAPTIVE IMPULSE TRAIL (Vol: {coin_vol*100:.2f}%, Ret: {t1_retention*100:.1f}%): Peak +{prev_peak*100:.2f}%, now +{pnl_pct_leveraged*100:.2f}% < {trailing_floor*100:.2f}%. Banking solid swing gain!")
                                        trailing_stop_triggered = True
                                # Tier 2: Developing Trend (+15% to +30% peak)
                                elif 0.15 <= prev_peak < 0.30:
                                    trailing_floor = max(0.08, prev_peak * t2_retention) # Dynamic breathing room, min +8% floor
                                    if pnl_pct_leveraged < trailing_floor:
                                        logger.info(f"[{sym}] 🔒 ATR-ADAPTIVE TRAIL TRIGGERED (Developing Swing, Vol: {coin_vol*100:.2f}%, Ret: {t2_retention*100:.1f}%): Peak was +{prev_peak*100:.2f}%, current +{pnl_pct_leveraged*100:.2f}% < {trailing_floor*100:.2f}%. Securing gain!")
                                        trailing_stop_triggered = True
                                # Tier 3: Mature Deep Trend Runner (>= +30% peak, e.g. +50%, +100%+)
                                elif prev_peak >= 0.30:
                                    trailing_floor = prev_peak * t3_retention # Harvest 80-90% of peak gains
                                    if pnl_pct_leveraged < trailing_floor:
                                        logger.info(f"[{sym}] 🔒 ATR-ADAPTIVE TRAIL TRIGGERED (Deep Runner, Vol: {coin_vol*100:.2f}%, Ret: {t3_retention*100:.1f}%): Peak was +{prev_peak*100:.2f}%, current +{pnl_pct_leveraged*100:.2f}% < {trailing_floor*100:.2f}%. Securing massive gain!")
                                        trailing_stop_triggered = True

                            # 4. INSTITUTIONAL STALE POSITION PRUNER (Capital Recycling)
                            # If a position has stagnated for >= 90 mins (5,400s) and hovering within ±2.0% leveraged PnL,
                            # prune it to liberate used margin for active high-conviction breakout setups!
                            entry_time = self.position_entry_times.get(sym, 0.0)
                            if entry_time > 0:
                                holding_time = now_t - entry_time
                                if holding_time >= 5400.0 and abs(pnl_pct_leveraged) < 0.020:
                                    logger.info(f"[{sym}] ♻️ STALE TRADE PRUNER: Stagnant trade held {holding_time/60.0:.1f}m with flat PnL ({pnl_pct_leveraged*100:+.2f}%). Pruning to recycle capital!")
                                    stale_trade_triggered = True
                        else:
                            self.peak_pnl_pct[sym] = 0.0
                            if sym in self.pnl_history:
                                self.pnl_history[sym].clear()

                    exit_override = hard_stop_triggered or hard_take_profit_triggered or sudden_shock_triggered or trailing_stop_triggered or stale_trade_triggered or freeroll_stop_triggered or partial_tp_triggered
                    if exit_override:
                        # Force an immediate MARKET CLOSE
                        action_val = -1.0 if pos_check.quantity > 0 else 1.0
                        confidence = 1.0
                        target_size = 0.0
                        price_offset = 0.0 # Force immediate execution (0 offset)
                        self.smoothed_action_history[sym].clear()
                        self.smoothed_confidence_history[sym].clear()
                    else:
                        # Pure Deterministic Policy (Learned AI Weights - Zero Random Exploration Noise)
                        raw_action = float(action_logits[0])
                        confidence_temperature = 1.0
                        raw_conf = float(1.0 / (1.0 + np.exp(-action_logits[1] / confidence_temperature)))
                        
                        # 1. Institutional Temporal Window Smoothing (3-tick moving average)
                        # Eliminates high-frequency neural jitter and noisy 1-tick oscillations
                        self.smoothed_action_history[sym].append(raw_action)
                        self.smoothed_confidence_history[sym].append(raw_conf)
                        action_val = float(np.mean(self.smoothed_action_history[sym]))
                        confidence = float(np.mean(self.smoothed_confidence_history[sym]))
                        
                        raw_size = 1.0 / (1.0 + np.exp(-action_logits[2]))
                        target_size = max(0.25, min(1.0, raw_size))
                        price_offset = (action_logits[3] + 1.0) / 2.0
                    
                    pos = self.portfolio_state.positions[sym]
                    lead_lag_mult = 1.00
                    
                    # 2. Institutional Hysteresis Deadband
                    # Entries from flat require |action_val| > 0.20
                    # Reversals/Exits against an established position require passing a wider barrier (|action_val| > 0.35)
                    # This prevents 1-tick micro-wobbles from closing positions early into chop!
                    if pos.quantity > 0:
                        is_closing = (action_val < -0.35) or exit_override
                    elif pos.quantity < 0:
                        is_closing = (action_val > 0.35) or exit_override
                    else:
                        is_closing = False

                    # Predicted side tracking for telemetry
                    if action_val < -0.2: predicted_side = "SHORT"
                    elif action_val > 0.2: predicted_side = "LONG"
                    else: predicted_side = "HOLD"
                    
                    if self.redis.redis:
                        critic_vals = precomputed_critics.get(sym)
                        pred_ret_5m = float(critic_vals[0]) if critic_vals is not None and len(critic_vals) > 0 else 0.0
                        pred_ret_1h = float(critic_vals[1]) if critic_vals is not None and len(critic_vals) > 1 else pred_ret_5m
                        pred_ret_4h = float(critic_vals[2]) if critic_vals is not None and len(critic_vals) > 2 else pred_ret_1h
                        ai_state = {
                            "predicted_side": predicted_side,
                            "confidence": float(confidence),
                            "target_size": float(target_size),
                            "critic_return": float(pred_ret_5m),
                            "critic_1h": float(pred_ret_1h),
                            "critic_4h": float(pred_ret_4h)
                        }
                        await self.redis.redis.set(f"ai:state:{sym}", json.dumps(ai_state))

                    # GUARD 0.5: Circuit Breaker Check (5% Daily Drawdown Kill Switch)
                    if not is_closing and time.time() < self.circuit_breaker_active_until:
                        time_left_cb = (self.circuit_breaker_active_until - time.time()) / 3600.0
                        logger.warning(f"[{sym}] 🛑 CIRCUIT BREAKER ACTIVE: Halting new entries ({time_left_cb:.1f}h remaining). Preserving capital!")
                        continue

                    # GUARD 1: Anti-Churn Cooldown with High-Conviction Fast-Pass (180s Baseline, 120s Ironclad Rest)
                    if not is_closing:
                        time_since_close = time.time() - self.last_close_time.get(sym, 0.0)
                        cooldown_period = 180.0 # 3 minutes baseline cooldown
                        mandatory_rest_period = 120.0 # 2 minutes ironclad minimum rest
                        
                        if time_since_close < cooldown_period:
                            # 1. Ironclad Mandatory Rest: Zero trades permitted within 120s of close (kills 5-second flip-flop trap)
                            if time_since_close < mandatory_rest_period:
                                logger.info(f"[{sym}] SKIPPING OPEN: In mandatory post-trade rest ({time_since_close:.1f}s / {mandatory_rest_period:.0f}s). Cooling down book.")
                                continue
                                
                            # 2. High-Conviction Fast-Pass Override (only eligible between 120s and 180s)
                            critic_vals = precomputed_critics.get(sym)
                            pred_ret_5m = float(critic_vals[0]) if critic_vals is not None and len(critic_vals) > 0 else 0.0
                            fast_pass_critic_hurdle = 0.85 # Raised to 0.85% move required to bypass remaining 60s
                            has_critic_conviction = (action_val > 0.2 and pred_ret_5m >= fast_pass_critic_hurdle) or (action_val < -0.2 and pred_ret_5m <= -fast_pass_critic_hurdle)
                            
                            if confidence >= 0.78 and has_critic_conviction:
                                logger.info(f"[{sym}] 🎯 HIGH-CONVICTION FAST-PASS: Bypassing remaining cooldown ({time_since_close:.1f}s/{cooldown_period:.0f}s) due to extreme confidence ({confidence:.2f} >= 0.78) & massive Critic return ({pred_ret_5m:+.2f}% >= {fast_pass_critic_hurdle:.2f}%)!")
                            else:
                                logger.info(f"[{sym}] SKIPPING OPEN: In anti-churn cooldown ({time_since_close:.1f}s / {cooldown_period:.0f}s).")
                                continue

                    # GUARD 1.2: Net Portfolio Beta & Correlation Exposure Governor
                    # Caps simultaneous unidirectional open positions to max 3 Longs or 3 Shorts,
                    # and restricts net directional margin exposure to <= 30% of account equity
                    if not is_closing and pos.quantity == 0:
                        open_longs = [s for s, p in self.portfolio_state.positions.items() if p.quantity > 1e-5]
                        open_shorts = [s for s, p in self.portfolio_state.positions.items() if p.quantity < -1e-5]
                        
                        long_margin = sum((p.quantity * self.market_states[s].mid_price / p.leverage)
                                          for s, p in self.portfolio_state.positions.items()
                                          if p.quantity > 1e-5 and s in self.market_states and self.market_states[s].mid_price > 0)
                        short_margin = sum((abs(p.quantity) * self.market_states[s].mid_price / p.leverage)
                                           for s, p in self.portfolio_state.positions.items()
                                           if p.quantity < -1e-5 and s in self.market_states and self.market_states[s].mid_price > 0)
                        
                        net_exposure_ceiling = max(800.0, self.portfolio_state.equity * 0.50) # 50% of equity (min $800 safety buffer)
                        
                        if action_val > 0.20:
                            if len(open_longs) >= 3:
                                logger.info(f"[{sym}] 🛑 PORTFOLIO BETA GOVERNOR: Maximum concurrent Longs reached ({len(open_longs)}/3: {open_longs}). Blocking new Long to prevent correlated drawdown!")
                                continue
                            if (long_margin - short_margin) >= net_exposure_ceiling:
                                logger.info(f"[{sym}] 🛑 PORTFOLIO BETA GOVERNOR: Net Long margin exposure (${long_margin - short_margin:.2f} >= ${net_exposure_ceiling:.2f} limit). Blocking Long!")
                                continue
                        elif action_val < -0.20:
                            if len(open_shorts) >= 3:
                                logger.info(f"[{sym}] 🛑 PORTFOLIO BETA GOVERNOR: Maximum concurrent Shorts reached ({len(open_shorts)}/3: {open_shorts}). Blocking new Short to prevent correlated drawdown!")
                                continue
                            if (short_margin - long_margin) >= net_exposure_ceiling:
                                logger.info(f"[{sym}] 🛑 PORTFOLIO BETA GOVERNOR: Net Short margin exposure (${short_margin - long_margin:.2f} >= ${net_exposure_ceiling:.2f} limit). Blocking Short!")
                                continue

                    # GUARD 1.5: Spread Firewall (Protect against illiquid book slippage bleed)
                    if not is_closing and market and market.mid_price > 0:
                        spread_pct = market.spread / market.mid_price
                        if spread_pct > session_max_spread:
                            logger.warning(f"[{sym}] 🛑 SPREAD FIREWALL ({session_name}): Spread {spread_pct*100:.3f}% > {session_max_spread*100:.3f}% limit. Skipping entry to protect from slippage fee bleed!")
                            continue

                    # GUARD 2: Minimum Hold Duration (600s / 10 mins with Smart Trap Escape Door)
                    if is_closing and not exit_override:
                        # 1. Early exit if in solid impulse profit (>= +4.0% leveraged to cover fees)
                        is_profit_impulse = pnl_pct_leveraged >= 0.04
                        
                        # 2. Smart Trap Escape Door: Only cut early if Critic confirms a REAL multi-timeframe adverse breakdown:
                        # BOTH 5m AND 1h must agree on adverse direction (5m >= 0.40% AND 1h >= 0.70% adverse move),
                        # or a severe 5m shock (5m >= 0.80% adverse) with high confidence (>= 0.72).
                        # Prevents cutting positions after 15 minutes on minor 1h drift while 5m is favorable!
                        critic_vals = precomputed_critics.get(sym)
                        pred_ret_5m = float(critic_vals[0]) if critic_vals is not None and len(critic_vals) > 0 else 0.0
                        pred_ret_1h = float(critic_vals[1]) if critic_vals is not None and len(critic_vals) > 1 else pred_ret_5m
                        is_trap_detected = (pos.quantity > 0 and ((pred_ret_5m <= -0.40 and pred_ret_1h <= -0.70) or pred_ret_5m <= -0.80) and confidence >= 0.72) or \
                                           (pos.quantity < 0 and ((pred_ret_5m >= 0.40 and pred_ret_1h >= 0.70) or pred_ret_5m >= 0.80) and confidence >= 0.72)
                        
                        is_protective_loss = pnl_pct_leveraged <= -0.12  # Allows surviving wicks up to -12% leveraged (-0.60% price move)
                        
                        entry_t = self.position_entry_times.get(sym, 0.0)
                        time_held = (time.time() - entry_t) if entry_t > 0 else 999.0
                        
                        if is_trap_detected:
                            logger.warning(f"[{sym}] 🚨 CONFIRMED SMART TRAP: Critic forecasts severe adverse reversal (5m: {pred_ret_5m:+.2f}%, 1h: {pred_ret_1h:+.2f}%) with {confidence:.2f} confidence. Cutting trap early!")
                        elif time_held < 600.0 and not (is_profit_impulse or is_protective_loss):
                            logger.info(f"[{sym}] ⏳ MICRO-SWING BREATHING: Minimum hold (600s) not met yet ({time_held:.1f}s, PnL: {pnl_pct_leveraged*100:+.2f}%). Giving swing room to run!")
                            continue

                    # GUARD 3: Confidence Filter (Session Adaptive & Profit Ratchet Modulated)
                    effective_conf_hurdle = max(0.74 if self.profit_ratchet_active else session_conf_hurdle, session_conf_hurdle)
                    if not is_closing and confidence < effective_conf_hurdle: 
                        reason = "Profit Ratchet Preservation" if self.profit_ratchet_active and effective_conf_hurdle > session_conf_hurdle else session_name
                        logger.info(f"[{sym}] SKIPPING OPEN (Low Confidence: {confidence:.2f} < {effective_conf_hurdle:.2f} in {reason})")
                        
                        # Emit authentic live HOLD experience into DB when flat (15% sample rate)
                        # Populates LINK, ADA, AVAX, NEAR, LTC with genuine 200-dim market state sequences
                        if pos.quantity == 0 and random.random() < 0.15:
                            active_version = self.registry.get_current_version()
                            hold_exp_data = {
                                "experience_id": str(uuid.uuid4()),
                                "timestamp": int(time.time()),
                                "symbol": sym,
                                "market_state": seq, 
                                "macro_state": None, 
                                "portfolio_state": {
                                    "free_margin": float(self.portfolio_state.free_margin),
                                    "total_unrealized_pnl": float(self.portfolio_state.total_unrealized_pnl)
                                },
                                "derivatives_state": {
                                    "target_size": float(target_size),
                                    "price_offset": float(price_offset)
                                },
                                "position_before": 0.0,
                                "position_after": 0.0,
                                "entry_price": 0.0,
                                "exit_price": None,
                                "leverage": int(pos.leverage) if hasattr(pos, 'leverage') else settings.max_leverage,
                                "margin": 0.0,
                                "fees": 0.0,
                                "holding_duration": 0,
                                "action": "HOLD",
                                "action_probability": float(confidence),
                                "confidence": float(confidence),
                                "model_version": active_version,
                                "reward": 0.05, 
                                "realized_pnl": 0.0,
                                "unrealized_pnl": 0.0
                            }
                            asyncio.create_task(redis_manager.publish_binary("experience:completed", msgpack.packb(hold_exp_data, use_bin_type=True)))
                        continue 

                    # PILLAR 3: Macro Regime Trend-Alignment Gate (Block Counter-Trend Whipsaws)
                    if not is_closing:
                        btc_market = self.market_states.get("BTCUSDT")
                        eth_market = self.market_states.get("ETHUSDT")
                        btc_mom_15m = btc_market.features[4] if btc_market and len(btc_market.features) == 90 else 0.0
                        eth_mom_15m = eth_market.features[4] if eth_market and len(eth_market.features) == 90 else 0.0
                        
                        btc_crit = precomputed_critics.get("BTCUSDT", [0.0, 0.0, 0.0])
                        eth_crit = precomputed_critics.get("ETHUSDT", [0.0, 0.0, 0.0])
                        btc_critic_5m = float(btc_crit[0]) if len(btc_crit) > 0 else 0.0
                        btc_critic_1h = float(btc_crit[1]) if len(btc_crit) > 1 else btc_critic_5m
                        btc_critic_4h = float(btc_crit[2]) if len(btc_crit) > 2 else btc_critic_1h
                        
                        eth_critic_5m = float(eth_crit[0]) if len(eth_crit) > 0 else 0.0
                        eth_critic_1h = float(eth_crit[1]) if len(eth_crit) > 1 else eth_critic_5m
                        eth_critic_4h = float(eth_crit[2]) if len(eth_crit) > 2 else eth_critic_1h

                        # HIGH-CONVICTION FAST PASS: If symbol has strong conviction (>= 70%) and large predicted move, bypass macro noise!
                        sym_critic = precomputed_critics.get(sym, [0.0, 0.0, 0.0])
                        sym_c5m = float(sym_critic[0]) if len(sym_critic) > 0 else 0.0
                        has_high_conviction = confidence >= 0.70 and abs(sym_c5m) >= 0.40

                        # Macro Downtrend: Clear downward momentum on both BTC & ETH (>= -0.80% move)
                        is_macro_down = (btc_mom_15m < -0.008 and eth_mom_15m < -0.008) or \
                                        (btc_critic_1h < -0.40 and eth_critic_1h < -0.40)
                        # Macro Uptrend: Clear upward momentum on both BTC & ETH (>= +0.80% move)
                        is_macro_up = (btc_mom_15m > 0.008 and eth_mom_15m > 0.008) or \
                                      (btc_critic_1h > 0.40 and eth_critic_1h > 0.40)

                        if not has_high_conviction:
                            if action_val > 0.2 and is_macro_down:
                                logger.info(f"[{sym}] 🛑 MACRO TREND GATE: BTC/ETH severe breakdown (15m: {btc_mom_15m*100:+.2f}%/{eth_mom_15m*100:+.2f}%). Blocking counter-trend OPEN_LONG!")
                                continue
                            elif action_val < -0.2 and is_macro_up:
                                logger.info(f"[{sym}] 🛑 MACRO TREND GATE: BTC/ETH severe breakout (15m: {btc_mom_15m*100:+.2f}%/{eth_mom_15m*100:+.2f}%). Blocking counter-trend OPEN_SHORT!")
                                continue
                        elif has_high_conviction and ((action_val > 0.2 and is_macro_down) or (action_val < -0.2 and is_macro_up)):
                            logger.info(f"[{sym}] 🚀 HIGH-CONVICTION BYPASS: Proceeding with trade (Confidence: {confidence:.2f}, 5m Critic: {sym_c5m:+.2f}%) despite macro divergence.")

                        # GUARD 3.4: Cross-Asset Lead-Lag Sniping ("The Crypto Echo" Momentum Front-Running)
                        # Altcoins lag Bitcoin impulse moves by 30-180 seconds.
                        # 1. Flash Dump Defense: Block Altcoin Longs if BTC is dumping (<= -0.35% in 1m)
                        # 2. Flash Pump Defense: Block Altcoin Shorts if BTC is pumping (>= +0.35% in 1m)
                        # 3. Echo Breakout Front-Running: Boost conviction 1.15x if BTC is breaking out and altcoin is lagging
                        if sym != "BTCUSDT":
                            btc_market = self.market_states.get("BTCUSDT")
                            if btc_market and len(btc_market.features) == 90 and market and len(market.features) == 90:
                                btc_1m_mom = float(btc_market.features[2])
                                sym_1m_mom = float(market.features[2])
                                
                                # Defense 1: Flash BTC Dump blocks Alt Longs
                                if action_val > 0.20 and btc_1m_mom <= -0.0035:
                                    logger.info(f"[{sym}] 🛑 LEAD-LAG DEFENSE: BTC 1m Flash Dump ({btc_1m_mom*100:+.2f}%). Blocking Altcoin Long to prevent correlated crash!")
                                    continue
                                # Defense 2: Flash BTC Pump blocks Alt Shorts
                                elif action_val < -0.20 and btc_1m_mom >= 0.0035:
                                    logger.info(f"[{sym}] 🛑 LEAD-LAG DEFENSE: BTC 1m Flash Pump ({btc_1m_mom*100:+.2f}%). Blocking Altcoin Short to prevent short-squeeze!")
                                    continue
                                    
                                # Offensive Echo Front-Running:
                                # If BTC has broken out (>= +0.20% in 1m) and this altcoin is lagging behind BTC (sym_1m_mom < btc_1m_mom):
                                if action_val > 0.20 and btc_1m_mom >= 0.0020 and sym_1m_mom < btc_1m_mom:
                                    lead_lag_mult = 1.15
                                    logger.info(f"[{sym}] ⚡ ECHO FRONT-RUNNER: BTC surging (+{btc_1m_mom*100:.2f}%) while {sym} lags (+{sym_1m_mom*100:.2f}%). Boosting conviction 1.15x to front-run the echo wave!")
                                # If BTC has broken down (<= -0.20% in 1m) and this altcoin is lagging behind BTC (sym_1m_mom > btc_1m_mom):
                                elif action_val < -0.20 and btc_1m_mom <= -0.0020 and sym_1m_mom > btc_1m_mom:
                                    lead_lag_mult = 1.15
                                    logger.info(f"[{sym}] ⚡ ECHO FRONT-RUNNER: BTC dumping ({btc_1m_mom*100:.2f}%) while {sym} lags ({sym_1m_mom*100:+.2f}%). Boosting Short conviction 1.15x to front-run the echo breakdown!")

                        # GUARD 3.5: Level-2 Order Flow Imbalance (OFI / Book Depth Wall Defense)
                        # Prevents buying directly into massive institutional sell walls or shorting into buy walls
                        if market and hasattr(market, 'order_book_imbalance') and market.order_book_imbalance is not None:
                            ob_imbalance = float(market.order_book_imbalance)
                            if action_val > 0.20 and ob_imbalance < -0.30:
                                logger.info(f"[{sym}] 🛑 OFI WALL DEFENSE: Orderbook skewed to sellers (Imbalance: {ob_imbalance:.2f} < -0.30). Skipping buy into sell wall!")
                                continue
                            elif action_val < -0.20 and ob_imbalance > 0.30:
                                logger.info(f"[{sym}] 🛑 OFI WALL DEFENSE: Orderbook skewed to buyers (Imbalance: {ob_imbalance:.2f} > +0.30). Skipping short into buy wall!")
                                continue

                        # GUARD 3.6: Cross-Exchange StatArb Confirmation (OKX & Bybit Lead-Lag Filter)
                        # Prevents buying into Binance-only local fakeout wicks; requires global market consensus
                        statarb_data = self.statarb_states.get(sym)
                        if statarb_data and (time.time() - float(statarb_data.get("timestamp", 0.0))) < 15.0:
                            okx_prem = float(statarb_data.get("okx_premium", 0.0))
                            bybit_prem = float(statarb_data.get("bybit_premium", 0.0))
                            
                            if action_val > 0.20 and (okx_prem < -0.0015 or bybit_prem < -0.0015):
                                logger.info(f"[{sym}] 🛑 STATARB DIVERGENCE: Global exchanges at discount (OKX: {okx_prem*100:+.2f}%, Bybit: {bybit_prem*100:+.2f}%). Blocking Binance fakeout LONG!")
                                continue
                            elif action_val < -0.20 and (okx_prem > 0.0015 or bybit_prem > 0.0015):
                                logger.info(f"[{sym}] 🛑 STATARB DIVERGENCE: Global exchanges at premium (OKX: {okx_prem*100:+.2f}%, Bybit: {bybit_prem*100:+.2f}%). Blocking Binance fakeout SHORT!")
                                continue

                        # GUARD 3.7: Whale Market Order Flow Momentum (Whale Tracker Integration)
                        # Prevents taking adverse trades when institutional whales are heavily dumping or sweeping the book.
                        # On testnet, synthetic orders from testing bots are ignored so they don't block legitimate trades.
                        whale_data = self.whale_states.get(sym)
                        if whale_data and (time.time() - float(whale_data.get("timestamp", 0.0))) < 60.0:
                            w_buy = float(whale_data.get("whale_buy_pressure", 0.0))
                            w_sell = float(whale_data.get("whale_sell_pressure", 0.0))
                            
                            # On Binance Testnet, testnet accounts trade with millions of fake dollars
                            if not settings.binance_testnet and not settings.allow_testnet:
                                sym_mom_15m = market.features[4] if market and len(market.features) == 90 else 0.0
                                critic_vals = precomputed_critics.get(sym)
                                c1h = float(critic_vals[1]) if critic_vals is not None and len(critic_vals) > 1 else 0.0
                                
                                # Extreme whale selling ($500k+ and 2.5x buy volume): Never buy into whale dumps UNLESS price/critic is surging
                                if action_val > 0.20 and w_sell >= 500_000.0 and w_sell > (2.5 * max(100_000.0, w_buy)):
                                    if sym_mom_15m > 0.002 or c1h > 0.10:
                                        logger.info(f"[{sym}] ⚠️ BYPASSING WHALE SELL DEFENSE: Real price momentum (+{sym_mom_15m*100:.2f}%) / 1h Critic ({c1h:+.2f}%) is strongly bullish. Overriding synthetic sell pressure (${w_sell:,.0f}).")
                                    else:
                                        logger.info(f"[{sym}] 🛑 WHALE FLOW DEFENSE: Extreme whale selling (${w_sell:,.0f} vs ${w_buy:,.0f}). Blocking Long entry!")
                                        continue
                                # Extreme whale buying ($500k+ and 2.5x sell volume): Never short into whale sweeps UNLESS price/critic is dumping
                                elif action_val < -0.20 and w_buy >= 500_000.0 and w_buy > (2.5 * max(100_000.0, w_sell)):
                                    if sym_mom_15m < -0.002 or c1h < -0.10:
                                        logger.info(f"[{sym}] ⚠️ BYPASSING WHALE BUY DEFENSE: Real price momentum ({sym_mom_15m*100:.2f}%) / 1h Critic ({c1h:+.2f}%) is strongly bearish. Overriding synthetic buy pressure (${w_buy:,.0f}).")
                                    else:
                                        logger.info(f"[{sym}] 🛑 WHALE FLOW DEFENSE: Extreme whale buying (${w_buy:,.0f} vs ${w_sell:,.0f}). Blocking Short entry!")
                                        continue
                            else:
                                if w_sell >= 1_000_000.0 or w_buy >= 1_000_000.0:
                                    logger.debug(f"[{sym}] 🐋 Testnet Whale Flow: Buy ${w_buy:,.0f} / Sell ${w_sell:,.0f} (Testnet telemetry only)")

                        # GUARD 3.8: Funding Rate Asymmetry & Squeeze Sniping
                        # Protects against paying high funding fees; exploits deep negative funding short squeezes
                        if market and hasattr(market, 'funding_rate') and market.funding_rate is not None:
                            f_rate = float(market.funding_rate)
                            # 1. Deep Negative Funding (<= -0.03%): Shorts are trapped paying longs (prime short squeeze zone)
                            if f_rate <= -0.0003:
                                if action_val < -0.20:
                                    logger.info(f"[{sym}] 🛑 FUNDING DEFENSE: Deep negative funding ({f_rate*100:.3f}%). Blocking Short to avoid funding drain & squeeze trap!")
                                    continue
                                elif action_val > 0.20:
                                    logger.info(f"[{sym}] 🧲 SHORT SQUEEZE SNIPER: Deep negative funding ({f_rate*100:.3f}%). Short squeeze conditions active!")
                            # 2. Overheated Positive Funding (>= +0.05%): Overcrowded long froth
                            elif f_rate >= 0.0005:
                                if action_val > 0.20:
                                    logger.info(f"[{sym}] 🛑 FUNDING DEFENSE: Frothy positive funding ({f_rate*100:.3f}%). Blocking Long to avoid heavy funding drain!")
                                    continue

                    elif is_closing and not exit_override:
                        # PILLAR 2: Dynamic Profit Harvesting & Multi-Horizon Critic-Assisted Reversal Exit
                        critic_vals = precomputed_critics.get(sym)
                        pred_ret_5m = float(critic_vals[0]) if critic_vals is not None and len(critic_vals) > 0 else 0.0
                        pred_ret_1h = float(critic_vals[1]) if critic_vals is not None and len(critic_vals) > 1 else pred_ret_5m
                        pred_ret_4h = float(critic_vals[2]) if critic_vals is not None and len(critic_vals) > 2 else pred_ret_1h
                        
                        has_reversal_critic = (pos.quantity > 0 and (pred_ret_5m <= -0.15 or pred_ret_1h <= -0.25)) or \
                                              (pos.quantity < 0 and (pred_ret_5m >= 0.15 or pred_ret_1h >= 0.25))
                        
                        # 1. Critic-Assisted Exit: If in solid profit (>= +3.5% leveraged) and Critic detects adverse reversal, lock in gain immediately!
                        if 0.035 <= pnl_pct_leveraged < 0.060 and has_reversal_critic:
                            logger.info(f"[{sym}] 🎯 CRITIC-ASSISTED PROFIT EXIT: PnL is +{pnl_pct_leveraged*100:.2f}% and Critic forecasts adverse reversal (5m: {pred_ret_5m:+.2f}%, 1h: {pred_ret_1h:+.2f}%). Securing green profit immediately!")
                            # Bypass fee-protection gate and allow smooth exit
                        elif 0.0 <= pnl_pct_leveraged < 0.040:
                            # 2. Fee Protection Gate: Widened from +1.8% to +4.0% leveraged (+0.20% price move at 20x) to ensure exchange fees (~$10) are cleanly paid with positive net USDT
                            logger.info(f"[{sym}] SKIPPING PREMATURE PROFIT CLOSE: PnL is +{pnl_pct_leveraged*100:.2f}% < +4.00% leveraged fee-protection threshold. Letting trend ride!")
                            continue
                        elif pnl_pct_leveraged >= 0.040:
                            # 3. Solid Profit (>= +4.0% leveraged): Allow smooth profit taking (0.40 hurdle)
                            close_hurdle = 0.40
                            if confidence < close_hurdle:
                                logger.info(f"[{sym}] SKIPPING PROFIT CLOSE (Low Confidence: {confidence:.2f} < {close_hurdle:.2f}, PnL: +{pnl_pct_leveraged*100:.2f}%)")
                                continue
                        else:
                            # 4. Negative PnL (< 0.0% leveraged):
                            # Crypto wicks are normal (noise buffer allows breathing room up to -15% hard stop).
                            # Cut early ONLY if Critic confirms a REAL adverse trend breakdown (5m >= 0.50% or 1h >= 0.70% adverse move)
                            has_adverse_critic = (pos.quantity > 0 and (pred_ret_5m <= -0.50 or pred_ret_1h <= -0.70)) or \
                                                 (pos.quantity < 0 and (pred_ret_5m >= 0.50 or pred_ret_1h >= 0.70))
                            
                            if pnl_pct_leveraged > -0.15 and not has_adverse_critic:
                                logger.info(f"[{sym}] SKIPPING WICK EXIT: PnL is {pnl_pct_leveraged*100:+.2f}% (surviving wick noise buffer up to -15%, Critic return 5m: {pred_ret_5m:+.2f}%, 1h: {pred_ret_1h:+.2f}% is not severe). Giving swing room to run!")
                                continue
                            
                            # Deep loss cut: require >= 0.65 conviction to cut before hard stop
                            if confidence < 0.65:
                                logger.info(f"[{sym}] SKIPPING LOSS CLOSE (Conviction {confidence:.2f} < 0.65, PnL: {pnl_pct_leveraged*100:+.2f}%)")
                                continue

                    # GUARD 4: Critic Multi-Horizon Swing Return Check (5m, 1h, 4h)
                    critic_vals = precomputed_critics.get(sym)
                    if not is_closing and critic_vals is not None and len(critic_vals) >= 1:
                        pred_ret_5m = float(critic_vals[0])
                        pred_ret_1h = float(critic_vals[1]) if len(critic_vals) > 1 else pred_ret_5m
                        pred_ret_4h = float(critic_vals[2]) if len(critic_vals) > 2 else pred_ret_1h
                        
                        min_hurdle = max(session_critic_hurdle, settings.min_open_expected_return_pct) # Dynamic session fee hurdle
                        
                        # 1. 5-Minute Entry Hurdle (Beats fee + spread)
                        if action_val > 0.2 and pred_ret_5m < min_hurdle:
                            logger.info(f"[{sym}] SKIPPING OPEN_LONG: 5m Critic return {pred_ret_5m:.2f}% < {min_hurdle:.2f}% fee hurdle.")
                            continue
                        elif action_val < -0.2 and pred_ret_5m > -min_hurdle:
                            logger.info(f"[{sym}] SKIPPING OPEN_SHORT: 5m Critic return {pred_ret_5m:.2f}% > -{min_hurdle:.2f}% fee hurdle.")
                            continue
                            
                        # 2. 1-Hour & 4-Hour Multi-Horizon Swing Alignment (Strict Anti-Suicide Rule)
                        # Strictly forbids entering counter-trend positions into strong macro swing momentum
                        if action_val > 0.2 and (pred_ret_1h < -0.05 or pred_ret_4h < -0.10):
                            logger.info(f"[{sym}] 🛑 SKIPPING OPEN_LONG: 1h Critic ({pred_ret_1h:+.2f}%) or 4h Critic ({pred_ret_4h:+.2f}%) opposes Long swing!")
                            continue
                        elif action_val < -0.2 and (pred_ret_1h > 0.05 or pred_ret_4h > 0.10):
                            logger.info(f"[{sym}] 🛑 SKIPPING OPEN_SHORT: 1h Critic ({pred_ret_1h:+.2f}%) or 4h Critic ({pred_ret_4h:+.2f}%) opposes Short swing!")
                            continue

                    # GUARD 5: Institutional Dynamic Volatility & Conviction Margin Sizing (Kelly / ATR Scaling)
                    # Dynamically scales between 6.0% and 15.0% based on conviction, horizon alignment, and coin volatility.
                    # Halved to 8.0% base margin (4.0% to 12.0% dynamic) in Profit Ratchet Capital Preservation Mode.
                    base_margin_pct = 0.08 if self.profit_ratchet_active else 0.15
                    
                    # 1. Conviction Multiplier (0.80x to 1.30x scaled to session hurdle)
                    conviction_mult = float(np.clip(0.80 + (confidence - effective_conf_hurdle) * 1.5, 0.80, 1.30))
                    
                    # 2. Multi-Horizon Critic Alignment Multiplier (0.85x to 1.15x)
                    critic_vals = precomputed_critics.get(sym)
                    if critic_vals is not None and len(critic_vals) >= 2:
                        c5m = float(critic_vals[0])
                        c1h = float(critic_vals[1])
                        # If 5m and 1h agree with trade direction, scale up; if divergent, scale down
                        if (action_val > 0.2 and c5m > 0 and c1h > 0) or (action_val < -0.2 and c5m < 0 and c1h < 0):
                            critic_mult = 1.15
                        elif (action_val > 0.2 and c1h < 0) or (action_val < -0.2 and c1h > 0):
                            critic_mult = 0.85
                        else:
                            critic_mult = 1.00
                    else:
                        critic_mult = 1.00
                        
                    # 3. Asset Beta / Volatility Dampener
                    # Large caps (BTC, ETH) can handle higher margin; high-beta altcoins get dampener to prevent big fee bleed
                    if sym in ["BTCUSDT", "ETHUSDT"]:
                        beta_dampener = 1.10
                    elif sym in ["SOLUSDT", "BNBUSDT", "LTCUSDT"]:
                        beta_dampener = 1.00
                    else: # ADA, XRP, NEAR, AVAX, LINK (wild altcoins)
                        beta_dampener = 0.75
                        
                    # 4. Volatility Scaling from Market Observation
                    vol_scale = 1.0
                    if market and hasattr(market, 'volatility') and market.volatility > 0.015:
                        vol_scale = float(np.clip(0.015 / market.volatility, 0.70, 1.00))
                        
                    # 5. Whale Flow Alignment Multiplier
                    whale_mult = 1.0
                    whale_data = self.whale_states.get(sym)
                    if whale_data and (time.time() - float(whale_data.get("timestamp", 0.0))) < 60.0:
                        w_b = float(whale_data.get("whale_buy_pressure", 0.0))
                        w_s = float(whale_data.get("whale_sell_pressure", 0.0))
                        if action_val > 0.20 and w_b >= 300_000.0 and w_b > (1.5 * w_s):
                            whale_mult = 1.05
                        elif action_val < -0.20 and w_s >= 300_000.0 and w_s > (1.5 * w_b):
                            whale_mult = 1.05

                    is_major = sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
                    if is_major:
                        min_margin_bound = 0.04 if self.profit_ratchet_active else 0.06
                        max_margin_bound = 0.08 if self.profit_ratchet_active else 0.15
                    else:
                        # Altcoins (ADA, XRP, LINK, AVAX, NEAR, LTC): 5% - 10% margin
                        min_margin_bound = 0.03 if self.profit_ratchet_active else 0.05
                        max_margin_bound = 0.06 if self.profit_ratchet_active else 0.10
                    dynamic_margin_pct = float(np.clip(base_margin_pct * conviction_mult * critic_mult * beta_dampener * vol_scale * whale_mult * lead_lag_mult, min_margin_bound, max_margin_bound))
                    
                    max_margin_budget = max(0.0, self.portfolio_state.equity * dynamic_margin_pct)
                    margin_allocated = min(max_margin_budget, max(0.0, self.portfolio_state.free_margin) * target_size)
                    notional_requested = margin_allocated * pos.leverage
                    
                    # Side determination respecting Hysteresis Deadband
                    side = "HOLD"
                    if pos.quantity > 0:
                        if is_closing:
                            side = "CLOSE_LONG"
                        elif action_val > 0.20:
                            side = "OPEN_LONG"
                    elif pos.quantity < 0:
                        if is_closing:
                            side = "CLOSE_SHORT"
                        elif action_val < -0.20:
                            side = "OPEN_SHORT"
                    else:
                        if action_val < -0.20:
                            side = "OPEN_SHORT"
                        elif action_val > 0.20:
                            side = "OPEN_LONG"
                    
                    if side == "HOLD":
                        logger.info(f"[{sym}] HOLDING (Action Val: {action_val:.2f})")
                        
                        # High-Quality HOLD Experiences (15% sample rate so the agent learns that holding is profitable)
                        if random.random() < 0.15:
                            hold_dur = 0
                            if pos.entry_price > 0 and getattr(pos, 'entry_time', 0.0) > 0:
                                hold_dur = int(time.time() - pos.entry_time)
                            
                            # Reward patience: If flat, +0.05 reward for preserving capital and avoiding fee bleed
                            unreal_pnl = float(pos.unrealized_pnl) if hasattr(pos, 'unrealized_pnl') else 0.0
                            hold_reward = 0.05 if pos.quantity == 0 else max(-1.0, min(1.0, unreal_pnl * 0.1))
                            active_version = self.registry.get_current_version()
                                
                            exp_data = {
                                "experience_id": str(uuid.uuid4()),
                                "timestamp": int(time.time()),
                                "symbol": sym,
                                "market_state": seq, 
                                "macro_state": None, 
                                "portfolio_state": {
                                    "free_margin": float(self.portfolio_state.free_margin),
                                    "total_unrealized_pnl": float(self.portfolio_state.total_unrealized_pnl)
                                },
                                "derivatives_state": {
                                    "target_size": float(target_size),
                                    "price_offset": float(price_offset)
                                },
                                "position_before": float(pos.quantity),
                                "position_after": float(pos.quantity),
                                "entry_price": float(pos.entry_price),
                                "exit_price": None,
                                "leverage": int(pos.leverage) if hasattr(pos, 'leverage') else settings.max_leverage,
                                "margin": (float(pos.quantity) * market.mid_price / pos.leverage) if pos.leverage and pos.quantity else 0.0,
                                "fees": 0.0,
                                "holding_duration": hold_dur,
                                "action": "HOLD",
                                "action_probability": float(confidence),
                                "confidence": float(confidence),
                                "model_version": active_version,
                                "reward": float(hold_reward), 
                                "realized_pnl": 0.0,
                                "unrealized_pnl": float(pos.unrealized_pnl) if hasattr(pos, 'unrealized_pnl') else 0.0
                            }
                            asyncio.create_task(redis_manager.publish_binary("experience:completed", msgpack.packb(exp_data, use_bin_type=True)))
                    
                    if side != "HOLD":
                        active_version = self.registry.get_current_version()
                        target_qty = notional_requested / market.mid_price # Use EXPLICIT mid_price, no feature[6] hack
                        
                        # MICRO-SWING CONSOLIDATED ENTRY: If position is already established (>= 60% of target),
                        # do NOT micro-slice tiny incremental dust (eliminates fee bleed!)
                        if "OPEN" in side:
                            if abs(pos.quantity) > 0:
                                if abs(pos.quantity) >= (0.60 * target_qty):
                                    logger.info(f"[{sym}] Micro-Swing position established ({abs(pos.quantity):.4f} >= 60% of target {target_qty:.4f}). Skipping micro-slice.")
                                    continue
                                qty_raw = max(0.0, target_qty - abs(pos.quantity))
                            else:
                                qty_raw = target_qty
                                
                            # Filter micro-dust orders under $50 notional to prevent fee bleed
                            if (qty_raw * market.mid_price) < 50.0:
                                logger.info(f"[{sym}] Skipping micro-order: notional ${qty_raw * market.mid_price:.2f} < $50.00 minimum.")
                                continue
                        else:
                            # For CLOSE actions, we close the entire existing quantity, UNLESS it's a 50% partial TP!
                            if partial_tp_triggered:
                                sym_config = registry.get_symbol(sym)
                                partial_qty = abs(pos.quantity) * 0.50
                                if (partial_qty * market.mid_price) >= sym_config.min_notional:
                                    qty_raw = partial_qty
                                else:
                                    logger.info(f"[{sym}] 50% partial quantity notional (${partial_qty * market.mid_price:.2f}) < Min Notional (${sym_config.min_notional:.2f}). Letting full position ride!")
                                    partial_tp_triggered = False
                                    qty_raw = 0.0
                            else:
                                qty_raw = abs(pos.quantity)
                            
                        if qty_raw <= 0.0001:
                            logger.info(f"[{sym}] Target quantity reached. No additional {side} needed.")
                            continue
                        
                        request = OrderRequest(
                            symbol=sym, action_type=side, confidence=float(confidence),
                            requested_quantity=float(qty_raw), target_position=float(target_size),
                            model_version=active_version, timestamp=time.time()
                        )
                        
                        # Evaluate Risk via Schema
                        decision = self.risk_manager.evaluate(request, self.portfolio_state, market, self.macro_state)
                        
                        if decision.approved and decision.adjusted_quantity > 0:
                            sym_config = registry.get_symbol(sym)
                            qty_str = sym_config.format_quantity(decision.adjusted_quantity)
                            
                            if float(qty_str) <= 0:
                                logger.info(f"[{sym}] Formatted quantity is zero. Skipping execution.")
                                continue
                                
                            notional_val = float(qty_str) * market.mid_price
                            if notional_val < sym_config.min_notional and "OPEN" in side:
                                logger.info(f"[{sym}] Notional {notional_val:.2f} < Min Notional {sym_config.min_notional}. Skipping.")
                                continue
                                
                            logger.info(f"[{sym}] EXECUTING: {side} {qty_str} (Confidence: {confidence:.2f})")
                            
                            if (self.trading_enabled and 
                                not self.emergency_stop and
                                not settings.emergency_stop and
                                not self.dry_run and 
                                (settings.allow_testnet or settings.allow_live_trading) and 
                                self.has_valid_model and 
                                decision.approved):
                                
                                
                                # Phase 9: Execution and Experience Broadcasting
                                try:
                                    # Set leverage before executing
                                    try:
                                        await self.binance.set_leverage(sym, settings.max_leverage)
                                    except Exception as e:
                                        logger.warning(f"[{sym}] Could not set leverage (might already be set): {e}")

                                    # Calculate LIMIT order price using offset
                                    # Price Offset [0, 1] mapped to spread (e.g. 0 to 10 ticks away)
                                    tick_size = sym_config.tick_size
                                    offset_amount = (price_offset * 10) * tick_size
                                    
                                    binance_side = "BUY" if side in ["OPEN_LONG", "CLOSE_SHORT"] else "SELL"
                                    
                                    # Institutional Top-of-Book Maker Pegging:
                                    # If BUY: peg to best bid (or 1 tick below mid) to guarantee Maker status
                                    # If SELL: peg to best ask (or 1 tick above mid) to guarantee Maker status
                                    if binance_side == "BUY":
                                        limit_price = market.bid if market.bid > 0 else (market.mid_price - tick_size)
                                    else:
                                        limit_price = market.ask if market.ask > 0 else (market.mid_price + tick_size)
                                        
                                    limit_price_str = sym_config.format_price(limit_price)
                                    
                                    # Pre-capture entry price BEFORE sending the order to avoid race conditions with market fill
                                    entry_px = self.portfolio_state.positions[sym].entry_price if sym in self.portfolio_state.positions else 0.0
                                    pos_leverage = self.portfolio_state.positions[sym].leverage if sym in self.portfolio_state.positions else 10
                                    
                                    # Always fetch authoritative Binance live position entry price before sending close order
                                    if "CLOSE" in side and self.trading_enabled and not self.dry_run:
                                        try:
                                            live_pos_check = await self.binance.get_positions()
                                            for lp in live_pos_check:
                                                if lp.get("symbol") == sym and float(lp.get("positionAmt", 0)) != 0:
                                                    binance_ep = float(lp.get("entryPrice", 0.0))
                                                    if binance_ep > 0:
                                                        entry_px = binance_ep
                                                        self.portfolio_state.positions[sym].entry_price = binance_ep
                                                    break
                                        except Exception as e:
                                            logger.warning(f"[{sym}] Could not pre-fetch entry price from Binance: {e}")

                                    # Execute on Binance
                                    client_order_id = f"ai_bot_{uuid.uuid4().hex[:10]}"
                                    is_closing = "CLOSE" in side
                                    is_market_exit = is_closing or exit_override
                                    
                                    if is_market_exit:
                                        exit_reason = "HARD STOP" if hard_stop_triggered else ("HARD TAKE-PROFIT" if hard_take_profit_triggered else ("SUDDEN SHOCK" if sudden_shock_triggered else ("TRAILING STOP" if trailing_stop_triggered else ("STALE PRUNE" if stale_trade_triggered else ("FREE-ROLL BREAKEVEN" if freeroll_stop_triggered else ("PARTIAL TP 50%" if partial_tp_triggered else "AI CLOSE"))))))
                                        logger.info(f"[{sym}] MARKET INTENT ({exit_reason}): {binance_side} {qty_str}")
                                        try:
                                             await self.binance.cancel_all_orders(sym)
                                        except Exception as e:
                                             logger.warning(f"[{sym}] Failed to cancel lingering orders: {e}")
                                        order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id, order_type="MARKET", reduce_only=True)
                                        self.active_orders.pop(sym, None)
                                        self.last_close_time[sym] = time.time()
                                        if not partial_tp_triggered:
                                            self.peak_pnl_pct[sym] = 0.0
                                            if sym in self.pnl_history:
                                                self.pnl_history[sym].clear()
                                    else:
                                        # Institutional Post-Only Maker Execution (GTX)
                                        # Forces Maker fill (0.02% fee) and strictly forbids taker fill (0.05% fee)
                                        logger.info(f"[{sym}] 🏛️ POST-ONLY MAKER INTENT (GTX): {binance_side} at {limit_price_str}")
                                        try:
                                            order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id, order_type="LIMIT", price=limit_price_str, time_in_force="GTX", reduce_only=False)
                                        except Exception as e:
                                            # If post-only was rejected (-5022: would match and take liquidity), re-peg 1 tick deeper
                                            if "-5022" in str(e) or "Post Only" in str(e):
                                                logger.info(f"[{sym}] Post-Only GTX would cross book. Re-pegging 1 tick deeper...")
                                                if binance_side == "BUY":
                                                    limit_price -= tick_size
                                                else:
                                                    limit_price += tick_size
                                                limit_price_str = sym_config.format_price(limit_price)
                                                order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id, order_type="LIMIT", price=limit_price_str, time_in_force="GTX", reduce_only=False)
                                            else:
                                                raise e
                                    
                                    actual_order_id = order_res.get('orderId') or client_order_id
                                    logger.info(f"[{sym}] ORDER SUCCESS: {actual_order_id}")
                                    
                                    avg_fill = float(order_res.get('avgPrice', 0.0))
                                    exec_qty = float(order_res.get('executedQty', 0.0))
                                    cum_quote = float(order_res.get('cumQuote', 0.0))
                                    if avg_fill <= 0.0 and exec_qty > 0.0 and cum_quote > 0.0:
                                        avg_fill = cum_quote / exec_qty
                                    order_exec_price = avg_fill if avg_fill > 0 else (float(limit_price_str) if not is_market_exit else market.mid_price)

                                    # Prepare Trade History Record
                                    record_side = f"PARTIAL_{side}" if partial_tp_triggered else side
                                    trade_record = {
                                        "timestamp": time.time(),
                                        "symbol": sym,
                                        "side": record_side,
                                        "quantity": float(qty_str),
                                        "price": order_exec_price,
                                        "confidence": float(confidence),
                                        "order_id": actual_order_id,
                                        "entry_price": entry_px if ("CLOSE" in side and entry_px > 0) else order_exec_price,
                                        "gross_pnl": 0.0,
                                        "fees_paid": 0.0,
                                        "realized_pnl": 0.0,
                                        "roi_pct": 0.0
                                    }
                                    
                                    # Add Analytics for Closures
                                    if "CLOSE" in side and entry_px > 0:
                                        close_px = order_exec_price
                                        qty_closed = float(qty_str)
                                        if side == "CLOSE_LONG":
                                             gross_pnl = (close_px - entry_px) * qty_closed
                                        else:
                                            gross_pnl = (entry_px - close_px) * qty_closed
                                            
                                        # Accurate trading fees: Maker for entry, Taker for market/hard stop or Maker for limit
                                        notional_cost_exit = qty_closed * close_px
                                        exit_fee_rate = settings.taker_fee_pct if is_market_exit else settings.maker_fee_pct
                                        dashboard_trade_fees = (entry_px * qty_closed * settings.maker_fee_pct) + (notional_cost_exit * exit_fee_rate)
                                        pnl = gross_pnl - dashboard_trade_fees
                                        
                                        # Record close timestamp for symbol cooldown (Guard 1)
                                        self.last_close_time[sym] = time.time()
                                        
                                        # Calculate Net ROI
                                        margin_used_for_trade = (entry_px * qty_closed) / pos_leverage
                                        roi = (pnl / margin_used_for_trade) * 100 if margin_used_for_trade > 0 else 0.0
                                            
                                        trade_record["entry_price"] = entry_px
                                        trade_record["gross_pnl"] = gross_pnl
                                        trade_record["fees_paid"] = dashboard_trade_fees
                                        trade_record["realized_pnl"] = pnl
                                        trade_record["roi_pct"] = roi
                                        
                                        # Update Self-Awareness Win Rate
                                        self.session_trades += 1
                                        if roi > 0:
                                            self.session_wins += 1
                                            self.last_win_time = time.time()
                                        self.session_win_rate = self.session_wins / self.session_trades
                                        
                                        # Exact Binance Income Ledger Audit (Matches exchange to the penny)
                                        if self.trading_enabled and not self.dry_run:
                                            try:
                                                await asyncio.sleep(0.5)
                                                recent_income = await self.binance._request(
                                                    "GET", "/fapi/v1/income", signed=True,
                                                    params={"symbol": sym, "limit": 10}
                                                )
                                                now_ms = time.time() * 1000
                                                real_pnl = 0.0
                                                real_comm = 0.0
                                                matched = False
                                                for inc in (recent_income or []):
                                                    if (now_ms - float(inc.get("time", 0))) < 15000:
                                                        if inc.get("incomeType") == "REALIZED_PNL":
                                                            real_pnl += float(inc.get("income", 0.0))
                                                            matched = True
                                                        elif inc.get("incomeType") == "COMMISSION":
                                                            real_comm += abs(float(inc.get("income", 0.0)))
                                                if matched:
                                                    gross_pnl = real_pnl
                                                    dashboard_trade_fees = real_comm
                                                    pnl = gross_pnl - dashboard_trade_fees
                                                    trade_record["gross_pnl"] = gross_pnl
                                                    trade_record["fees_paid"] = dashboard_trade_fees
                                                    trade_record["realized_pnl"] = pnl
                                                    margin_used_for_trade = (entry_px * qty_closed) / pos_leverage if pos_leverage else (entry_px * qty_closed / 10)
                                                    roi = (pnl / margin_used_for_trade) * 100 if margin_used_for_trade > 0 else 0.0
                                                    trade_record["roi_pct"] = roi
                                                    logger.info(f"[{sym}] 📜 BINANCE OFFICIAL AUDIT: Gross: ${gross_pnl:+.4f}, Fees: ${dashboard_trade_fees:.4f}, Net: ${pnl:+.4f} (ROI: {roi:+.2f}%)")
                                            except Exception as audit_err:
                                                logger.warning(f"[{sym}] Income audit skipped: {audit_err}")
                                        
                                    if self.redis.redis and (exit_override or is_closing):
                                        await self.redis.redis.lpush("dashboard:trade_history", json.dumps(trade_record))
                                        await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)
                                        
                                    # Only track pending LIMIT orders (market closes fill immediately)
                                    if not is_market_exit:
                                        self.active_orders[sym] = {'order_id': actual_order_id, 'timestamp': time.time(), 'trade_record': trade_record}
                                    
                                    # Update local position state immediately to prevent over-buying before the next sync
                                    notional_cost = float(qty_str) * order_exec_price
                                    margin_used = notional_cost / settings.max_leverage

                                    if "OPEN_LONG" in side:
                                        if self.portfolio_state.positions[sym].quantity == 0.0:
                                            self.portfolio_state.positions[sym].entry_time = time.time()
                                            self.position_entry_times[sym] = time.time()
                                            self.portfolio_state.positions[sym].entry_price = order_exec_price
                                        else:
                                            old_q = abs(self.portfolio_state.positions[sym].quantity)
                                            new_q = float(qty_str)
                                            old_ep = self.portfolio_state.positions[sym].entry_price
                                            self.portfolio_state.positions[sym].entry_price = ((old_ep * old_q) + (order_exec_price * new_q)) / (old_q + new_q)
                                        self.portfolio_state.positions[sym].quantity += float(qty_str)
                                        self.portfolio_state.free_margin -= margin_used
                                    elif "CLOSE_LONG" in side:
                                        self.portfolio_state.positions[sym].quantity -= float(qty_str)
                                        self.portfolio_state.free_margin += margin_used
                                        if abs(self.portfolio_state.positions[sym].quantity) < 1e-5:
                                            self.portfolio_state.positions[sym].quantity = 0.0
                                            self.portfolio_state.positions[sym].entry_price = 0.0
                                            self.position_entry_times[sym] = 0.0
                                            self.peak_pnl_pct[sym] = 0.0
                                            self.partial_tp_triggered[sym] = False
                                            if sym in self.pnl_history:
                                                self.pnl_history[sym].clear()
                                    elif "OPEN_SHORT" in side:
                                        if self.portfolio_state.positions[sym].quantity == 0.0:
                                            self.portfolio_state.positions[sym].entry_time = time.time()
                                            self.position_entry_times[sym] = time.time()
                                            self.portfolio_state.positions[sym].entry_price = order_exec_price
                                        else:
                                            old_q = abs(self.portfolio_state.positions[sym].quantity)
                                            new_q = float(qty_str)
                                            old_ep = self.portfolio_state.positions[sym].entry_price
                                            self.portfolio_state.positions[sym].entry_price = ((old_ep * old_q) + (order_exec_price * new_q)) / (old_q + new_q)
                                        self.portfolio_state.positions[sym].quantity -= float(qty_str)
                                        self.portfolio_state.free_margin -= margin_used
                                    elif "CLOSE_SHORT" in side:
                                        self.portfolio_state.positions[sym].quantity += float(qty_str)
                                        self.portfolio_state.free_margin += margin_used
                                        if abs(self.portfolio_state.positions[sym].quantity) < 1e-5:
                                            self.portfolio_state.positions[sym].quantity = 0.0
                                            self.portfolio_state.positions[sym].entry_price = 0.0
                                            self.position_entry_times[sym] = 0.0
                                            self.peak_pnl_pct[sym] = 0.0
                                            self.partial_tp_triggered[sym] = False
                                            if sym in self.pnl_history:
                                                self.pnl_history[sym].clear()
                                                
                                    if partial_tp_triggered:
                                        self.partial_tp_triggered[sym] = True
                                        logger.info(f"[{sym}] 🚀 FREE-ROLL RUNNER ARMED: 50% banked at market. Remaining {abs(self.portfolio_state.positions[sym].quantity):.4f} is now a 100% Risk-Free Free-Roll!")
                                    
                                    # Calculate immediate RL Reward
                                    imm_reward = 0.0
                                    imm_pnl = 0.0
                                    exit_px = None
                                    trade_fees = 0.0
                                    hold_dur = None
                                    
                                    if "CLOSE" in side and entry_px > 0:
                                        exit_px = float(order_exec_price)
                                        trade_fees = dashboard_trade_fees
                                        # pnl is already net of fees from above calculation
                                        
                                        entry_t = getattr(self.portfolio_state.positions[sym], 'entry_time', 0.0)
                                        if entry_t > 0:
                                            hold_dur = int(time.time() - entry_t)
                                        
                                        # --- ACCOUNT-AGNOSTIC REWARD UPGRADE ---
                                        # Use ROI (Percentage) instead of absolute PnL so the AI 
                                        # does not get confused when moving from $100k Testnet to $500 Live.
                                        #
                                        # FIX: Confidence-Weighted Reward — teaches the AI that
                                        # high confidence on a losing trade = EXTRA pain,
                                        # high confidence on a winning trade = EXTRA reward.
                                        # This creates a gradient signal for calibrating confidence honestly.
                                        if roi < 0:
                                            # Confidence-Amplified Loss Penalty:
                                            # At confidence=0.0 → 1.0x base penalty
                                            # At confidence=0.5 → 1.75x penalty  
                                            # At confidence=1.0 → 2.5x penalty (overconfident = extra pain)
                                            confidence_penalty = 1.0 + (confidence * 1.5)
                                            imm_reward = roi * confidence_penalty
                                        elif roi > 0 and roi < 5.0:
                                            # Time Decay Proxy: Punish tiny "lazy" wins (< 5% ROI)
                                            # Forces the AI to look for real momentum instead of micro-scalping
                                            imm_reward = -abs(roi * 0.5) 
                                        else:
                                            # Confidence-Amplified Sniper Bonus:
                                            # High confidence on a big win = rewarded for accurate conviction
                                            confidence_bonus = 1.0 + (confidence * 0.5)
                                            imm_reward = roi * confidence_bonus
                                            
                                        imm_pnl = pnl
                                    elif "OPEN" in side:
                                        # Maker fee penalty (Standardized to ROI percentage to avoid scale-attachment)
                                        imm_reward = -(settings.maker_fee_pct * 100.0) * pos_leverage
                                        
                                    pos_after = float(self.portfolio_state.positions[sym].quantity)
                                    
                                    # Record Experience
                                    exp_data = {
                                        "experience_id": str(uuid.uuid4()),
                                        "timestamp": int(time.time()),
                                        "symbol": sym,
                                        "market_state": seq, # Save full 2D sequence [300, 108] so the AI has the full movie for this specific trade!
                                        "macro_state": None, 
                                        "portfolio_state": {
                                            "free_margin": float(self.portfolio_state.free_margin),
                                            "total_unrealized_pnl": float(self.portfolio_state.total_unrealized_pnl)
                                        },
                                        "derivatives_state": {
                                            "target_size": float(target_size),
                                            "price_offset": float(price_offset)
                                        },
                                        "position_before": float(pos.quantity),
                                        "position_after": pos_after,
                                        "entry_price": float(entry_px if ("CLOSE" in side and entry_px > 0) else order_exec_price),
                                        "exit_price": exit_px,
                                        "leverage": int(pos_leverage),
                                        "margin": float(margin_used),
                                        "fees": float(trade_fees) if trade_fees else 0.0,
                                        "holding_duration": hold_dur,
                                        "action": side,
                                        "action_probability": float(confidence),
                                        "confidence": float(confidence),
                                        "model_version": active_version,
                                        "reward": float(imm_reward),
                                        "realized_pnl": float(imm_pnl),
                                        "unrealized_pnl": 0.0
                                    }
                                    experience_payload = msgpack.packb(
                                        exp_data,
                                        use_bin_type=True,
                                    )
                                    if hard_stop_triggered:
                                        # Market order executed immediately
                                        await redis_manager.publish_binary(
                                            "experience:completed",
                                            experience_payload,
                                        )
                                    elif sym in self.active_orders:
                                        # Limit order: attach to active_orders to publish ONLY when filled!
                                        self.active_orders[sym]['experience_payload'] = experience_payload
                                except Exception as e:
                                    logger.error(f"[{sym}] ORDER/EXPERIENCE FAILED: {e}")
                                    if "-4118" in str(e) or "ReduceOnly Order Failed" in str(e):
                                        logger.warning(f"[{sym}] Ghost limit order detected. Clearing order book to unstuck AI.")
                                        try:
                                            await self.binance.cancel_all_orders(sym)
                                            # Clean up local active orders tracking to match reality
                                            if sym in self.active_orders:
                                                del self.active_orders[sym]
                                        except Exception as cancel_e:
                                            logger.error(f"[{sym}] Failed to clear ghost orders: {cancel_e}")
                                    
                            else:
                                logger.info(f"[{sym}] NO ORDER (Blocked by final safety gate): {side} {qty_str}")
                        else:
                            logger.warning(f"[{sym}] Risk Manager REJECTED: {decision.reason}")
                            
            except Exception as e:
                logger.critical(f"FAIL CLOSED: Critical error in inference loop: {e}")
                self.running = False # Halt the bot immediately
                raise RuntimeError(f"FAIL CLOSED: Inference loop error: {e}")
                
            await asyncio.sleep(5.0)
            
    async def monitor_kill_switch(self):
        while self.running:
            try:
                if self.redis.redis:
                    ks = await self.redis.redis.get("system:kill_switch:active")
                    val = (ks.decode("utf-8") if isinstance(ks, bytes) else str(ks or "")).strip().lower()
                    if val in ["true", "1", "yes"]:
                        if not self.emergency_stop:
                            logger.critical("🚨 EMERGENCY KILL SWITCH ACTIVATED VIA REDIS! 🚨")
                            self.emergency_stop = True
                            self.trading_enabled = False
                            settings.emergency_stop = True
                            await self.panic_liquidate()
                    else:
                        if self.emergency_stop:
                            logger.info("🟢 EMERGENCY KILL SWITCH DISARMED VIA REDIS. Normal trading restored.")
                            self.emergency_stop = False
                            self.trading_enabled = settings.trading_enabled
                            settings.emergency_stop = False
            except Exception as e:
                logger.error(f"Error in monitor_kill_switch: {e}")
            await asyncio.sleep(2.0)

    async def panic_liquidate(self):
        """Immediately closes all open positions and cancels open orders."""
        logger.critical("Initiating PANIC LIQUIDATION for all open positions!")
        try:
            live_positions = await self.binance.get_positions()
            for p in live_positions:
                sym = p.get("symbol")
                amt = float(p.get("positionAmt", 0))
                if abs(amt) > 0.0001:
                    logger.critical(f"Panic liquidating {sym} {amt}")
                    side = "SELL" if amt > 0 else "BUY"
                    
                    if not self.dry_run:
                        # Cancel any open orders for this symbol first
                        try:
                            await self.binance.cancel_all_orders(sym)
                        except Exception as e:
                            logger.warning(f"Failed to cancel orders for {sym}: {e}")
                            
                        # Send market order to close
                        client_order_id = f"panic_{uuid.uuid4().hex[:10]}"
                        try:
                            await self.binance.create_order(
                                symbol=sym,
                                side=side,
                                quantity=abs(amt),
                                client_order_id=client_order_id,
                                order_type="MARKET",
                                reduce_only=True
                            )
                            logger.critical(f"Panic Market {side} for {sym} executed.")
                        except Exception as order_e:
                            logger.error(f"Failed to execute panic liquidation for {sym}: {order_e}")
        except Exception as e:
            logger.error(f"Error during panic liquidation: {e}")

    async def start(self):
        self.running = True
        await registry.initialize_from_exchange(self.binance)
        
        # Cancel all open orders on startup to ensure a clean slate and free up margin
        try:
            for sym in self.symbols:
                await self.binance.cancel_all_orders(sym)
            logger.info("Cleared all pre-existing open orders on startup.")
        except Exception as e:
            logger.warning(f"Failed to clear open orders on startup: {e}")
        
        # Fetch actual account balance to allow RiskManager to approve trades
        try:
            usdt = await self.binance.get_account_details()
            if not usdt and self.dry_run:
                usdt = {"balance": 1000.0, "crossWalletBalance": 1000.0, "crossUnPnl": 0.0, "availableBalance": 1000.0}
            self.portfolio_state.wallet_balance = float(usdt.get("balance", 0.0))
            self.portfolio_state.equity = float(usdt.get("crossWalletBalance", self.portfolio_state.wallet_balance)) + float(usdt.get("crossUnPnl", 0.0))
            self.portfolio_state.free_margin = float(usdt.get("availableBalance", self.portfolio_state.wallet_balance))
            balance = self.portfolio_state.wallet_balance
            logger.info(f"Initialized Portfolio with Balance: {balance} USDT")
        except Exception as e:
            logger.error(f"Failed to fetch account balance: {e}")
            if self.dry_run:
                self.portfolio_state.wallet_balance = 1000.0
                self.portfolio_state.equity = 1000.0
                self.portfolio_state.free_margin = 1000.0
                logger.info("Using mock 1000 USDT balance for Dry Run.")

        # Check if kill switch is active at startup
        try:
            if self.redis.redis:
                ks = await self.redis.redis.get("system:kill_switch:active")
                val = (ks.decode("utf-8") if isinstance(ks, bytes) else str(ks or "")).strip().lower()
                if val in ["true", "1", "yes"]:
                    logger.critical("🚨 STARTUP BLOCKED: Kill switch is currently ACTIVE in Redis! Halting trading immediately.")
                    self.emergency_stop = True
                    self.trading_enabled = False
                    settings.emergency_stop = True
                    await self.panic_liquidate()
        except Exception as ks_err:
            logger.error(f"Failed to check initial kill switch: {ks_err}")
        
        tasks = [
            asyncio.create_task(self.listen_macro()),
            asyncio.create_task(self.inference_loop()),
            asyncio.create_task(self.sync_dashboard()),
            asyncio.create_task(self.monitor_kill_switch())
        ]
        for sym in self.symbols:
            tasks.append(asyncio.create_task(self.listen_market(sym)))
            tasks.append(asyncio.create_task(self.listen_whale(sym)))
            tasks.append(asyncio.create_task(self.listen_statarb(sym)))
            
        try:
            await asyncio.gather(*tasks)
        finally:
            await self.binance.close()
        
    def stop(self):
        self.running = False

if __name__ == "__main__":
    set_log_file("logs/trading_bot.log")
    from core.config.settings import settings
    bot = ProductionTradingBot(symbols=settings.symbol_universe)
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        bot.stop()
    except Exception as e:
        logger.critical(f"FATAL ERROR ON STARTUP: {e}", exc_info=True)
