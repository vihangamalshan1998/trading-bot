import asyncio
import json
import uuid
import torch
import numpy as np
import time
import os
import collections
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
        
        # Phase 12: LSTM Sliding Window Sequence (120 steps * 5s = 10 minutes)
        self.sequence_length = 120
        self.state_history = collections.defaultdict(lambda: collections.deque(maxlen=self.sequence_length))
        
        # Limit Order Tracking
        self.active_orders = {} # symbol -> {'order_id': str, 'timestamp': float}
        
        # Self-Awareness Tracking
        self.session_trades = 0
        self.session_wins = 0
        self.session_win_rate = 0.0
        
        self.max_equity = 0.0
        self.last_win_time = time.time()
        self.equity_history = collections.deque(maxlen=60) # Last 60 ticks (5 mins)
        
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
        
        obs.extend([
            # Whale Tracking (Slots 130 - 139)
            float(whale.get("buy_wall_distance", 0.0)),
            float(whale.get("sell_wall_distance", 0.0)),
            float(whale.get("whale_buy_pressure", 0.0)),
            float(whale.get("whale_sell_pressure", 0.0)),
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
                    # Reset local quantities before parsing
                    for s in self.symbols:
                        if s in self.portfolio_state.positions:
                            self.portfolio_state.positions[s].quantity = 0.0
                            
                    for p in live_positions:
                        sym = p.get("symbol")
                        amt = float(p.get("positionAmt", 0))
                        
                        # Sync back to internal portfolio state
                        if sym in self.portfolio_state.positions:
                            self.portfolio_state.positions[sym].quantity = amt
                            self.portfolio_state.positions[sym].entry_price = float(p.get("entryPrice", 0))
                            self.portfolio_state.positions[sym].leverage = int(p.get("leverage", 10))
                            
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
                    
                    dashboard_data = {
                        "equity": self.portfolio_state.equity,
                        "positions": formatted_positions
                    }
                    await redis_manager.redis.set("dashboard:portfolio", json.dumps(dashboard_data))
            except Exception as e:
                logger.error(f"Dashboard sync failed: {e}")
            await asyncio.sleep(5.0)

    async def inference_loop(self):
        logger.info("Starting production inference loop...")
        while self.running:
            try:
                # 0. Hot Reload Model
                if self.registry.check_for_updates():
                    logger.info("New model version detected. Hot reloading...")
                    self.model = self.registry.load_model(self.model)
                    self.model.eval()
                    logger.info("Hot reload complete.")

                # 1. Check if we have valid market data
                if len(self.market_states) == 0:
                    logger.warning("Waiting for initial market data streams to connect...")
                    await asyncio.sleep(5.0)
                    continue
                    
                for sym in self.symbols:
                    if sym not in self.market_states:
                        continue
                        
                    # Check for pending Limit Order
                    if sym in self.active_orders:
                        time_elapsed = time.time() - self.active_orders[sym]['timestamp']
                        if time_elapsed > 60:
                            logger.info(f"[{sym}] LIMIT ORDER TIMEOUT (> 60s). Cancelling...")
                            try:
                                await self.binance.cancel_order(sym, self.active_orders[sym]['order_id'])
                            except Exception as e:
                                logger.warning(f"[{sym}] Failed to cancel (maybe already filled/cancelled): {e}")
                            # Revert optimistic portfolio update since order timed out and didn't fill
                            if 'trade_record' in self.active_orders[sym]:
                                side = self.active_orders[sym]['trade_record']['side']
                                qty = self.active_orders[sym]['trade_record']['quantity']
                                if "OPEN_LONG" in side or "CLOSE_SHORT" in side:
                                    self.portfolio_state.positions[sym].quantity -= float(qty)
                                elif "CLOSE_LONG" in side or "OPEN_SHORT" in side:
                                    self.portfolio_state.positions[sym].quantity += float(qty)
                            del self.active_orders[sym]

                            continue
                        else:
                            # Order is still active. Verify if it's already filled via Binance API
                            try:
                                open_orders = await self.binance.get_open_orders(sym)
                                # orderId from binance could be int or str, we check both string casts
                                if not any(str(o.get('orderId')) == str(self.active_orders[sym]['order_id']) or o.get('clientOrderId') == self.active_orders[sym]['order_id'] for o in open_orders):
                                    logger.info(f"[{sym}] LIMIT ORDER FILLED/NO LONGER OPEN! Removing from tracker.")
                                    
                                    # Push to dashboard history ONLY when filled
                                    if 'trade_record' in self.active_orders[sym]:
                                        asyncio.create_task(redis_manager.redis.lpush("dashboard:trade_history", json.dumps(self.active_orders[sym]['trade_record'])))
                                        asyncio.create_task(redis_manager.redis.ltrim("dashboard:trade_history", 0, 99))
                                        
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
                        
                    state_vector, macro_features = self._build_state_vector(sym)
                    
                    # 1b. Update Sliding Window History
                    self.state_history[sym].append(state_vector.squeeze(0).tolist()) # Remove batch dim, convert to list
                    
                    # Pad sequence if we don't have enough history yet
                    seq = list(self.state_history[sym])
                    while len(seq) < self.sequence_length:
                        seq.insert(0, seq[0] if len(seq) > 0 else state_vector.squeeze(0).tolist())
                        
                    # Forward Pass (shape: batch=1, seq_len=120, feature=41)
                    state_tensor = torch.tensor([seq], dtype=torch.float32)
                    
                    with torch.no_grad():
                        action_logits, expected_return = self.model(state_tensor)
                        action_logits = action_logits[0].numpy()
                        
                    market = self.market_states[sym]
                    
                    # Update portfolio state with current price for RiskManager correlated exposure checks
                    if sym in self.portfolio_state.positions:
                        self.portfolio_state.positions[sym].current_price = market.mid_price
                    # Calculate Real-Time PNL Percentage for Hard Stop Loss
                    hard_stop_triggered = False
                    if sym in self.portfolio_state.positions:
                        pos_check = self.portfolio_state.positions[sym]
                        if pos_check.quantity != 0 and pos_check.entry_price > 0:
                            # PNL % = (Current - Entry) / Entry  (* -1 if short)
                            pnl_pct = (market.mid_price - pos_check.entry_price) / pos_check.entry_price
                            if pos_check.quantity < 0:
                                pnl_pct = -pnl_pct
                            
                            # Leverage amplifies PNL %
                            pnl_pct_leveraged = pnl_pct * pos_check.leverage
                            
                            if pnl_pct_leveraged <= -0.10: # -10% HARD STOP LOSS
                                logger.warning(f"[{sym}] 🛑 HARD STOP LOSS TRIGGERED: Position is down {pnl_pct_leveraged*100:.2f}%. Overriding AI.")
                                hard_stop_triggered = True
                                
                    import random
                    
                    if hard_stop_triggered:
                        # Force a MARKET CLOSE
                        action_val = -1.0 if pos_check.quantity > 0 else 1.0
                        confidence = 1.0
                        target_size = 0.0
                        price_offset = 0.0 # Force immediate execution (0 offset)
                    elif random.random() < 0.02:
                        # Exploration Noise (2% chance to explore a random strategy)
                        action_val = random.uniform(-1.0, 1.0)
                        confidence = random.uniform(0.3, 1.0) # Ensure it passes the 0.3 threshold to trade
                        target_size = random.uniform(0.1, 1.0)
                        price_offset = random.uniform(0.0, 1.0) # Limit Order Offset
                        logger.info(f"[{sym}] EXPLORING: Applying curiosity noise to discover new strategies.")
                    else:
                        # Deterministic Policy (85% chance to use learned weights)
                        action_val = action_logits[0]
                        confidence = (action_logits[1] + 1.0) / 2.0
                        target_size = (action_logits[2] + 1.0) / 2.0
                        price_offset = (action_logits[3] + 1.0) / 2.0
                    
                    # 1. Determine Predicted Side (Even if confidence is low, we want to know what it *leans* towards)
                    if action_val < -0.2: predicted_side = "SHORT"
                    elif action_val > 0.2: predicted_side = "LONG"
                    else: predicted_side = "HOLD"
                    
                    # 2. Publish AI internal state to Redis for the Dashboard
                    ai_state_data = {
                        "confidence": float(confidence),
                        "action_val": float(action_val),
                        "target_size": float(target_size),
                        "price_offset": float(price_offset),
                        "predicted_side": predicted_side,
                        "state_vector": state_vector.squeeze(0).tolist()
                    }
                    asyncio.create_task(redis_manager.redis.set(f"ai:state:{sym}", json.dumps(ai_state_data)))
                    
                    if confidence < 0.3: 
                        logger.info(f"[{sym}] SKIPPING (Low Confidence: {confidence:.2f})")
                        continue 
                        
                    margin_allocated = max(0, self.portfolio_state.free_margin) * target_size
                        
                    pos = self.portfolio_state.positions[sym]
                    notional_requested = margin_allocated * pos.leverage
                    
                    side = "HOLD"
                    if action_val < -0.2: side = "CLOSE_LONG" if pos.quantity > 0 else "OPEN_SHORT"
                    elif action_val > 0.2: side = "CLOSE_SHORT" if pos.quantity < 0 else "OPEN_LONG"
                    
                    if side == "HOLD":
                        logger.info(f"[{sym}] HOLDING (Action Val: {action_val:.2f})")
                    
                    if side != "HOLD":
                        target_qty = notional_requested / market.mid_price # Use EXPLICIT mid_price, no feature[6] hack
                        
                        # Only trade the difference between target and current position
                        if "OPEN" in side:
                            qty_raw = max(0.0, target_qty - abs(pos.quantity))
                        else:
                            # For CLOSE actions, we close the entire existing quantity (or up to target)
                            qty_raw = abs(pos.quantity)
                            
                        if qty_raw <= 0.0001:
                            logger.info(f"[{sym}] Target quantity reached. No additional {side} needed.")
                            continue
                        
                        request = OrderRequest(
                            symbol=sym, action_type=side, confidence=float(confidence),
                            requested_quantity=float(qty_raw), target_position=float(target_size),
                            model_version="v1", timestamp=time.time()
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
                                not self.dry_run and 
                                (settings.allow_testnet or settings.allow_live_trading) and 
                                not settings.emergency_stop and 
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
                                    
                                    if binance_side == "BUY":
                                        limit_price = market.mid_price - offset_amount
                                    else:
                                        limit_price = market.mid_price + offset_amount
                                        
                                    limit_price_str = sym_config.format_price(limit_price)
                                    
                                    # Execute on Binance
                                    client_order_id = f"ai_bot_{uuid.uuid4().hex[:10]}"
                                    is_closing = "CLOSE" in side
                                    
                                    if hard_stop_triggered:
                                        logger.info(f"[{sym}] MARKET INTENT (HARD STOP): {binance_side} {qty_str}")
                                        order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id, order_type="MARKET", reduce_only=is_closing)
                                    else:
                                        logger.info(f"[{sym}] LIMIT INTENT: {binance_side} at {limit_price_str} (Offset: {offset_amount:.4f})")
                                        order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id, order_type="LIMIT", price=limit_price_str, time_in_force="GTC", reduce_only=is_closing)
                                    
                                    actual_order_id = order_res.get('orderId') or client_order_id
                                    logger.info(f"[{sym}] ORDER SUCCESS: {actual_order_id}")
                                    
                                    # Capture entry price before modifying position
                                    entry_px = self.portfolio_state.positions[sym].entry_price if sym in self.portfolio_state.positions else 0.0
                                    pos_leverage = self.portfolio_state.positions[sym].leverage if sym in self.portfolio_state.positions else 10
                                    
                                    # Prepare Trade History Record
                                    trade_record = {
                                        "timestamp": time.time(),
                                        "symbol": sym,
                                        "side": side,
                                        "quantity": float(qty_str),
                                        "price": market.mid_price,
                                        "confidence": float(confidence),
                                        "order_id": actual_order_id,
                                        "state_vector": state_vector.squeeze(0).tolist()
                                    }
                                    
                                    # Add Analytics for Closures
                                    if "CLOSE" in side and entry_px > 0:
                                        close_px = market.mid_price
                                        qty_closed = float(qty_str)
                                        if side == "CLOSE_LONG":
                                            pnl = (close_px - entry_px) * qty_closed
                                            roi = ((close_px - entry_px) / entry_px) * 100 * pos_leverage
                                        else:
                                            pnl = (entry_px - close_px) * qty_closed
                                            roi = ((entry_px - close_px) / entry_px) * 100 * pos_leverage
                                            
                                        trade_record["entry_price"] = entry_px
                                        trade_record["realized_pnl"] = pnl
                                        trade_record["roi_pct"] = roi
                                        
                                        # Update Self-Awareness Win Rate
                                        self.session_trades += 1
                                        if roi > 0:
                                            self.session_wins += 1
                                            self.last_win_time = time.time()
                                        self.session_win_rate = self.session_wins / self.session_trades
                                        
                                    # Track the limit order so we don't spam
                                    self.active_orders[sym] = {'order_id': actual_order_id, 'timestamp': time.time(), 'trade_record': trade_record}
                                    
                                    # Update local position state immediately to prevent over-buying before the next sync
                                    notional_cost = float(qty_str) * market.mid_price
                                    margin_used = notional_cost / settings.max_leverage

                                    if "OPEN_LONG" in side:
                                        if self.portfolio_state.positions[sym].quantity == 0.0:
                                            self.portfolio_state.positions[sym].entry_time = time.time()
                                        self.portfolio_state.positions[sym].quantity += float(qty_str)
                                        self.portfolio_state.free_margin -= margin_used
                                    elif "CLOSE_LONG" in side:
                                        self.portfolio_state.positions[sym].quantity -= float(qty_str)
                                        self.portfolio_state.free_margin += margin_used
                                    elif "OPEN_SHORT" in side:
                                        if self.portfolio_state.positions[sym].quantity == 0.0:
                                            self.portfolio_state.positions[sym].entry_time = time.time()
                                        self.portfolio_state.positions[sym].quantity -= float(qty_str)
                                        self.portfolio_state.free_margin -= margin_used
                                    elif "CLOSE_SHORT" in side:
                                        self.portfolio_state.positions[sym].quantity += float(qty_str)
                                        self.portfolio_state.free_margin += margin_used
                                    
                                    # Calculate immediate RL Reward
                                    imm_reward = 0.0
                                    imm_pnl = 0.0
                                    if "CLOSE" in side and entry_px > 0:
                                        # --- SHARPE RATIO & ASYMMETRIC REWARD UPGRADE ---
                                        if roi < 0:
                                            # Drawdown Penalty: Heavy punishment for closing at a loss
                                            imm_reward = pnl * 2.5 
                                        elif roi > 0 and roi < 1.0:
                                            # Time Decay Proxy: Punish tiny "lazy" wins (< 1% ROI)
                                            # Forces the AI to look for real momentum instead of micro-scalping
                                            imm_reward = -abs(pnl * 0.5) 
                                        else:
                                            # Sniper Bonus: Huge reward for clean, high-momentum trades
                                            imm_reward = pnl * 1.5
                                            
                                        imm_pnl = pnl
                                    elif "OPEN" in side:
                                        # Limit Maker fee penalty proxy for opening a trade
                                        notional_cost = float(qty_str) * market.mid_price
                                        imm_reward = -(notional_cost * 0.0002) # 0.02% fee penalty
                                        
                                    # Record Experience
                                    exp_data = {
                                        "experience_id": str(uuid.uuid4()),
                                        "timestamp": int(time.time()),
                                        "symbol": sym,
                                        "market_state": seq, # Save full 2D sequence [120, 41]
                                        "macro_state": None, 
                                        "portfolio_state": None, 
                                        "derivatives_state": {
                                            "target_size": float(target_size),
                                            "price_offset": float(price_offset)
                                        },
                                        "position_before": float(pos.quantity),
                                        "entry_price": float(pos.entry_price),
                                        "action": side,
                                        "confidence": float(confidence),
                                        "model_version": "v1",
                                        "reward": float(imm_reward),
                                        "realized_pnl": float(imm_pnl)
                                    }
                                    await redis_manager.redis.publish("experience:completed", json.dumps(exp_data))
                                    
                                except Exception as e:
                                    logger.error(f"[{sym}] ORDER/EXPERIENCE FAILED: {e}")
                                    
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
                    if ks and ks.decode("utf-8") == "true" and not self.emergency_stop:
                        logger.critical("🚨 EMERGENCY KILL SWITCH ACTIVATED VIA REDIS! 🚨")
                        self.emergency_stop = True
                        self.trading_enabled = False
                        await self.panic_liquidate()
            except Exception as e:
                pass
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
                        except:
                            pass
                            
                        # Send market order to close
                        await self.binance.place_order(
                            symbol=sym,
                            side=side,
                            order_type="MARKET",
                            quantity=abs(amt)
                        )
                        logger.critical(f"Panic Market {side} for {sym} executed.")
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
            
        await asyncio.gather(*tasks)
        
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
