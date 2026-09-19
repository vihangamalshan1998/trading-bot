import asyncio
import json
import uuid
import torch
import numpy as np
import time
import os
from typing import Dict, Any

from core.db.redis import redis_manager
from core.ai.registry import ModelRegistry
from apps.research.model import SingleSymbolActorCritic
from core.risk.risk_manager import RiskManager
from core.exchange.binance_client import BinanceFuturesAdapter
from core.exchange.symbol_registry import registry
from core.logging.logger import logger
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
        self.model = SingleSymbolActorCritic(input_dim=28)
        try:
            # This calls the strictly validated loader that checks architecture and active status
            self.model = self.registry.load_model(self.model)
            self.model.eval()
            self.has_valid_model = True
            logger.info(f"Loaded rigorously validated SingleSymbolActorCritic for {self.num_symbols} symbols.")
        except Exception as e:
            logger.warning(f"No trained model checkpoint found yet (training in progress). Running in OBSERVATION-ONLY mode. Error: {e}")
            self.model.eval()
            self.dry_run = True  # Force safety - no trading without a trained model
            self.trading_enabled = False
        
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
                    
    def _build_state_vector(self, symbol: str) -> torch.Tensor:
        obs = []
        
        # 1. Portfolio (2 dims)
        obs.extend([self.portfolio_state.wallet_balance, self.portfolio_state.equity])
        
        # 2. Market Features (25 dims)
        if symbol in self.market_states:
            obs.extend(self.market_states[symbol].features)
        else:
            obs.extend([0.0] * 25)
            
        # 3. Position Before (1 dim)
        pos = self.portfolio_state.positions[symbol]
        obs.append(float(pos.quantity))
        
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
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
                pass # Fail silently so it doesn't crash the bot
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
                if len(self.market_states) != self.num_symbols:
                    logger.warning("Missing market data for some symbols. Skipping inference.")
                    await asyncio.sleep(5.0)
                    continue
                    
                for sym in self.symbols:
                    if sym not in self.market_states:
                        continue
                        
                    state_tensor = self._build_state_vector(sym)
                    
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
                            
                            if pnl_pct_leveraged <= -0.15: # -15% HARD STOP LOSS
                                logger.warning(f"[{sym}] 🛑 HARD STOP LOSS TRIGGERED: Position is down {pnl_pct_leveraged*100:.2f}%. Overriding AI.")
                                hard_stop_triggered = True
                                
                    import random
                    
                    if hard_stop_triggered:
                        # Force a MARKET CLOSE
                        action_val = -1.0 if pos_check.quantity > 0 else 1.0
                        confidence = 1.0
                        target_size = 0.0
                    elif random.random() < 0.02:
                        # Exploration Noise (2% chance to explore a random strategy)
                        action_val = random.uniform(-1.0, 1.0)
                        confidence = random.uniform(0.3, 1.0) # Ensure it passes the 0.3 threshold to trade
                        target_size = random.uniform(0.1, 1.0)
                        logger.info(f"[{sym}] EXPLORING: Applying curiosity noise to discover new strategies.")
                    else:
                        # Deterministic Policy (85% chance to use learned weights)
                        action_val = action_logits[0]
                        confidence = (action_logits[1] + 1.0) / 2.0
                        target_size = (action_logits[2] + 1.0) / 2.0
                    
                    # 1. Determine Predicted Side (Even if confidence is low, we want to know what it *leans* towards)
                    if action_val < -0.2: predicted_side = "SHORT"
                    elif action_val > 0.2: predicted_side = "LONG"
                    else: predicted_side = "HOLD"
                    
                    # 2. Publish AI internal state to Redis for the Dashboard
                    ai_state_data = {
                        "confidence": float(confidence),
                        "action_val": float(action_val),
                        "target_size": float(target_size),
                        "predicted_side": predicted_side
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
                            if notional_val < sym_config.min_notional:
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

                                    # Execute on Binance
                                    client_order_id = f"ai_bot_{uuid.uuid4().hex[:10]}"
                                    binance_side = "BUY" if side in ["OPEN_LONG", "CLOSE_SHORT"] else "SELL"
                                    
                                    order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id)
                                    logger.info(f"[{sym}] ORDER SUCCESS: {order_res.get('orderId')}")
                                    
                                    # Capture entry price before modifying position
                                    entry_px = self.portfolio_state.positions[sym].entry_price if sym in self.portfolio_state.positions else 0.0
                                    pos_leverage = self.portfolio_state.positions[sym].leverage if sym in self.portfolio_state.positions else 10
                                    
                                    # Update local position state immediately to prevent over-buying before the next sync
                                    notional_cost = float(qty_str) * market.mid_price
                                    margin_used = notional_cost / settings.max_leverage

                                    if "OPEN_LONG" in side:
                                        self.portfolio_state.positions[sym].quantity += float(qty_str)
                                        self.portfolio_state.free_margin -= margin_used
                                    elif "CLOSE_LONG" in side:
                                        self.portfolio_state.positions[sym].quantity -= float(qty_str)
                                        self.portfolio_state.free_margin += margin_used
                                    elif "OPEN_SHORT" in side:
                                        self.portfolio_state.positions[sym].quantity -= float(qty_str)
                                        self.portfolio_state.free_margin -= margin_used
                                    elif "CLOSE_SHORT" in side:
                                        self.portfolio_state.positions[sym].quantity += float(qty_str)
                                        self.portfolio_state.free_margin += margin_used
                                    
                                    # Log Trade History for Dashboard
                                    trade_record = {
                                        "timestamp": time.time(),
                                        "symbol": sym,
                                        "side": side,
                                        "quantity": float(qty_str),
                                        "price": market.mid_price,
                                        "confidence": float(confidence),
                                        "order_id": order_res.get('orderId')
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
                                        
                                    asyncio.create_task(redis_manager.redis.lpush("dashboard:trade_history", json.dumps(trade_record)))
                                    asyncio.create_task(redis_manager.redis.ltrim("dashboard:trade_history", 0, 99))
                                    
                                    # Calculate immediate RL Reward
                                    imm_reward = 0.0
                                    imm_pnl = 0.0
                                    if "CLOSE" in side and entry_px > 0:
                                        imm_reward = pnl
                                        imm_pnl = pnl
                                    elif "OPEN" in side:
                                        # Simple fee penalty proxy for opening a trade
                                        notional_cost = float(qty_str) * market.mid_price
                                        imm_reward = -(notional_cost * 0.0005) # 0.05% fee penalty
                                        
                                    # Record Experience
                                    exp_data = {
                                        "experience_id": str(uuid.uuid4()),
                                        "timestamp": int(time.time()),
                                        "symbol": sym,
                                        "market_state": market.features,
                                        "portfolio_state": [self.portfolio_state.wallet_balance, self.portfolio_state.equity], # Abbreviated
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
            
    async def start(self):
        self.running = True
        await registry.initialize_from_exchange(self.binance)
        
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
            asyncio.create_task(self.sync_dashboard())
        ]
        for sym in self.symbols:
            tasks.append(asyncio.create_task(self.listen_market(sym)))
            
        await asyncio.gather(*tasks)
        
    def stop(self):
        self.running = False

if __name__ == "__main__":
    from core.config.settings import settings
    bot = ProductionTradingBot(symbols=settings.symbol_universe)
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        bot.stop()
