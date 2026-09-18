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
                    
                    action_val = action_logits[0]
                    confidence = (action_logits[1] + 1.0) / 2.0
                    target_size = (action_logits[2] + 1.0) / 2.0
                    
                    if confidence < 0.3:
                        continue 
                        
                    margin_allocated = max(0, self.portfolio_state.free_margin) * target_size
                    pos = self.portfolio_state.positions[sym]
                    notional_requested = margin_allocated * pos.leverage
                    
                    side = "HOLD"
                    if action_val < -0.2: side = "CLOSE_LONG" if pos.quantity > 0 else "OPEN_SHORT"
                    elif action_val > 0.2: side = "CLOSE_SHORT" if pos.quantity < 0 else "OPEN_LONG"
                    
                    if side != "HOLD":
                        qty_raw = notional_requested / market.mid_price # Use EXPLICIT mid_price, no feature[6] hack
                        
                        request = OrderRequest(
                            symbol=sym, action_type=side, confidence=float(confidence),
                            requested_quantity=float(qty_raw), target_position=float(target_size),
                            model_version="v1", timestamp=time.time()
                        )
                        
                        # Evaluate Risk via Schema
                        decision = self.risk_manager.evaluate(request, self.portfolio_state, market)
                        
                        if decision.approved and decision.adjusted_quantity > 0:
                            qty_str = registry.get_symbol(sym).format_quantity(decision.adjusted_quantity)
                            logger.info(f"[{sym}] EXECUTING: {side} {qty_str} (Confidence: {confidence:.2f})")
                            
                            if (self.trading_enabled and 
                                not self.dry_run and 
                                (settings.allow_testnet or settings.allow_live_trading) and 
                                not settings.emergency_stop and 
                                self.has_valid_model and 
                                decision.approved):
                                
                                
                                # Phase 9: Execution and Experience Broadcasting
                                try:
                                    # Execute on Binance
                                    client_order_id = f"ai_bot_{uuid.uuid4().hex[:10]}"
                                    binance_side = "BUY" if "LONG" in side else "SELL"
                                    
                                    order_res = await self.binance.create_order(sym, binance_side, decision.adjusted_quantity, client_order_id)
                                    logger.info(f"[{sym}] ORDER SUCCESS: {order_res.get('orderId')}")
                                    
                                    # Record Experience
                                    exp_data = {
                                        "timestamp": time.time(),
                                        "symbol": sym,
                                        "market_state": market.features,
                                        "portfolio_state": [self.portfolio_state.wallet_balance, self.portfolio_state.equity], # Abbreviated
                                        "position_state": [pos.quantity, pos.entry_price],
                                        "action_type": side,
                                        "confidence": float(confidence),
                                        "requested_size": float(qty_raw),
                                        "approved_size": float(decision.adjusted_quantity),
                                        "model_version": "v1"
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
        
        tasks = [
            asyncio.create_task(self.listen_macro()),
            asyncio.create_task(self.inference_loop())
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
