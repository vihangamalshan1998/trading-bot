import asyncio
import json
import torch
import numpy as np
import time
import os
from typing import Dict, Any

from core.db.redis import redis_manager
from core.ai.registry import ModelRegistry
from apps.research.model import MultiSymbolActorCritic
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
        self.redis = redis_manager
        self.risk_manager = RiskManager()
        self.registry = ModelRegistry()
        self.binance = BinanceFuturesAdapter()
        
        # Disable Unsafe Execution explicitly
        self.dry_run = os.environ.get("TRADING_ENABLED", "false").lower() != "true"
        if not self.dry_run and not settings.binance_testnet:
            logger.error("SAFETY GATE: Live trading on mainnet is strictly disabled in this phase.")
            self.dry_run = True
            
        logger.info(f"Trading Mode: {'DRY_RUN' if self.dry_run else 'TESTNET'}")
        
        # Load the ActorCritic
        self.model = MultiSymbolActorCritic(num_symbols=self.num_symbols, macro_dim=8)
        self.model.eval()
        logger.info(f"Loaded MultiSymbolActorCritic for {self.num_symbols} symbols.")
        
        self.running = False
        self.event_memory = EventMemoryBuffer()
        
        # Tracking live states using Canonical Schemas
        self.portfolio_state = PortfolioState(
            wallet_balance=10000.0, equity=10000.0, used_margin=0.0,
            free_margin=10000.0, total_unrealized_pnl=0.0, total_exposure=0.0,
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
                    
    def _build_state_vector(self) -> torch.Tensor:
        obs = []
        
        # 1. Portfolio (9 dims) - using a normalized vector method 
        # (Assuming we migrate to_array() to the schema or use a helper, but for now we manually construct)
        obs.extend([
            self.portfolio_state.wallet_balance,
            self.portfolio_state.equity,
            self.portfolio_state.used_margin,
            self.portfolio_state.free_margin,
            self.portfolio_state.total_unrealized_pnl,
            self.portfolio_state.total_exposure,
            0.0, 0.0, 0.0 # Pad for 9 dims
        ])
        
        # 2. Market (25) + Position (12) per symbol
        for sym in self.symbols:
            if sym in self.market_states:
                m_state = self.market_states[sym]
                obs.extend(m_state.features) # Exactly 25
            else:
                obs.extend([0.0]*25)
                
            pos = self.portfolio_state.positions[sym]
            obs.extend([
                pos.quantity, pos.entry_price, pos.current_price, pos.unrealized_pnl,
                pos.realized_pnl, float(pos.leverage), pos.margin, pos.liquidation_price,
                0.0, 0.0, 0.0, 0.0 # Pad for 12 dims
            ])
            
        # 3. Macro (3 dims + 5 pad = 8 dims)
        obs.extend([
            self.macro_state.sentiment_score,
            self.macro_state.volatility_expectation,
            self.macro_state.regime,
            0.0, 0.0, 0.0, 0.0, 0.0 # Pad to 8
        ])
        
        # 4. Memory (5 dims)
        # obs.extend(self.event_memory.step().tolist()) # Disabled for exact dimension matching until fully validated
        
        # Dimension validation
        expected_dim = 9 + (self.num_symbols * 37) + 8
        actual_dim = len(obs)
        if actual_dim != expected_dim:
            logger.error(f"STATE DIMENSION MISMATCH: Expected {expected_dim}, got {actual_dim}. Halting inference.")
            return None
            
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
    async def inference_loop(self):
        logger.info("Starting production inference loop...")
        while self.running:
            try:
                # 1. Check if we have valid market data
                if len(self.market_states) != self.num_symbols:
                    logger.warning("Missing market data for some symbols. Skipping inference.")
                    await asyncio.sleep(5.0)
                    continue
                    
                state_tensor = self._build_state_vector()
                if state_tensor is None:
                    await asyncio.sleep(5.0)
                    continue
                
                with torch.no_grad():
                    action_logits, expected_return = self.model(state_tensor)
                    action_logits = action_logits[0].numpy()
                    
                for i, sym in enumerate(self.symbols):
                    market = self.market_states[sym]
                    
                    action_val = action_logits[i][0]
                    confidence = (action_logits[i][1] + 1.0) / 2.0
                    target_size = (action_logits[i][2] + 1.0) / 2.0
                    
                    if confidence < 0.3:
                        continue 
                        
                    margin_allocated = max(0, self.portfolio_state.free_margin) * target_size
                    pos = self.portfolio_state.positions[sym]
                    notional_requested = margin_allocated * pos.leverage
                    
                    side = "HOLD"
                    if action_val < -0.2: side = "CLOSE_LONG" if pos.quantity > 0 else "OPEN_SHORT"
                    elif action_val > 0.2: side = "CLOSE_SHORT" if pos.quantity < 0 else "OPEN_LONG"
                    
                    if side != "HOLD" and notional_requested > 10.0:
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
                            
                            if not self.dry_run:
                                # await self.binance.create_order(sym, "BUY" if "LONG" in side else "SELL", float(qty_str))
                                pass
                            else:
                                logger.info(f"[{sym}] DRY RUN: Blocked execution of {side} {qty_str}")
                        else:
                            logger.warning(f"[{sym}] Risk Manager REJECTED: {decision.reason}")
                            
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                
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
    bot = ProductionTradingBot(symbols=["BTCUSDT", "ETHUSDT"])
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        bot.stop()
