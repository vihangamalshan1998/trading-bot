import asyncio
import json
import torch
import numpy as np
import time
from core.db.redis import redis_manager
from core.ai.registry import ModelRegistry
from apps.research.model import MultiSymbolActorCritic
from core.risk.risk_manager import RiskManager
from core.exchange.binance_client import BinanceFuturesClient
from core.exchange.symbol_registry import registry
from core.logging.logger import logger
from core.ai.memory import EventMemoryBuffer
from apps.research.environment import PortfolioState, PositionState

class ProductionTradingBot:
    """
    Phase 12: Live execution engine for the fully upgraded RL model.
    Constructs the dense (9 + N*37 + 8) state vector, queries PyTorch,
    intercepts actions via the Risk Engine, and executes via Binance.
    """
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.redis = redis_manager
        self.risk_manager = RiskManager()
        self.registry = ModelRegistry()
        self.binance = BinanceFuturesClient()
        
        # Load the ActorCritic
        self.model = MultiSymbolActorCritic(num_symbols=self.num_symbols, macro_dim=8)
        self.model.eval()
        logger.info(f"Loaded MultiSymbolActorCritic for {self.num_symbols} symbols.")
        
        self.running = False
        self.event_memory = EventMemoryBuffer()
        
        # Tracking live states
        self.portfolio_state = PortfolioState(initial_balance=10000.0) # Assume 10k paper for now
        self.positions = {sym: PositionState(sym) for sym in self.symbols}
        self.macro_state = {"sentiment_score": 0.0, "volatility_expectation": 0.5, "regime": 0.0}
        self.market_features = {sym: np.zeros(25) for sym in self.symbols}
        
    async def listen_macro(self):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe("macro:state:global")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    self.macro_state["sentiment_score"] = payload.get("sentiment_score", 0.0)
                    self.macro_state["volatility_expectation"] = payload.get("volatility_expectation", 0.5)
                    self.macro_state["regime"] = payload.get("regime", 0.0)
                except Exception as e:
                    logger.error(f"Error parsing macro: {e}")
                    
    async def listen_market(self, symbol: str):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe(f"market:features:{symbol}")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    features = payload.get("features", [])
                    if len(features) == 25:
                        self.market_features[symbol] = np.array(features, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Error parsing market features: {e}")
                    
    def _build_state_vector(self) -> torch.Tensor:
        obs = []
        
        # 1. Portfolio (9 dims)
        obs.extend(self.portfolio_state.to_array().tolist())
        
        # 2. Market (25) + Position (12) per symbol
        for sym in self.symbols:
            obs.extend(self.market_features[sym].tolist())
            obs.extend(self.positions[sym].to_array(portfolio_equity=self.portfolio_state.equity).tolist())
            
        # 3. Macro (3 dims)
        obs.extend([
            self.macro_state["sentiment_score"],
            self.macro_state["volatility_expectation"],
            self.macro_state["regime"]
        ])
        
        # 4. Memory (5 dims)
        obs.extend(self.event_memory.step().tolist())
        
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
    async def inference_loop(self):
        logger.info("Starting production inference loop...")
        while self.running:
            try:
                state_tensor = self._build_state_vector()
                
                with torch.no_grad():
                    # action_logits shape: (1, num_symbols, 3), value shape: (1, 1)
                    action_logits, expected_return = self.model(state_tensor)
                    action_logits = action_logits[0].numpy()
                    
                for i, sym in enumerate(self.symbols):
                    # Extract continuous values
                    action_val = action_logits[i][0]
                    confidence = (action_logits[i][1] + 1.0) / 2.0
                    target_size = (action_logits[i][2] + 1.0) / 2.0
                    
                    if confidence < 0.3:
                        continue # Skip uncertain trades
                        
                    # Calculate proposed notional based on target size and free margin
                    margin_allocated = max(0, self.portfolio_state.free_margin) * target_size
                    notional_requested = margin_allocated * self.positions[sym].leverage
                    
                    # Decide side based on action_val threshold (using same logic as Env)
                    side = None
                    if action_val < -0.2: side = "SELL"
                    elif action_val > 0.2: side = "BUY"
                    
                    if side and notional_requested > 10.0: # Minimum order threshold
                        # Ask Risk Manager for approval
                        approved_notional = self.risk_manager.evaluate_risk(
                            symbol=sym,
                            proposed_notional=notional_requested,
                            side=side,
                            portfolio_equity=self.portfolio_state.equity,
                            current_exposure=self.portfolio_state.total_exposure
                        )
                        
                        if approved_notional > 0:
                            # Convert to formatted quantity using SymbolRegistry
                            mid_price = self.market_features[sym][6] if self.market_features[sym][6] > 0 else 1.0 # fallback
                            qty_raw = approved_notional / mid_price
                            qty_str = registry.get_symbol(sym).format_quantity(qty_raw)
                            
                            logger.info(f"[{sym}] EXECUTING: {side} {qty_str} (Confidence: {confidence:.2f})")
                            # await self.binance.create_order(sym, side, float(qty_str)) # Disabled for safety
                        else:
                            logger.warning(f"[{sym}] Risk Manager BLOCKED {side} order.")
                            
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                
            await asyncio.sleep(5.0) # Evaluate every 5 seconds
            
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
