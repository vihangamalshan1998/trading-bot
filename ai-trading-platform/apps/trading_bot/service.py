import asyncio
import json
from typing import Dict, Any

from core.logging.logger import logger
from core.config.settings import settings
from core.db.redis import redis_manager
from core.risk.manager import RiskManager
from apps.market_collector.binance import BinanceSpotAdapter
from apps.trading_bot.strategy import BaseStrategy, DummySpreadStrategy

class TradingBotService:
    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.adapter = BinanceSpotAdapter()
        self.risk_manager = RiskManager(max_position_usd=1000.0, max_drawdown_usd=50.0)
        self.symbol = settings.trading_symbol
        self.running = False
        
    async def start(self):
        logger.info("Starting Trading Bot Service...", extra={"symbol": self.symbol})
        self.running = True
        
        await redis_manager.connect()
        state_key = f"market:state:{self.symbol}"
        
        # Start the background publisher task
        publisher_task = asyncio.create_task(self._publish_portfolio_loop())
        
        try:
            while self.running:
                # Poll state from Redis
                if redis_manager.redis is not None:
                    state_raw = await redis_manager.redis.get(state_key)
                    if state_raw:
                        state = json.loads(state_raw)
                        await self._evaluate_state(state)
                
                # Poll frequently to react quickly, but not so fast as to burn CPU
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("Trading Bot Service shutting down...")
        finally:
            publisher_task.cancel()
            await self.adapter.close()
            await redis_manager.disconnect()
            
    async def _publish_portfolio_loop(self):
        """Continuously publishes the live portfolio state to Redis for the Dashboard."""
        while self.running:
            try:
                if redis_manager.redis:
                    state = self.risk_manager.get_portfolio_summary()
                    await redis_manager.redis.set("dashboard:portfolio", json.dumps(state))
            except Exception as e:
                logger.error(f"Error publishing portfolio state: {e}")
            await asyncio.sleep(1.0)
            
    async def _evaluate_state(self, state: Dict[str, Any]):
        order_intent = self.strategy.evaluate(state)
        
        if order_intent:
            side = order_intent["side"]
            quantity = order_intent["quantity"]
            price = order_intent["price"]
            
            logger.info("Strategy generated signal", extra=order_intent)
            
            # Pass through Risk Manager
            approved, reason = self.risk_manager.approve_order(self.symbol, side, quantity, price)
            
            if approved:
                logger.info("Order approved by Risk Manager, executing...")
                try:
                    # In dry-run mode or for safety, we can wrap this in a config check.
                    # Since this is Testnet and we are testing pipeline:
                    result = await self.adapter.place_order(
                        symbol=self.symbol,
                        side=side,
                        order_type="LIMIT",
                        quantity=quantity,
                        price=price
                    )
                    logger.info("Order executed successfully", extra={"order_result": result})
                    
                    # Update risk manager state
                    self.risk_manager.update_position(self.symbol, side, quantity, price)
                    
                except Exception as e:
                    logger.error("Failed to execute order", exc_info=True)
            else:
                logger.warning("Order rejected by Risk Manager", extra={"reason": reason})

    def stop(self):
        self.running = False
