import asyncio
import json
import time
import ccxt.pro as ccxt
from core.db.redis import redis_manager
from core.config.settings import settings
from core.logging.logger import logger, set_log_file

class StatArbCollector:
    def __init__(self):
        self.symbols = settings.symbol_universe
        self.running = False
        
        # We need the format CCXT uses (e.g., 'BTC/USDT:USDT' for futures)
        self.ccxt_symbols = []
        self.symbol_map = {}
        for sym in self.symbols:
            # Assuming sym is like BTCUSDT
            if sym.endswith("USDT"):
                base = sym[:-4]
                ccxt_sym = f"{base}/USDT:USDT"
                self.ccxt_symbols.append(ccxt_sym)
                self.symbol_map[ccxt_sym] = sym
                
        # Store latest prices
        self.prices = {
            "binance": {sym: 0.0 for sym in self.symbols},
            "okx": {sym: 0.0 for sym in self.symbols},
            "bybit": {sym: 0.0 for sym in self.symbols}
        }
        
    async def _publish_state(self, sym: str):
        p_binance = self.prices["binance"][sym]
        p_okx = self.prices["okx"][sym]
        p_bybit = self.prices["bybit"][sym]
        
        if p_binance == 0.0:
            return
            
        okx_premium = (p_okx - p_binance) / p_binance if p_okx > 0 else 0.0
        bybit_premium = (p_bybit - p_binance) / p_binance if p_bybit > 0 else 0.0
        
        state = {
            "timestamp": time.time(),
            "okx_premium": okx_premium,
            "bybit_premium": bybit_premium,
            # Momentum lead could be added by tracking history, keeping it simple for now
            "okx_momentum_lead": 0.0 
        }
        
        await redis_manager.redis.publish(f"statarb:state:{sym}", json.dumps(state))

    async def watch_exchange(self, exchange_id: str):
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        
        logger.info(f"Connecting to {exchange_id}...")
        
        while self.running:
            try:
                # CCXT watch_tickers returns a dict of tickers
                tickers = await exchange.watch_tickers(self.ccxt_symbols)
                for ccxt_sym, ticker in tickers.items():
                    if ccxt_sym in self.symbol_map:
                        sym = self.symbol_map[ccxt_sym]
                        price = float(ticker.get('last', 0.0))
                        if price > 0:
                            self.prices[exchange_id][sym] = price
                            # If we just updated an external exchange, publish the new statarb state
                            if exchange_id != "binance":
                                await self._publish_state(sym)
                                
            except Exception as e:
                logger.error(f"{exchange_id} watcher error: {e}")
                await asyncio.sleep(5)
                
        await exchange.close()

    async def start(self):
        self.running = True
        await redis_manager.connect()
        
        # We watch Binance (baseline), OKX, and Bybit
        tasks = [
            asyncio.create_task(self.watch_exchange("binance")),
            asyncio.create_task(self.watch_exchange("okx")),
            asyncio.create_task(self.watch_exchange("bybit"))
        ]
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    set_log_file("logs/statarb_collector.log")
    collector = StatArbCollector()
    asyncio.run(collector.start())
