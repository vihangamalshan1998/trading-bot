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
        self.last_update_time = time.time()
        
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
            "okx_momentum_lead": 0.0 
        }
        
        await redis_manager.redis.publish(f"statarb:state:{sym}", json.dumps(state))

    def _resolve_exchange_symbols(self, exchange, exchange_id: str):
        """
        Dynamically maps our 18-symbol universe to the exact symbol format used by the exchange.
        Handles ticker differences (e.g. OKX has PEPE/USDT:USDT with 1000x multiplier instead of 1000PEPE/USDT:USDT).
        """
        valid_symbols = []
        symbol_map = {}  # ccxt_symbol -> (canonical_sym, price_multiplier)
        
        for sym in self.symbols:
            if not sym.endswith("USDT"):
                continue
            base = sym[:-4]
            candidates = []
            
            # Standard futures symbol format (e.g. BTC/USDT:USDT)
            candidates.append((f"{base}/USDT:USDT", 1.0))
            
            # 1000-unit meme coin mapping (e.g. Binance has 1000PEPE, OKX has PEPE with 1000x multiplier)
            if base.startswith("1000"):
                raw_base = base[4:]
                candidates.append((f"{raw_base}/USDT:USDT", 1000.0))
                candidates.append((f"{raw_base}/USDT", 1000.0))
                
            # Rebranded / alternative tokens (e.g. RENDER vs RNDR)
            if base == "RENDER":
                candidates.append(("RNDR/USDT:USDT", 1.0))
            elif base == "RNDR":
                candidates.append(("RENDER/USDT:USDT", 1.0))
                
            # Spot fallback if futures contract doesn't exist
            candidates.append((f"{base}/USDT", 1.0))
            
            matched = False
            for ccxt_sym, mult in candidates:
                if exchange.markets and ccxt_sym in exchange.markets:
                    valid_symbols.append(ccxt_sym)
                    symbol_map[ccxt_sym] = (sym, mult)
                    matched = True
                    break
                    
            if not matched:
                logger.warning(f"[{exchange_id}] Symbol {sym} not available in {exchange_id} markets. Skipping for this exchange.")
                
        return valid_symbols, symbol_map

    async def watch_exchange(self, exchange_id: str):
        exchange_class = getattr(ccxt, exchange_id)
        exchange = None
        consecutive_errors = 0
        valid_symbols = []
        symbol_map = {}
        
        while self.running:
            try:
                if exchange is None:
                    logger.info(f"Connecting to {exchange_id}...")
                    exchange = exchange_class({'enableRateLimit': True})
                    await exchange.load_markets()
                    valid_symbols, symbol_map = self._resolve_exchange_symbols(exchange, exchange_id)
                    logger.info(f"[{exchange_id}] Successfully mapped {len(valid_symbols)}/{len(self.symbols)} active symbols.")
                    
                    if not valid_symbols:
                        logger.error(f"[{exchange_id}] No matching symbols found. Retrying in 60s...")
                        await asyncio.sleep(60)
                        continue
                
                # Timeout after 30s to detect and recover from deadlocked CCXT websocket connections
                tickers = await asyncio.wait_for(exchange.watch_tickers(valid_symbols), timeout=30.0)
                consecutive_errors = 0 # reset on successful tick
                
                for ccxt_sym, ticker in tickers.items():
                    if ccxt_sym in symbol_map:
                        sym, mult = symbol_map[ccxt_sym]
                        price = float(ticker.get('last', 0.0))
                        if price > 0:
                            self.prices[exchange_id][sym] = price * mult
                            self.last_update_time = time.time()
                            # If we just updated an external exchange, publish the new statarb state
                            if exchange_id != "binance":
                                await self._publish_state(sym)
                                
                # VPS CPU FIX: Rest for 0.5s between updates.
                await asyncio.sleep(0.5)
                
            except asyncio.TimeoutError:
                consecutive_errors += 1
                backoff = min(60, 5 * consecutive_errors)
                logger.warning(f"{exchange_id} watcher timed out (>30s no ticks). Errors: {consecutive_errors}. Backoff {backoff}s...")
                if exchange:
                    try:
                        await exchange.close()
                    except Exception:
                        pass
                    exchange = None
                await asyncio.sleep(backoff)
                
            except Exception as e:
                consecutive_errors += 1
                backoff = min(60, 5 * consecutive_errors)
                logger.error(f"{exchange_id} watcher error: {e}. Consecutive errors: {consecutive_errors}. Backoff {backoff}s...")
                if exchange:
                    try:
                        await exchange.close()
                    except Exception:
                        pass
                    exchange = None
                await asyncio.sleep(backoff)
                
        if exchange:
            try:
                await exchange.close()
            except Exception:
                pass

    async def _watchdog_loop(self):
        """Monitors overall statarb health and publishes heartbeats to Redis."""
        while self.running:
            try:
                now_t = time.time()
                if redis_manager.redis:
                    await redis_manager.redis.set("statarb:heartbeat", str(now_t), ex=60)
                
                # If no price updates have arrived from ANY exchange for > 120s, exit for PM2 auto-restart
                if self.last_update_time > 0 and (now_t - self.last_update_time) > 120.0:
                    import sys
                    logger.critical(f"🚨 StatArb Collector freeze detected: No ticks from any exchange for {now_t - self.last_update_time:.1f}s. Exiting for PM2 auto-recovery...")
                    sys.exit(1)
            except Exception:
                pass
            await asyncio.sleep(15.0)

    async def start(self):
        self.running = True
        await redis_manager.connect()
        asyncio.create_task(self._watchdog_loop())
        
        # We watch Binance (baseline), OKX, and Bybit
        tasks = [
            asyncio.create_task(self.watch_exchange("binance")),
            asyncio.create_task(self.watch_exchange("okx")),
            asyncio.create_task(self.watch_exchange("bybit"))
        ]
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    import sys
    set_log_file("logs/statarb_collector.log")
    collector = StatArbCollector()
    asyncio.run(collector.start())
