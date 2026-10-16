import asyncio
import json
import time
import websockets
import logging
from collections import deque
from core.db.redis import redis_manager
from core.config.settings import settings
from core.logging.logger import logger, set_log_file

WHALE_TRADE_THRESHOLD_USD = 100_000.0  # $100k+ is considered a whale market order
WHALE_WALL_THRESHOLD_USD = 500_000.0   # $500k+ limit order is considered a whale wall

class WhaleTracker:
    def __init__(self):
        self.symbols = settings.symbol_universe
        self.whale_trades = {sym: deque(maxlen=1000) for sym in self.symbols}
        self.last_published = {sym: 0.0 for sym in self.symbols}
        self.running = False
        
    def _clean_old_trades(self, sym: str, current_time: float):
        """Removes trades older than 60 seconds from the rolling window."""
        while self.whale_trades[sym] and current_time - self.whale_trades[sym][0]['time'] > 60:
            self.whale_trades[sym].popleft()

    async def _publish_state(self, sym: str, buy_wall_dist: float, sell_wall_dist: float, force: bool = False):
        current_time = time.time()
        # VPS CPU & REDIS THROTTLE: Only publish depth state once per second per symbol unless forced by a whale trade
        if not force and (current_time - self.last_published.get(sym, 0.0)) < 1.0:
            return
            
        self.last_published[sym] = current_time
        self._clean_old_trades(sym, current_time)
        
        # Calculate Rolling 60s Pressure with institutional sanity caps ($10M max window to prevent synthetic testnet distortion)
        whale_buy_pressure = min(10_000_000.0, float(sum(t['amount'] for t in self.whale_trades[sym] if t['side'] == 'BUY')))
        whale_sell_pressure = min(10_000_000.0, float(sum(t['amount'] for t in self.whale_trades[sym] if t['side'] == 'SELL')))
        
        state = {
            "timestamp": current_time,
            "buy_wall_distance": float(buy_wall_dist),
            "sell_wall_distance": float(sell_wall_dist),
            "whale_buy_pressure": whale_buy_pressure,
            "whale_sell_pressure": whale_sell_pressure
        }
        
        # Publish to Redis so main.py can pick it up
        await redis_manager.redis.publish(f"whale:state:{sym}", json.dumps(state))

    async def connect_binance_ws(self):
        streams = []
        for sym in self.symbols:
            s_low = sym.lower()
            streams.append(f"{s_low}@aggTrade")
            streams.append(f"{s_low}@depth20@100ms")
            
        stream_url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        logger.info(f"Connecting to LIVE Mainnet Whale Streams: {stream_url}")
        
        # Store latest orderbook state
        latest_walls = {sym: {'buy': 0.0, 'sell': 0.0} for sym in self.symbols}
        
        async with websockets.connect(stream_url, ping_interval=20, ping_timeout=10) as ws:
            logger.info("Whale Tracker Connected to Binance!")
            while self.running:
                try:
                    # Timeout after 30s to detect and recover from silent TCP/WS freezes
                    msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    data = json.loads(msg)
                    stream = data['stream']
                    payload = data['data']
                    
                    sym = payload.get('s', '')
                    if not sym or sym not in self.symbols:
                        continue
                        
                    if '@aggTrade' in stream:
                        price = float(payload['p'])
                        qty = float(payload['q'])
                        # Cap single trade USD value to $2,000,000 to prevent single-order testnet distortions
                        raw_usd = price * qty
                        usd_val = min(raw_usd, 2_000_000.0)
                        
                        if usd_val >= WHALE_TRADE_THRESHOLD_USD:
                            is_buyer_maker = payload['m']
                            side = "SELL" if is_buyer_maker else "BUY"
                            self.whale_trades[sym].append({
                                'time': time.time(),
                                'side': side,
                                'amount': usd_val
                            })
                            logger.debug(f"🚨 WHALE MARKET {side}: {usd_val:,.2f} USD on {sym} (raw: {raw_usd:,.2f})")
                            
                            # Publish immediately on a whale trade
                            await self._publish_state(sym, latest_walls[sym]['buy'], latest_walls[sym]['sell'], force=True)

                    elif '@depth' in stream:
                        bids = payload.get('b', [])
                        asks = payload.get('a', [])
                        
                        mid_price = 0.0
                        if bids and asks:
                            mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2.0
                            
                        # Find closest Buy Wall
                        buy_wall_dist = 0.0
                        for bid in bids:
                            px, qty = float(bid[0]), float(bid[1])
                            if px * qty >= WHALE_WALL_THRESHOLD_USD:
                                buy_wall_dist = (mid_price - px) / mid_price
                                break
                                
                        # Find closest Sell Wall
                        sell_wall_dist = 0.0
                        for ask in asks:
                            px, qty = float(ask[0]), float(ask[1])
                            if px * qty >= WHALE_WALL_THRESHOLD_USD:
                                sell_wall_dist = (px - mid_price) / mid_price
                                break
                                
                        latest_walls[sym]['buy'] = buy_wall_dist
                        latest_walls[sym]['sell'] = sell_wall_dist
                        
                        # Periodically publish (e.g. on depth update)
                        await self._publish_state(sym, buy_wall_dist, sell_wall_dist)
                        
                except asyncio.TimeoutError:
                    logger.warning("Whale Tracker: No websocket message received in 30s (silent freeze). Reconnecting...")
                    break
                except (websockets.ConnectionClosed, websockets.ConnectionClosedError, websockets.ConnectionClosedOK) as e:
                    logger.warning(f"Whale Tracker: Websocket closed ({e}). Reconnecting cleanly...")
                    break
                except Exception as e:
                    logger.error(f"Error processing whale ws msg: {e}. Breaking to reconnect cleanly...")
                    await asyncio.sleep(1.0)
                    break

    async def _heartbeat_loop(self):
        """Publishes a heartbeat to Redis every 10s for the health watchdog."""
        while self.running:
            try:
                if redis_manager.redis:
                    await redis_manager.redis.set("whale:heartbeat", str(time.time()), ex=60)
            except Exception:
                pass
            await asyncio.sleep(10.0)

    async def start(self):
        self.running = True
        await redis_manager.connect()
        asyncio.create_task(self._heartbeat_loop())
        
        consecutive_failures = 0
        while self.running:
            try:
                await self.connect_binance_ws()
                consecutive_failures = 0 # reset on successful connection
            except Exception as e:
                consecutive_failures += 1
                backoff = min(60, 5 * consecutive_failures)
                logger.error(f"Whale stream disconnected: {e}. Consecutive failures: {consecutive_failures}. Reconnecting in {backoff}s...")
                
                # If network is persistently unreachable (6 consecutive failures ~ 2 mins),
                # exit process with code 1 so PM2 automatically reboots the bot cleanly!
                if consecutive_failures >= 6:
                    import sys
                    logger.critical("🚨 Whale Tracker exceeded 6 consecutive connection failures. Exiting with code 1 for PM2 auto-restart...")
                    sys.exit(1)
                    
                await asyncio.sleep(backoff)

if __name__ == "__main__":
    import sys
    set_log_file("logs/whale_tracker.log")
    tracker = WhaleTracker()
    asyncio.run(tracker.start())
