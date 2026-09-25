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
        self.running = False
        
    def _clean_old_trades(self, sym: str, current_time: float):
        """Removes trades older than 60 seconds from the rolling window."""
        while self.whale_trades[sym] and current_time - self.whale_trades[sym][0]['time'] > 60:
            self.whale_trades[sym].popleft()

    async def _publish_state(self, sym: str, buy_wall_dist: float, sell_wall_dist: float):
        current_time = time.time()
        self._clean_old_trades(sym, current_time)
        
        # Calculate Rolling 60s Pressure
        whale_buy_pressure = sum(t['amount'] for t in self.whale_trades[sym] if t['side'] == 'BUY')
        whale_sell_pressure = sum(t['amount'] for t in self.whale_trades[sym] if t['side'] == 'SELL')
        
        state = {
            "timestamp": current_time,
            "buy_wall_distance": buy_wall_dist,
            "sell_wall_distance": sell_wall_dist,
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
            
        stream_url = f"wss://stream.binancefuture.com/stream?streams={'/'.join(streams)}"
        logger.info(f"Connecting to Whale Streams: {stream_url}")
        
        # Store latest orderbook state
        latest_walls = {sym: {'buy': 0.0, 'sell': 0.0} for sym in self.symbols}
        
        async with websockets.connect(stream_url, ping_interval=20, ping_timeout=10) as ws:
            logger.info("Whale Tracker Connected to Binance!")
            while self.running:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    stream = data['stream']
                    payload = data['data']
                    
                    sym = payload.get('s', '')
                    if not sym or sym not in self.symbols:
                        continue
                        
                    if '@aggTrade' in stream:
                        price = float(payload['p'])
                        qty = float(payload['q'])
                        usd_val = price * qty
                        
                        if usd_val >= WHALE_TRADE_THRESHOLD_USD:
                            is_buyer_maker = payload['m']
                            side = "SELL" if is_buyer_maker else "BUY"
                            self.whale_trades[sym].append({
                                'time': time.time(),
                                'side': side,
                                'amount': usd_val
                            })
                            logger.debug(f"🚨 WHALE MARKET {side}: {usd_val:,.2f} USD on {sym}")
                            
                            # Publish immediately on a whale trade
                            await self._publish_state(sym, latest_walls[sym]['buy'], latest_walls[sym]['sell'])

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
                        
                except Exception as e:
                    logger.error(f"Error processing whale ws msg: {e}")

    async def start(self):
        self.running = True
        await redis_manager.connect()
        while self.running:
            try:
                await self.connect_binance_ws()
            except Exception as e:
                logger.error(f"Whale stream disconnected: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

if __name__ == "__main__":
    set_log_file("logs/whale_tracker.log")
    tracker = WhaleTracker()
    asyncio.run(tracker.start())
