import asyncio
import json
import time
import websockets
from core.exchange.binance_streams import single_stream_url
import collections
from typing import Dict, Any, List
from core.db.redis import redis_manager
from core.config.settings import settings
from core.logging.logger import logger, set_log_file

class MultiExchangeLiquidationCollector:
    """
    Multi-Exchange Real-Time Liquidation Stream Aggregator.
    Subscribes to live liquidation feeds across:
    1. Binance Futures (!forceOrder@arr - market-wide)
    2. Bybit Futures (v5 linear liquidation stream)
    3. OKX Futures (v5 SWAP liquidation-orders)
    
    Maintains rolling 1-minute and 3-minute aggregated liquidation USD volumes
    and detects institutional cascades ($20M+ long liquidations in < 3 minutes).
    """

    def __init__(self):
        self.symbols = set(settings.symbol_universe)
        self.running = False
        
        # Rolling liquidation records: (timestamp, exchange, symbol, side, usd_value, price)
        # side: "SELL" = Long liquidation (forced market sell)
        # side: "BUY"  = Short liquidation (forced market buy)
        self.raw_liquidations = collections.deque(maxlen=10000)
        
        # Per symbol rolling metrics
        self.symbol_stats = {
            sym: {
                "long_liq_1m": 0.0,
                "short_liq_1m": 0.0,
                "long_liq_3m": 0.0,
                "short_liq_3m": 0.0,
                "peak_velocity_long": 0.0,
                "lowest_cascade_price": float("inf"),
                "highest_cascade_price": 0.0,
                "last_event_time": 0.0
            }
            for sym in self.symbols
        }
        
        # Global market metrics
        self.global_stats = {
            "long_liq_1m": 0.0,
            "short_liq_1m": 0.0,
            "long_liq_3m": 0.0,
            "short_liq_3m": 0.0,
            "peak_velocity_long_3m": 0.0,
            "is_cascade_active": False,
            "cascade_type": "NONE",  # LONG_FLUSH, SHORT_SQUEEZE, NONE
            "cascade_started_at": 0.0,
            "total_cascade_usd": 0.0,
            "timestamp": time.time()
        }
        
        self.last_published = 0.0

    def _normalize_symbol(self, raw_symbol: str) -> str:
        """Converts exchange specific symbols (e.g. BTC-USDT-SWAP, BTCUSDT) to canonical BTCUSDT."""
        s = raw_symbol.upper().replace("-", "").replace("_", "").replace("/", "")
        if "SWAP" in s:
            s = s.replace("SWAP", "")
        return s

    def record_liquidation(self, exchange: str, symbol: str, side: str, usd_value: float, price: float):
        """Records a normalized liquidation event."""
        canonical_sym = self._normalize_symbol(symbol)
        now = time.time()
        
        # Record raw event
        self.raw_liquidations.append({
            "timestamp": now,
            "exchange": exchange,
            "symbol": canonical_sym,
            "side": side.upper(),  # SELL = Long forced liquidation, BUY = Short forced liquidation
            "usd": float(usd_value),
            "price": float(price)
        })

    def _prune_and_aggregate(self):
        """Prunes events older than 3 minutes (180s) and calculates rolling aggregate metrics."""
        now = time.time()
        
        # Prune events older than 180 seconds
        while self.raw_liquidations and (now - self.raw_liquidations[0]["timestamp"]) > 180.0:
            self.raw_liquidations.popleft()
            
        # Reset symbol aggregations (the cascade price extremes too: they used to keep the extreme of every
        # cascade since the service started)
        for sym in self.symbols:
            st = self.symbol_stats[sym]
            st["long_liq_1m"] = 0.0
            st["short_liq_1m"] = 0.0
            st["long_liq_3m"] = 0.0
            st["short_liq_3m"] = 0.0
            st["lowest_cascade_price"] = float("inf")
            st["highest_cascade_price"] = 0.0
            
        g_long_1m = 0.0
        g_short_1m = 0.0
        g_long_3m = 0.0
        g_short_3m = 0.0
        
        for liq in self.raw_liquidations:
            age = now - liq["timestamp"]
            usd = liq["usd"]
            sym = liq["symbol"]
            side = liq["side"]
            px = liq["price"]
            
            # Aggregate 3-minute window
            if side == "SELL":
                g_long_3m += usd
                if sym in self.symbol_stats:
                    self.symbol_stats[sym]["long_liq_3m"] += usd
                    if px > 0:
                        self.symbol_stats[sym]["lowest_cascade_price"] = min(
                            self.symbol_stats[sym]["lowest_cascade_price"], px
                        )
            else:
                g_short_3m += usd
                if sym in self.symbol_stats:
                    self.symbol_stats[sym]["short_liq_3m"] += usd
                    if px > 0:
                        self.symbol_stats[sym]["highest_cascade_price"] = max(
                            self.symbol_stats[sym]["highest_cascade_price"], px
                        )
                        
            # Aggregate 1-minute window
            if age <= 60.0:
                if side == "SELL":
                    g_long_1m += usd
                    if sym in self.symbol_stats:
                        self.symbol_stats[sym]["long_liq_1m"] += usd
                else:
                    g_short_1m += usd
                    if sym in self.symbol_stats:
                        self.symbol_stats[sym]["short_liq_1m"] += usd

        self.global_stats["long_liq_1m"] = g_long_1m
        self.global_stats["short_liq_1m"] = g_short_1m
        self.global_stats["long_liq_3m"] = g_long_3m
        self.global_stats["short_liq_3m"] = g_short_3m
        self.global_stats["timestamp"] = now
        
        # Track peak 1-minute velocity
        current_long_velocity = g_long_1m / 60.0
        if current_long_velocity > self.global_stats["peak_velocity_long_3m"]:
            self.global_stats["peak_velocity_long_3m"] = current_long_velocity
            
        # Institutional Cascade Trigger ($20M+ Long Liquidations in 3 minutes)
        threshold = settings.liquidation_cascade_threshold_usd
        if g_long_3m >= threshold:
            if not self.global_stats["is_cascade_active"]:
                self.global_stats["is_cascade_active"] = True
                self.global_stats["cascade_type"] = "LONG_FLUSH"
                self.global_stats["cascade_started_at"] = now
                self.global_stats["total_cascade_usd"] = g_long_3m
                logger.critical(
                    f"🌊 LIQUIDATION CASCADE DETECTED: ${g_long_3m:,.0f} Longs liquidated in < 3m! "
                    f"Monitoring for bottom-wick exhaustion..."
                )
        elif g_long_3m < (threshold * 0.40):
            # Cascade cleared
            self.global_stats["is_cascade_active"] = False
            self.global_stats["cascade_type"] = "NONE"
            self.global_stats["peak_velocity_long_3m"] = current_long_velocity

    async def _publish_to_redis(self):
        """Publishes aggregated metrics to Redis every second."""
        if not redis_manager.redis:
            return
            
        try:
            # Publish global state
            payload_global = json.dumps(self.global_stats)
            await redis_manager.redis.publish("liquidation:cascade:global", payload_global)
            await redis_manager.redis.set("liquidation:state:global", payload_global)
            
            # Publish individual symbol stats
            for sym, stats in self.symbol_stats.items():
                if stats["long_liq_3m"] > 0 or stats["short_liq_3m"] > 0:
                    stats_copy = dict(stats)
                    stats_copy["timestamp"] = time.time()
                    payload_sym = json.dumps(stats_copy)
                    await redis_manager.redis.publish(f"liquidation:cascade:{sym}", payload_sym)
                    await redis_manager.redis.set(f"liquidation:state:{sym}", payload_sym)
        except Exception as e:
            logger.error(f"Error publishing liquidation state to Redis: {e}")

    # =========================================================================
    # Exchange 1: Binance Futures All-Market Liquidation Stream
    # =========================================================================
    async def listen_binance(self):
        # Mainnet liquidations are only delivered on the /market endpoint (legacy URL connects but stays silent)
        url = single_stream_url("!forceOrder@arr", mainnet=True)
        while self.running:
            try:
                logger.info(f"Connecting to Binance All-Market Liquidation Stream: {url}")
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("✅ Connected to Binance Liquidation Feed!")
                    async for raw_msg in ws:
                        if not self.running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            order = msg.get("o", {})
                            sym = order.get("s", "")
                            side = order.get("S", "")  # SELL = Long liq, BUY = Short liq
                            qty = float(order.get("q", 0.0))
                            price = float(order.get("p", 0.0))
                            usd_val = qty * price
                            
                            if usd_val > 0:
                                self.record_liquidation("binance", sym, side, usd_val, price)
                        except Exception as e:
                            logger.error(f"Error parsing Binance liquidation: {e}")
            except Exception as e:
                logger.warning(f"Binance liquidation stream disconnected ({e}). Reconnecting in 5s...")
                await asyncio.sleep(5)

    # =========================================================================
    # Exchange 2: Bybit Linear Liquidation Stream
    # =========================================================================
    async def listen_bybit(self):
        # Bybit retired the "liquidation.<symbol>" topic (subscribing answers "handler not found", so this feed
        # received nothing). "allLiquidation.<symbol>" pushes a list of {s, S (liquidated position side), v, p}.
        url = "wss://stream.bybit.com/v5/public/linear"
        while self.running:
            try:
                logger.info(f"Connecting to Bybit Linear Liquidation Stream: {url}")
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    sub_symbols = sorted(self.symbols)
                    for i in range(0, len(sub_symbols), 10):  # Bybit takes at most 10 topics per request
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [f"allLiquidation.{sym}" for sym in sub_symbols[i:i + 10]],
                        }))
                    logger.info("✅ Connected & Subscribed to Bybit Liquidation Feed!")

                    async for raw_msg in ws:
                        if not self.running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            if msg.get("op") == "subscribe" and not msg.get("success", True):
                                logger.error(f"Bybit liquidation subscribe rejected: {msg.get('ret_msg')}")
                                continue
                            data = msg.get("data")
                            for item in (data if isinstance(data, list) else []):
                                sym = item.get("s", "")
                                # S = side of the liquidated POSITION: Buy = a long was liquidated (forced sell)
                                side = "SELL" if str(item.get("S", "")).upper() == "BUY" else "BUY"
                                size = float(item.get("v") or 0.0)
                                price = float(item.get("p") or 0.0)
                                usd_val = size * price
                                if usd_val > 0:
                                    self.record_liquidation("bybit", sym, side, usd_val, price)
                        except Exception as e:
                            logger.error(f"Error parsing Bybit liquidation: {e}")
            except Exception as e:
                logger.warning(f"Bybit liquidation stream disconnected ({e}). Reconnecting in 5s...")
                await asyncio.sleep(5)

    # =========================================================================
    # Exchange 3: OKX SWAP Liquidation Stream
    # =========================================================================
    @staticmethod
    def _okx_contract_sizes() -> Dict[str, float]:
        """{instId: coins per contract} for OKX USDT swaps (e.g. BTC-USDT-SWAP 0.01, PEPE-USDT-SWAP 10,000,000)."""
        import urllib.request
        req = urllib.request.Request("https://www.okx.com/api/v5/public/instruments?instType=SWAP",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            rows = json.load(resp).get("data", [])
        return {r["instId"]: float(r["ctVal"]) for r in rows
                if r.get("settleCcy") == "USDT" and r.get("ctValCcy") != "USD" and float(r.get("ctVal") or 0) > 0}

    async def listen_okx(self):
        url = "wss://ws.okx.com:8443/ws/v5/public"
        contract_sizes: Dict[str, float] = {}
        while self.running:
            try:
                if not contract_sizes:
                    # OKX reports liquidations in CONTRACTS: sz x price overstated BTC 100x and understated PEPE 10^7x
                    contract_sizes = await asyncio.to_thread(self._okx_contract_sizes)
                    logger.info(f"OKX contract sizes loaded for {len(contract_sizes)} USDT swaps.")
                logger.info(f"Connecting to OKX Liquidation Stream: {url}")
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    sub_msg = {
                        "op": "subscribe",
                        "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info("✅ Connected & Subscribed to OKX Liquidation Feed!")
                    
                    async for raw_msg in ws:
                        if not self.running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            data_list = msg.get("data", [])
                            for item in data_list:
                                inst_id = item.get("instId", "")
                                ct_val = contract_sizes.get(inst_id)
                                if not ct_val:
                                    continue  # not a USDT swap with a known contract size
                                # liquidation-orders: details[] holds the fills (side, sz in contracts, bkPx)
                                for det in item.get("details") or [item]:
                                    side = str(det.get("side", "")).upper()  # sell = a long was liquidated
                                    sz = float(det.get("sz") or 0.0)
                                    bk_px = float(det.get("bkPx") or 0.0)
                                    usd_val = sz * ct_val * bk_px
                                    if usd_val > 0:
                                        self.record_liquidation("okx", inst_id, side, usd_val, bk_px)
                        except Exception as e:
                            logger.error(f"Error parsing OKX liquidation: {e}")
            except Exception as e:
                logger.warning(f"OKX liquidation stream disconnected ({e}). Reconnecting in 5s...")
                await asyncio.sleep(5)

    # =========================================================================
    # Aggregation & State Broadcasting Loop
    # =========================================================================
    async def aggregation_loop(self):
        while self.running:
            try:
                self._prune_and_aggregate()
                await self._publish_to_redis()
            except Exception as e:
                logger.error(f"Error in liquidation aggregation loop: {e}")
            await asyncio.sleep(1.0)

    async def start(self):
        self.running = True
        await redis_manager.connect()
        logger.info("Starting Multi-Exchange Liquidation Cascade Hunter...")
        
        tasks = [
            asyncio.create_task(self.listen_binance()),
            asyncio.create_task(self.listen_bybit()),
            asyncio.create_task(self.listen_okx()),
            asyncio.create_task(self.aggregation_loop())
        ]
        await asyncio.gather(*tasks)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    set_log_file("logs/liquidation_hunter.log")
    collector = MultiExchangeLiquidationCollector()
    try:
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        collector.stop()
