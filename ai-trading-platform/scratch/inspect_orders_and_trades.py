import asyncio
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.exchange.binance_client import BinanceFuturesAdapter

async def inspect_recent_orders():
    client = BinanceFuturesAdapter()
    symbols = ["XRPUSDT", "ADAUSDT", "NEARUSDT"]
    
    for sym in symbols:
        print(f"\n==================== {sym} ALL ORDERS ====================")
        orders = await client._request("GET", "/fapi/v1/allOrders", signed=True, params={"symbol": sym, "limit": 10})
        for o in orders[-10:]:
            t_ms = o.get("time", 0)
            t_str = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"[{t_str}] OrderId: {o.get('orderId')} | Side: {o.get('side')} | Type: {o.get('type')} | OrigQty: {o.get('origQty')} | ExecQty: {o.get('executedQty')} | Price: {o.get('price')} | AvgPrice: {o.get('avgPrice')} | Status: {o.get('status')} | reduceOnly: {o.get('reduceOnly')}")
            
        print(f"\n--- {sym} RECENT TRADES ---")
        trades = await client._request("GET", "/fapi/v1/userTrades", signed=True, params={"symbol": sym, "limit": 10})
        for t in trades[-10:]:
            t_ms = t.get("time", 0)
            t_str = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"[{t_str}] TradeId: {t.get('id')} | Side: {t.get('side')} | Qty: {t.get('qty')} @ ${t.get('price')} | RealizedPnL: ${float(t.get('realizedPnl', 0)):+.4f} | Comm: ${float(t.get('commission', 0)):.4f}")
            
    await client.close()

if __name__ == "__main__":
    asyncio.run(inspect_recent_orders())
