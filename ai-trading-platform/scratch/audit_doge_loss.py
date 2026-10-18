import asyncio
import sys
import os
import datetime

sys.path.insert(0, ".")
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    
    # 1. Fetch recent orders for DOGEUSDT
    orders = await client._request('GET', '/fapi/v1/allOrders', signed=True, params={'symbol': 'DOGEUSDT', 'limit': 15})
    print("=== RECENT DOGEUSDT ORDERS ===")
    for o in orders:
        t_ms = int(o.get('time', 0))
        dt = datetime.datetime.fromtimestamp(t_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{dt}] OID: {o.get('orderId')} | ClientOID: {o.get('clientOrderId')} | Side: {o.get('side')} | Type: {o.get('type')} | Status: {o.get('status')} | OrigQty: {o.get('origQty')} | ExecQty: {o.get('executedQty')} | Price: {o.get('price')} | Avg: {o.get('avgPrice')}")

    # 2. Fetch user trades for DOGEUSDT
    trades = await client._request('GET', '/fapi/v1/userTrades', signed=True, params={'symbol': 'DOGEUSDT', 'limit': 20})
    print("\n=== RECENT DOGEUSDT USER TRADES ===")
    for tr in trades:
        t_ms = int(tr.get('time', 0))
        dt = datetime.datetime.fromtimestamp(t_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{dt}] TradeID: {tr.get('id')} | OID: {tr.get('orderId')} | Side: {tr.get('side')} | Qty: {tr.get('qty')} | Price: {tr.get('price')} | Fee: {tr.get('commission')} | RPNL: {tr.get('realizedPnl')} | Maker: {tr.get('maker')}")

if __name__ == "__main__":
    asyncio.run(main())
