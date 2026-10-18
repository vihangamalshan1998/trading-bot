import asyncio
import sys
import datetime

sys.path.insert(0, ".")
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    
    positions = await client.get_positions()
    print("=== LIVE POSITIONS ON BINANCE ===")
    for p in positions:
        amt = float(p.get('positionAmt', 0))
        if abs(amt) > 0:
            print(f"SYMBOL: {p.get('symbol')} | Amt: {amt} | Entry: {p.get('entryPrice')} | UnPnl: {p.get('unRealizedProfit')}")
            
    orders = await client._request('GET', '/fapi/v1/allOrders', signed=True, params={'symbol': 'XRPUSDT', 'limit': 10})
    print("\n=== RECENT XRP ORDERS ===")
    for o in orders:
        t_ms = int(o.get('time', 0))
        dt = datetime.datetime.fromtimestamp(t_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{dt}] OID: {o.get('orderId')} | ClientOID: {o.get('clientOrderId')} | Side: {o.get('side')} | Type: {o.get('type')} | Status: {o.get('status')} | OrigQty: {o.get('origQty')} | ExecQty: {o.get('executedQty')} | Price: {o.get('price')} | Avg: {o.get('avgPrice')}")

    trades = await client._request('GET', '/fapi/v1/userTrades', signed=True, params={'symbol': 'XRPUSDT', 'limit': 10})
    print("\n=== RECENT XRP USER TRADES ===")
    for tr in trades:
        t_ms = int(tr.get('time', 0))
        dt = datetime.datetime.fromtimestamp(t_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{dt}] TradeID: {tr.get('id')} | OID: {tr.get('orderId')} | Side: {tr.get('side')} | Qty: {tr.get('qty')} | Price: {tr.get('price')} | Fee: {tr.get('commission')} | RPNL: {tr.get('realizedPnl')}")

if __name__ == "__main__":
    asyncio.run(main())
