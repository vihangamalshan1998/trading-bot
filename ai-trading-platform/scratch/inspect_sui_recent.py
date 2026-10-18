import asyncio
import sys
import datetime

sys.path.insert(0, ".")
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    
    print("=== SUIUSDT ALL RECENT ORDERS ===")
    orders = await client._request('GET', '/fapi/v1/allOrders', signed=True, params={'symbol': 'SUIUSDT', 'limit': 10})
    for o in orders:
        t_ms = int(o.get('time', 0))
        dt = datetime.datetime.fromtimestamp(t_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{dt}] OID: {o.get('orderId')} | ClientOID: {o.get('clientOrderId')} | Side: {o.get('side')} | Type: {o.get('type')} | Status: {o.get('status')} | OrigQty: {o.get('origQty')} | ExecQty: {o.get('executedQty')} | Price: {o.get('price')} | AvgPrice: {o.get('avgPrice')}")

    print("\n=== SUIUSDT ALL RECENT TRADES ===")
    trades = await client._request('GET', '/fapi/v1/userTrades', signed=True, params={'symbol': 'SUIUSDT', 'limit': 10})
    for tr in trades:
        t_ms = int(tr.get('time', 0))
        dt = datetime.datetime.fromtimestamp(t_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{dt}] TradeID: {tr.get('id')} | OID: {tr.get('orderId')} | Side: {tr.get('side')} | Qty: {tr.get('qty')} | Price: {tr.get('price')} | Fee: {tr.get('commission')} | RPNL: {tr.get('realizedPnl')} | Maker: {tr.get('maker')}")

    print("\n=== SUIUSDT INCOMES ===")
    incomes = await client._request('GET', '/fapi/v1/income', signed=True, params={'symbol': 'SUIUSDT', 'limit': 10})
    for inc in incomes:
        t_ms = int(inc.get('time', 0))
        dt = datetime.datetime.fromtimestamp(t_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{dt}] {inc.get('incomeType')}: {inc.get('income')} USDT | TradeID: {inc.get('tradeId')}")

if __name__ == "__main__":
    asyncio.run(main())
