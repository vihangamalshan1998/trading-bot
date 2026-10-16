import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    c = BinanceFuturesAdapter()
    await c.connect()
    for sym in ['NEARUSDT']:
        res = await c._request('GET', '/fapi/v1/allOrders', params={'symbol': sym, 'limit': 10}, signed=True)
        print(f"=== {sym} RECENT ORDERS ===")
        for o in res:
            t = datetime.datetime.fromtimestamp(o['time']/1000).strftime('%H:%M:%S UTC')
            print(f"{t} | Side: {o['side']} | Type: {o['type']} | Status: {o['status']} | ClientID: {o['clientOrderId']} | Qty: {o['origQty']} | ExecQty: {o['executedQty']} | Price: {o['price']} | AvgPrice: {o.get('avgPrice')}")
    await c.close()

if __name__ == '__main__':
    asyncio.run(main())
