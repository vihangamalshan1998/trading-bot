import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    try:
        orders = await client._request("GET", "/fapi/v1/allOrders", signed=True, params={"symbol": "NEARUSDT", "limit": 10})
        print("=== RECENT NEARUSDT ALL ORDERS ===")
        for o in orders:
            ts = datetime.datetime.fromtimestamp(o.get('time', 0)/1000.0, tz=datetime.timezone.utc)
            update_ts = datetime.datetime.fromtimestamp(o.get('updateTime', 0)/1000.0, tz=datetime.timezone.utc)
            print(f"ID: {o.get('orderId')} | Side: {o.get('side')} | Type: {o.get('type')} | Status: {o.get('status')} | Price: {o.get('price')} | AvgPrice: {o.get('avgPrice')} | OrigQty: {o.get('origQty')} | ExecQty: {o.get('executedQty')} | Time: {ts} | Updated: {update_ts}")
    except Exception as e:
        print(f"Error querying Binance: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
