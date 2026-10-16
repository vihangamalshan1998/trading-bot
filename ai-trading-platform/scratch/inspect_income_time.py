import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def inspect():
    client = BinanceFuturesAdapter()
    try:
        data = await client._request("GET", "/fapi/v1/income", signed=True, params={"limit": 10})
        print("Raw time values:")
        for d in data:
            t = datetime.datetime.fromtimestamp(d["time"]/1000.0, tz=datetime.timezone.utc)
            print(f"time_ms: {d['time']} -> {t} | type: {d['incomeType']} | sym: {d.get('symbol')}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(inspect())
