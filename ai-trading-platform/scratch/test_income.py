import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    try:
        # Fetch income without startTime first
        data = await client._request("GET", "/fapi/v1/income", signed=True, params={"limit": 50})
        print(f"Total income records returned: {len(data)}")
        for r in data[:10]:
            t = datetime.datetime.fromtimestamp(r["time"]/1000.0, tz=datetime.timezone.utc)
            print(f"[{t}] {r.get('symbol')} | {r.get('incomeType')} | {r.get('income')} | info: {r.get('info')}")
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
