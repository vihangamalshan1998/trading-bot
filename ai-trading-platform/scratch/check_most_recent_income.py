import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.getcwd())

from core.exchange.binance_client import BinanceFuturesAdapter

async def check_recent():
    adapter = BinanceFuturesAdapter()
    await adapter.connect()
    try:
        # Fetch the MOST RECENT 100 income records up to right now
        income_records = await adapter._request(
            "GET",
            "/fapi/v1/income",
            signed=True,
            params={"limit": 100}
        )
        
        print(f"Fetched {len(income_records)} most recent income records:")
        for r in income_records[-25:]:
            dt = datetime.fromtimestamp(r["time"] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            itype = r["incomeType"]
            amt = float(r["income"])
            sym = r.get("symbol", "GLOBAL")
            print(f" -> {dt} | {sym:10s} | {itype:15s} | {amt:+10.4f} USDT")

    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(check_recent())
