import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.getcwd())

from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    adapter = BinanceFuturesAdapter()
    await adapter.connect()
    
    # Check income in the last 30 minutes
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (30 * 60 * 1000)
    
    records = await adapter._request("GET", "/fapi/v1/income", signed=True, params={"startTime": start_ms, "limit": 50})
    if records:
        print("\n=== RECENT CLOSED TRADES & FEES (LAST 30 MIN) ===")
        for r in records:
            dt = datetime.fromtimestamp(r["time"] / 1000).strftime('%H:%M:%S')
            itype = r["incomeType"]
            amt = float(r["income"])
            sym = r.get("symbol", "GLOBAL")
            if itype in ["REALIZED_PNL", "COMMISSION"]:
                print(f" {dt} | {sym:8s} | {itype:14s} | {amt:+.4f} USDT")
    else:
        print("No income records in last 30 min.")

if __name__ == "__main__":
    asyncio.run(main())
