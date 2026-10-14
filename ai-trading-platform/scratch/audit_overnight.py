import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.getcwd())

from core.exchange.binance_client import BinanceFuturesAdapter

async def audit_overnight():
    adapter = BinanceFuturesAdapter()
    await adapter.connect()
    try:
        # Fetch from yesterday 20:00 to now
        # Yesterday 20:00 is approx timestamp 1791900000
        # Let's get now_ms and paginate backwards
        all_records = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (10 * 3600 * 1000) # last 10 hours
        
        # Paginate
        current_start = start_time
        while True:
            records = await adapter._request(
                "GET",
                "/fapi/v1/income",
                signed=True,
                params={"startTime": current_start, "limit": 1000}
            )
            if not records:
                break
            all_records.extend(records)
            if len(records) < 1000:
                break
            current_start = records[-1]["time"] + 1

        print(f"Total income events in last 10 hours: {len(all_records)}")
        
        pnl = 0.0
        comm = 0.0
        transfers = []
        
        for r in all_records:
            t_type = r["incomeType"]
            val = float(r["income"])
            if t_type == "TRANSFER":
                transfers.append(r)
            elif t_type == "COMMISSION":
                comm += val
            elif t_type == "REALIZED_PNL":
                pnl += val
                
        print(f"\nTransfers found: {len(transfers)}")
        for tr in transfers:
            dt = datetime.fromtimestamp(tr["time"] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  TRANSFER: {dt} | Amount: {tr['income']} USDT")
            
        print(f"\nSum of Realized PnL:  {pnl:+10.2f} USDT")
        print(f"Sum of Commissions:   {comm:+10.2f} USDT")
        print(f"Net Realized Change:  {(pnl + comm):+10.2f} USDT")

    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(audit_overnight())
