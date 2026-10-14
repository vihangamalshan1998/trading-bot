import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.getcwd())

from core.exchange.binance_client import BinanceFuturesAdapter

async def check_income():
    adapter = BinanceFuturesAdapter()
    await adapter.connect()
    try:
        # Query /fapi/v1/income (shows REALIZED_PNL, COMMISSION, TRANSFER)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (12 * 3600 * 1000) # Last 12 hours
        
        income_records = await adapter._request(
            "GET",
            "/fapi/v1/income",
            signed=True,
            params={"startTime": start_ms, "limit": 100}
        )
        
        if not isinstance(income_records, list):
            print(f"Error fetching income: {income_records}")
            return
            
        print(f"Total income records in last 12 hours: {len(income_records)}")
        
        commissions = 0.0
        realized_pnl = 0.0
        transfers = 0.0
        funding_fees = 0.0
        
        income_records.sort(key=lambda x: x["time"])
        
        print("\n=== RECENT FINANCIAL TRANSACTIONS (BINANCE TESTNET) ===")
        for r in income_records[-30:]: # show last 30
            dt = datetime.fromtimestamp(r["time"] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            itype = r["incomeType"]
            amt = float(r["income"])
            sym = r.get("symbol", "GLOBAL")
            
            if itype == "COMMISSION":
                commissions += amt
            elif itype == "REALIZED_PNL":
                realized_pnl += amt
            elif itype == "TRANSFER":
                transfers += amt
            elif itype == "FUNDING_FEE":
                funding_fees += amt
                
            print(f" -> {dt} | {sym:8s} | {itype:15s} | {amt:+10.4f} USDT")
            
        # Total sums
        for r in income_records:
            itype = r["incomeType"]
            amt = float(r["income"])
            if itype == "COMMISSION": commissions += amt
            elif itype == "REALIZED_PNL": realized_pnl += amt
            elif itype == "TRANSFER": transfers += amt
            elif itype == "FUNDING_FEE": funding_fees += amt
            
        print("\n================ TOTALS (LAST 12 HOURS) ================")
        print(f"Total Transfers:     {transfers:+10.2f} USDT")
        print(f"Total Realized PnL:  {realized_pnl:+10.2f} USDT")
        print(f"Total Commissions:   {commissions:+10.2f} USDT")
        print(f"Total Funding Fees:  {funding_fees:+10.2f} USDT")
        print(f"NET WALLET CHANGE:   {(realized_pnl + commissions + funding_fees):+10.2f} USDT")
        print("=========================================================")

    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(check_income())
