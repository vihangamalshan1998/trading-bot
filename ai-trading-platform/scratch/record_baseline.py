import asyncio
import os
import sys
import json
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.exchange.binance_client import BinanceFuturesAdapter

async def record():
    client = BinanceFuturesAdapter()
    await client.connect()
    
    acc = await client.get_account_details()
    wallet_balance = float(acc.get("balance", 0.0))
    avail_balance = float(acc.get("availableBalance", 0.0))
    unpnl = float(acc.get("crossUnPnl", 0.0))
    
    open_positions = await client.get_positions()
    
    # Get last 20 income events
    incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={"limit": 20})
    last_income = incomes[-1] if incomes else {}
    
    print("=== 72-HOUR BENCHMARK BASELINE SNAPSHOT ===")
    print(f"BASELINE_TIMESTAMP_UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"BASELINE_TIMESTAMP_EPOCH_MS: {int(time.time() * 1000)}")
    print(f"WALLET_BALANCE: ${wallet_balance:.4f} USDT")
    print(f"AVAILABLE_BALANCE: ${avail_balance:.4f} USDT")
    print(f"UNREALIZED_PNL: ${unpnl:.4f} USDT")
    print(f"OPEN_POSITIONS_COUNT: {len(open_positions)}")
    print(f"LAST_INCOME_ID: {last_income.get('tranId')}")
    print(f"LAST_INCOME_TIME_UTC: {datetime.fromtimestamp(last_income.get('time', 0)/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if last_income else 'None'}")
    print(f"LAST_INCOME_TYPE: {last_income.get('incomeType')}")
    print(f"LAST_INCOME_AMOUNT: {last_income.get('income')} {last_income.get('asset')}")
    print("===========================================")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(record())
