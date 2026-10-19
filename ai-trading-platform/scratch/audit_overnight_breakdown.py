import asyncio
import os
import sys
import datetime
from collections import defaultdict

sys.path.insert(0, ".")
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    
    # Starting epoch: 2026-10-18 14:00:00 UTC
    start_ms = 1792332000000 
    
    all_incomes = []
    current_start = start_ms
    while True:
        incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={
            "startTime": current_start,
            "limit": 1000
        })
        if not incomes:
            break
        all_incomes.extend(incomes)
        if len(incomes) < 1000:
            break
        # update current_start to last income time + 1
        current_start = int(incomes[-1]['time']) + 1
        await asyncio.sleep(0.2)
        
    print(f"Total income records fetched: {len(all_incomes)}")
    
    symbol_pnl = defaultdict(float)
    symbol_fee = defaultdict(float)
    symbol_trades = defaultdict(int)
    type_counts = defaultdict(int)
    
    for inc in all_incomes:
        sym = inc.get("symbol", "UNKNOWN")
        itype = inc.get("incomeType")
        amt = float(inc.get("income", 0.0))
        type_counts[itype] += 1
        
        if itype == "REALIZED_PNL":
            symbol_pnl[sym] += amt
            symbol_trades[sym] += 1
        elif itype == "COMMISSION":
            symbol_fee[sym] += amt
            
    print("\n=== BREAKDOWN BY SYMBOL (Since 14:00 UTC Yesterday) ===")
    print(f"{'Symbol':<15} | {'Realized PnL':<15} | {'Fees':<12} | {'Net PnL':<12} | {'Trades':<8}")
    print("-" * 70)
    
    total_net = 0.0
    for sym in sorted(symbol_pnl.keys() | symbol_fee.keys()):
        rpnl = symbol_pnl[sym]
        fee = symbol_fee[sym]
        net = rpnl + fee
        total_net += net
        cnt = symbol_trades[sym]
        print(f"{sym:<15} | ${rpnl:<14.4f} | ${fee:<11.4f} | ${net:<11.4f} | {cnt:<8}")
        
    print("-" * 70)
    print(f"TOTAL NET OVERNIGHT: ${total_net:.4f}")
    print(f"Income type counts: {dict(type_counts)}")

if __name__ == '__main__':
    asyncio.run(main())
