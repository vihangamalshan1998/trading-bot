import asyncio
import os
import sys
import datetime
from datetime import timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.exchange.binance_client import BinanceFuturesAdapter

# EXACT BASELINE FIXED VALUES (Locked at 2026-10-18 03:36:52 UTC / 09:06:52 IST)
STARTING_EPOCH_MS = 1792294612528
STARTING_WALLET_BALANCE = 4440.1045

async def run_72h_audit():
    client = BinanceFuturesAdapter()
    await client.connect()
    
    acc = await client.get_account_details()
    current_wallet = float(acc.get("balance", 0.0))
    current_unpnl = float(acc.get("crossUnPnl", 0.0))
    current_equity = current_wallet + current_unpnl
    
    # Fetch all income records since benchmark start
    incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={
        "startTime": STARTING_EPOCH_MS,
        "limit": 1000
    })
    
    open_pos = await client.get_positions()
    
    total_realized_pnl = 0.0
    total_commissions = 0.0
    total_funding = 0.0
    trades_count = 0
    wins = 0
    losses = 0
    
    for inc in incomes:
        itype = inc.get("incomeType")
        amount = float(inc.get("income", 0.0))
        
        if itype == "REALIZED_PNL":
            total_realized_pnl += amount
            if amount > 0:
                wins += 1
            elif amount < 0:
                losses += 1
        elif itype == "COMMISSION":
            total_commissions += amount
            trades_count += 1
        elif itype == "FUNDING_FEE":
            total_funding += amount
            
    net_pnl = current_equity - STARTING_WALLET_BALANCE
    pct_return = (net_pnl / STARTING_WALLET_BALANCE) * 100.0
    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0
    
    print("\n=======================================================")
    print("      72-HOUR EXPERIMENT FORENSIC AUDIT REPORT         ")
    print("=======================================================")
    print(f"Benchmark Start Time: 2026-10-18 03:36:52 UTC (09:06:52 IST)")
    print(f"Current Audit Time:   {datetime.datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Starting Wallet:      ${STARTING_WALLET_BALANCE:.4f} USDT")
    print(f"Current Wallet:       ${current_wallet:.4f} USDT")
    print(f"Unrealized PnL:       ${current_unpnl:+.4f} USDT")
    print(f"Current Net Equity:   ${current_equity:.4f} USDT")
    print("-------------------------------------------------------")
    print(f"Gross Realized PnL:   ${total_realized_pnl:+.4f} USDT")
    print(f"Commissions (Fees):   ${total_commissions:+.4f} USDT")
    print(f"Funding Fees:         ${total_funding:+.4f} USDT")
    print(f"Total Income Events:  {len(incomes)}")
    print(f"Win/Loss Record:      {wins} Wins / {losses} Losses (Win Rate: {win_rate:.1f}%)")
    print("-------------------------------------------------------")
    print(f"NET PROFIT/LOSS:      ${net_pnl:+.4f} USDT ({pct_return:+.2f}%)")
    print(f"Open Positions:       {len(open_pos)}")
    for p in open_pos:
        print(f"  * {p.get('symbol')}: amt={p.get('positionAmt')} ep={p.get('entryPrice')}")
    print("=======================================================\n")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(run_72h_audit())
