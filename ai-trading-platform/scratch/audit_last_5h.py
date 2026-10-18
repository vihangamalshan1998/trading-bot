import asyncio
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.exchange.binance_client import BinanceFuturesAdapter

async def run():
    client = BinanceFuturesAdapter()
    srv_time_res = await client._request("GET", "/fapi/v1/time")
    srv_ms = srv_time_res.get("serverTime")
    # Last 5 hours
    start_ms = srv_ms - (5 * 3600 * 1000)
    
    incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={"startTime": start_ms, "limit": 1000})
    print(f"Total income records in last 5 hours: {len(incomes)}")
    
    events = []
    tot_pnl = 0.0
    tot_comm = 0.0
    tot_funding = 0.0
    
    for inc in incomes:
        itype = inc.get("incomeType")
        amt = float(inc.get("income", 0))
        sym = inc.get("symbol", "GLOBAL")
        t_ms = inc.get("time", 0)
        t_str = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        if itype == "REALIZED_PNL":
            tot_pnl += amt
            events.append((t_str, sym, "REALIZED_PNL", amt))
        elif itype == "COMMISSION":
            tot_comm += amt
            events.append((t_str, sym, "COMMISSION", amt))
        elif itype == "FUNDING_FEE":
            tot_funding += amt
            events.append((t_str, sym, "FUNDING_FEE", amt))
            
    print("\n--- ALL RECENT INCOME EVENTS (LAST 5 HOURS) ---")
    for t_str, sym, itype, amt in events:
        if itype == "REALIZED_PNL":
            print(f"[{t_str}] {sym} -> {itype}: ${amt:+.4f} USDT")
        elif itype == "COMMISSION" and abs(amt) > 0.05:
            print(f"[{t_str}] {sym} -> {itype}: ${amt:.4f} USDT")
            
    print("--------------------------------------------------")
    print(f"Total Gross Realized PnL: ${tot_pnl:+.2f} USDT")
    print(f"Total Commissions Paid:   ${tot_comm:+.2f} USDT")
    print(f"Total Funding Fees:       ${tot_funding:+.2f} USDT")
    net = tot_pnl + tot_comm + tot_funding
    print(f"NET INCOME (LAST 5H):     ${net:+.2f} USDT")
    print("==================================================")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(run())
