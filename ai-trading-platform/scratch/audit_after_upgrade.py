import asyncio
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.exchange.binance_client import BinanceFuturesAdapter

async def run():
    client = BinanceFuturesAdapter()
    deploy_dt = datetime.datetime(2026, 10, 17, 6, 1, 0, tzinfo=datetime.timezone.utc)
    deploy_ms = int(deploy_dt.timestamp() * 1000)
    
    incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={"startTime": deploy_ms, "limit": 1000})
    realized_trades = []
    fees_paid = 0.0
    funding_earned = 0.0
    wins = 0
    losses = 0
    
    for inc in incomes:
        itype = inc.get("incomeType")
        amt = float(inc.get("income", 0))
        sym = inc.get("symbol", "GLOBAL")
        t_ms = inc.get("time", 0)
        t_str = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc).strftime("%H:%M UTC")
        
        if itype == "REALIZED_PNL":
            realized_trades.append((t_str, sym, amt))
            if amt > 0:
                wins += 1
            elif amt < 0:
                losses += 1
        elif itype == "COMMISSION":
            fees_paid += amt
        elif itype == "FUNDING_FEE":
            funding_earned += amt
            
    gross_pnl = sum(x[2] for x in realized_trades)
    net_pnl = gross_pnl + fees_paid + funding_earned
    
    print("==================================================")
    print("  AUDIT: PERFORMANCE AFTER THE BIG UPGRADE       ")
    print("==================================================")
    print(f"Deploy Time:        2026-10-17 06:01 UTC")
    print(f"Total Trade Events: {len(realized_trades)} (Wins: {wins}, Cuts: {losses})")
    print(f"Gross Profit:       ${gross_pnl:+.2f} USDT")
    print(f"Commissions Paid:   ${fees_paid:+.2f} USDT")
    print(f"Funding Collected:  ${funding_earned:+.2f} USDT")
    print("--------------------------------------------------")
    print(f"NET PROFIT / LOSS:  ${net_pnl:+.2f} USDT")
    print("==================================================")
    
    print("\nClosed Trades Since Deployment:")
    for t_str, sym, amt in realized_trades:
        print(f"  [{t_str}] {sym}: ${amt:+.2f} USDT")
        
    await client.close()

if __name__ == "__main__":
    asyncio.run(run())
