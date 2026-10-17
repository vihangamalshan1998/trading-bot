import asyncio
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.exchange.binance_client import BinanceFuturesAdapter

async def run():
    client = BinanceFuturesAdapter()
    today_start = datetime.datetime(2026, 10, 17, 0, 0, 0, tzinfo=datetime.timezone.utc)
    start_ms = int(today_start.timestamp() * 1000)
    
    incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={"startTime": start_ms, "limit": 1000})
    realized_pnl = 0.0
    commission = 0.0
    funding = 0.0
    trades_by_sym = {}
    trade_list = []
    
    for inc in incomes:
        itype = inc.get("incomeType")
        amt = float(inc.get("income", 0))
        sym = inc.get("symbol", "GLOBAL")
        t_ms = inc.get("time", 0)
        t_str = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc).strftime("%H:%M:%S")
        
        if itype == "REALIZED_PNL":
            realized_pnl += amt
            trades_by_sym[sym] = trades_by_sym.get(sym, 0.0) + amt
            trade_list.append((t_str, sym, amt))
        elif itype == "COMMISSION":
            commission += amt
        elif itype == "FUNDING_FEE":
            funding += amt
            
    print("========================================")
    print("TODAY'S TRADING AUDIT (2026-10-17 UTC)")
    print("========================================")
    print(f"Total Closed Trades: {len(trade_list)}")
    for t_str, sym, amt in trade_list:
        print(f"  [{t_str} UTC] {sym}: ${amt:+.2f} USDT")
        
    print("----------------------------------------")
    print(f"Gross Realized PnL:  ${realized_pnl:+.2f} USDT")
    print(f"Commissions (Fees):  ${commission:+.2f} USDT")
    print(f"Funding Payments:    ${funding:+.2f} USDT")
    net_pnl = realized_pnl + commission + funding
    print("----------------------------------------")
    print(f"NET TOTAL PROFIT/LOSS TODAY: ${net_pnl:+.2f} USDT")
    print("========================================")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(run())
