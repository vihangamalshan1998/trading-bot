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
    srv_dt = datetime.datetime.fromtimestamp(srv_ms / 1000.0, tz=datetime.timezone.utc)
    start_24h_ms = srv_ms - (24 * 3600 * 1000)
    start_24h_dt = datetime.datetime.fromtimestamp(start_24h_ms / 1000.0, tz=datetime.timezone.utc)
    
    incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={"startTime": start_24h_ms, "limit": 1000})
    realized_trades = []
    fees_24h = 0.0
    funding_24h = 0.0
    wins = 0
    losses = 0
    
    by_symbol = {}
    
    for inc in incomes:
        itype = inc.get("incomeType")
        amt = float(inc.get("income", 0))
        sym = inc.get("symbol", "GLOBAL")
        t_ms = inc.get("time", 0)
        t_str = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc).strftime("%H:%M UTC")
        
        if itype == "REALIZED_PNL":
            realized_trades.append((t_str, sym, amt))
            if sym not in by_symbol:
                by_symbol[sym] = {"gross": 0.0, "wins": 0, "losses": 0}
            by_symbol[sym]["gross"] += amt
            if amt > 0:
                wins += 1
                by_symbol[sym]["wins"] += 1
            elif amt < 0:
                losses += 1
                by_symbol[sym]["losses"] += 1
        elif itype == "COMMISSION":
            fees_24h += amt
        elif itype == "FUNDING_FEE":
            funding_24h += amt
            
    gross_pnl = sum(x[2] for x in realized_trades)
    net_pnl = gross_pnl + fees_24h + funding_24h
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
    
    print("==================================================")
    print("        BINANCE 24-HOUR OFFICIAL AUDIT            ")
    print("==================================================")
    print(f"Window: {start_24h_dt.strftime('%Y-%m-%d %H:%M')} to {srv_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Total Trade Events: {len(realized_trades)}")
    print(f"Win/Loss Record:    {wins} Wins / {losses} Losses (Win Rate: {win_rate:.1f}%)")
    print(f"Gross Realized PnL: ${gross_pnl:+.2f} USDT")
    print(f"Exchange Fees Paid: ${fees_24h:+.2f} USDT")
    print(f"Funding Fees Net:   ${funding_24h:+.2f} USDT")
    print("--------------------------------------------------")
    print(f"NET 24-HOUR RESULT: ${net_pnl:+.2f} USDT")
    print("==================================================")
    print("\nBreakdown by Symbol (Gross PnL):")
    for s, data in by_symbol.items():
        print(f"  {s}: ${data['gross']:+.2f} USDT ({data['wins']}W / {data['losses']}L)")
        
    await client.close()

if __name__ == "__main__":
    asyncio.run(run())
