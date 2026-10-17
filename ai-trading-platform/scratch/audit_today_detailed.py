import asyncio
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.exchange.binance_client import BinanceFuturesAdapter

async def run():
    client = BinanceFuturesAdapter()
    # Let's get server time
    srv_time_res = await client._request("GET", "/fapi/v1/time")
    srv_time_ms = srv_time_res.get("serverTime")
    srv_dt = datetime.datetime.fromtimestamp(srv_time_ms / 1000.0, tz=datetime.timezone.utc)
    print(f"Binance Server Time (UTC): {srv_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Start of today UTC
    start_today = datetime.datetime(srv_dt.year, srv_dt.month, srv_dt.day, 0, 0, 0, tzinfo=datetime.timezone.utc)
    start_today_ms = int(start_today.timestamp() * 1000)
    print(f"Start of today UTC: {start_today.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={"startTime": start_today_ms, "limit": 1000})
    print(f"Total income records since start of today UTC: {len(incomes)}")
    
    realized_trades = []
    fees_today = 0.0
    funding_today = 0.0
    
    for inc in incomes:
        itype = inc.get("incomeType")
        amt = float(inc.get("income", 0))
        sym = inc.get("symbol", "GLOBAL")
        t_ms = inc.get("time", 0)
        t_str = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        if itype == "REALIZED_PNL":
            realized_trades.append({"time": t_str, "symbol": sym, "pnl": amt})
        elif itype == "COMMISSION":
            fees_today += amt
        elif itype == "FUNDING_FEE":
            funding_today += amt
            
    print("\n--- REALIZED PNL TRADES TODAY ---")
    tot_pnl = 0.0
    for r in realized_trades:
        print(f"  [{r['time']}] {r['symbol']}: ${r['pnl']:+.4f} USDT")
        tot_pnl += r['pnl']
        
    print("---------------------------------")
    print(f"Gross Realized PnL:   ${tot_pnl:+.4f} USDT")
    print(f"Total Commissions:    ${fees_today:+.4f} USDT")
    print(f"Total Funding Fees:   ${funding_today:+.4f} USDT")
    net = tot_pnl + fees_today + funding_today
    print(f"NET PROFIT/LOSS TODAY: ${net:+.4f} USDT")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(run())
