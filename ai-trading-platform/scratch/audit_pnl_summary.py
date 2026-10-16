import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def detailed_pnl_report():
    client = BinanceFuturesAdapter()
    try:
        # Check income history (realized PnL, funding, commission) for the past 24 hours
        start_time = int((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=6)).timestamp() * 1000)
        income_records = await client._request(
            "GET", 
            "/fapi/v1/income", 
            signed=True, 
            params={"startTime": start_time, "limit": 100}
        )
        
        print("=== BINANCE REALIZED INCOME / PNL (PAST 6 HOURS) ===")
        total_realized_pnl = 0.0
        total_commission = 0.0
        total_funding = 0.0
        
        trade_pnls = {}
        
        for inc in income_records:
            itype = inc.get("incomeType")
            income = float(inc.get("income", 0))
            sym = inc.get("symbol", "GLOBAL")
            t = datetime.datetime.fromtimestamp(inc.get("time", 0)/1000.0, tz=datetime.timezone.utc)
            
            if sym not in trade_pnls:
                trade_pnls[sym] = {"REALIZED_PNL": 0.0, "COMMISSION": 0.0, "FUNDING_FEE": 0.0}
                
            if itype == "REALIZED_PNL":
                total_realized_pnl += income
                trade_pnls[sym]["REALIZED_PNL"] += income
                print(f"[{t.strftime('%H:%M:%S')} UTC] {sym} | REALIZED PNL: ${income:+.2f}")
            elif itype == "COMMISSION":
                total_commission += income
                trade_pnls[sym]["COMMISSION"] += income
            elif itype == "FUNDING_FEE":
                total_funding += income
                trade_pnls[sym]["FUNDING_FEE"] += income

        print("\n=== PER-SYMBOL BREAKDOWN ===")
        for sym, data in trade_pnls.items():
            net = data["REALIZED_PNL"] + data["COMMISSION"] + data["FUNDING_FEE"]
            print(f"{sym}: Realized PnL = ${data['REALIZED_PNL']:+.2f} | Fees = ${data['COMMISSION']:+.2f} | Net = ${net:+.2f}")
            
        print(f"\nTOTAL REALIZED PNL: ${total_realized_pnl:+.2f}")
        print(f"TOTAL FEES/COMMISSION: ${total_commission:+.2f}")
        print(f"NET PROFIT OVER PERIOD: ${total_realized_pnl + total_commission + total_funding:+.2f}")
        
        acc = await client.get_account_details()
        print(f"\nCURRENT WALLET BALANCE: ${float(acc.get('balance', 0)):.2f} USDT")
        
    except Exception as e:
        print(f"Error generating PnL report: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(detailed_pnl_report())
