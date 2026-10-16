import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def full_daily_pnl_audit():
    client = BinanceFuturesAdapter()
    try:
        # Start of today (2026-10-16 00:00:00 UTC)
        today_start_utc = datetime.datetime(2026, 10, 16, 0, 0, 0, tzinfo=datetime.timezone.utc)
        start_ms = int(today_start_utc.timestamp() * 1000)
        
        income_records = []
        # Paginate to fetch all records today
        from_id = None
        while True:
            params = {"startTime": start_ms, "limit": 1000}
            data = await client._request("GET", "/fapi/v1/income", signed=True, params=params)
            if not data:
                break
            income_records.extend(data)
            if len(data) < 1000:
                break
            start_ms = data[-1]["time"] + 1
            
        print(f"Total income records retrieved for today (2026-10-16 UTC): {len(income_records)}")
        
        # Breakdown
        # We need to distinguish:
        # 1. stable-version-3 experiment trades:
        #    - DOGE trades (DOGE was ONLY traded on stable-version-3!)
        #    - Specific counter-trend ETH & XRP shorts opened between 10:45 UTC and 11:25 UTC when stable-version-3 ran
        
        main_realized_pnl = 0.0
        main_fees = 0.0
        main_funding = 0.0
        
        sv3_realized_pnl = 0.0
        sv3_fees = 0.0
        sv3_funding = 0.0
        
        main_trades_by_sym = {}
        sv3_trades_by_sym = {}
        
        for inc in income_records:
            itype = inc.get("incomeType")
            income = float(inc.get("income", 0))
            sym = inc.get("symbol", "GLOBAL")
            t_ms = inc.get("time", 0)
            t = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc)
            
            # Identify stable-version-3 trades:
            # DOGE was 100% on stable-version-3
            # Also trades executed between 10:45 UTC and 11:22 UTC on ETH/XRP when stable-version-3 was deployed
            is_sv3 = False
            if sym == "DOGEUSDT":
                is_sv3 = True
            elif sym in ["ETHUSDT", "XRPUSDT"] and (datetime.datetime(2026, 10, 16, 10, 45, 0, tzinfo=datetime.timezone.utc) <= t <= datetime.datetime(2026, 10, 16, 11, 25, 0, tzinfo=datetime.timezone.utc)):
                is_sv3 = True
                
            if is_sv3:
                if sym not in sv3_trades_by_sym:
                    sv3_trades_by_sym[sym] = {"realized": 0.0, "fees": 0.0, "funding": 0.0}
                if itype == "REALIZED_PNL":
                    sv3_realized_pnl += income
                    sv3_trades_by_sym[sym]["realized"] += income
                elif itype == "COMMISSION":
                    sv3_fees += income
                    sv3_trades_by_sym[sym]["fees"] += income
                elif itype == "FUNDING_FEE":
                    sv3_funding += income
                    sv3_trades_by_sym[sym]["funding"] += income
            else:
                if sym not in main_trades_by_sym:
                    main_trades_by_sym[sym] = {"realized": 0.0, "fees": 0.0, "funding": 0.0, "pnl_events": []}
                if itype == "REALIZED_PNL":
                    main_realized_pnl += income
                    main_trades_by_sym[sym]["realized"] += income
                    main_trades_by_sym[sym]["pnl_events"].append((t.strftime("%H:%M:%S UTC"), income))
                elif itype == "COMMISSION":
                    main_fees += income
                    main_trades_by_sym[sym]["fees"] += income
                elif itype == "FUNDING_FEE":
                    main_funding += income
                    main_trades_by_sym[sym]["funding"] += income

        print("\n" + "="*50)
        print("🏛️ MAIN PRODUCTION BOT TRADES (TODAY)")
        print("="*50)
        for sym, d in main_trades_by_sym.items():
            net = d["realized"] + d["fees"] + d["funding"]
            print(f"\n[{sym}]")
            print(f"  Realized PnL: ${d['realized']:+.2f}")
            print(f"  Commission Fees: ${d['fees']:+.2f}")
            print(f"  Funding Fees: ${d['funding']:+.2f}")
            print(f"  Net Total: ${net:+.2f}")
            if d["pnl_events"]:
                print("  Key Realized Events:")
                for time_str, pnl_val in d["pnl_events"][-6:]:
                    print(f"    - {time_str}: ${pnl_val:+.2f}")

        net_main = main_realized_pnl + main_fees + main_funding
        print("\n" + "-"*50)
        print(f"MAIN BOT TOTAL REALIZED PNL: ${main_realized_pnl:+.2f}")
        print(f"MAIN BOT TOTAL FEES:        ${main_fees:+.2f}")
        print(f"MAIN BOT TOTAL FUNDING:     ${main_funding:+.2f}")
        print(f"MAIN BOT NET PROFIT:        ${net_main:+.2f}")
        print("-"*50)

        print("\n" + "="*50)
        print("🧪 EXCLUDED: STABLE-VERSION-3 EXPERIMENT TRADES")
        print("="*50)
        for sym, d in sv3_trades_by_sym.items():
            net = d["realized"] + d["fees"] + d["funding"]
            print(f"{sym}: Realized = ${d['realized']:+.2f} | Fees = ${d['fees']:+.2f} | Net = ${net:+.2f}")
        net_sv3 = sv3_realized_pnl + sv3_fees + sv3_funding
        print(f"STABLE-VERSION-3 NET IMPACT (EXCLUDED): ${net_sv3:+.2f}")
        print("="*50)

        acc = await client.get_account_details()
        print(f"\nCURRENT LIVE WALLET BALANCE: ${float(acc.get('balance', 0)):.2f} USDT")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(full_daily_pnl_audit())
