import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def run_full_daily_audit():
    client = BinanceFuturesAdapter()
    try:
        today_start = datetime.datetime(2026, 10, 16, 0, 0, 0, tzinfo=datetime.timezone.utc)
        today_start_ms = int(today_start.timestamp() * 1000)
        
        all_records = []
        start_ms = today_start_ms
        
        while True:
            params = {"startTime": start_ms, "limit": 1000}
            records = await client._request("GET", "/fapi/v1/income", signed=True, params=params)
            if not records:
                break
            
            all_records.extend(records)
            if len(records) < 1000:
                break
                
            start_ms = records[-1]["time"] + 1

        print(f"Total income records fetched for today (Oct 16 UTC): {len(all_records)}")
        
        main_realized = 0.0
        main_commission = 0.0
        main_funding = 0.0
        
        sv3_realized = 0.0
        sv3_commission = 0.0
        sv3_funding = 0.0
        
        main_sym_stats = {}
        sv3_sym_stats = {}
        
        for r in all_records:
            itype = r["incomeType"]
            amt = float(r["income"])
            sym = r.get("symbol", "GLOBAL")
            t_ms = r["time"]
            t = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc)
            
            # Is this stable-version-3?
            # 1. DOGE was 100% stable-version-3
            # 2. Counter-trend shorts between 10:40 UTC and 11:25 UTC when stable-version-3 was tested
            is_sv3 = False
            if sym == "DOGEUSDT":
                is_sv3 = True
            elif sym in ["ETHUSDT"] and (datetime.datetime(2026, 10, 16, 10, 40, 0, tzinfo=datetime.timezone.utc) <= t <= datetime.datetime(2026, 10, 16, 11, 25, 0, tzinfo=datetime.timezone.utc)):
                is_sv3 = True
                
            if is_sv3:
                target_dict = sv3_sym_stats
                if sym not in target_dict:
                    target_dict[sym] = {"realized": 0.0, "commission": 0.0, "funding": 0.0, "trades": 0}
                if itype == "REALIZED_PNL":
                    sv3_realized += amt
                    target_dict[sym]["realized"] += amt
                    target_dict[sym]["trades"] += 1
                elif itype == "COMMISSION":
                    sv3_commission += amt
                    target_dict[sym]["commission"] += amt
                elif itype == "FUNDING_FEE":
                    sv3_funding += amt
                    target_dict[sym]["funding"] += amt
            else:
                target_dict = main_sym_stats
                if sym not in target_dict:
                    target_dict[sym] = {"realized": 0.0, "commission": 0.0, "funding": 0.0, "trades": 0, "pnl_list": []}
                if itype == "REALIZED_PNL":
                    main_realized += amt
                    target_dict[sym]["realized"] += amt
                    target_dict[sym]["trades"] += 1
                    target_dict[sym]["pnl_list"].append((t.strftime("%H:%M:%S UTC"), amt))
                elif itype == "COMMISSION":
                    main_commission += amt
                    target_dict[sym]["commission"] += amt
                elif itype == "FUNDING_FEE":
                    main_funding += amt
                    target_dict[sym]["funding"] += amt

        print("\n" + "="*60)
        print("MAIN PRODUCTION BOT - SYMBOL BREAKDOWN (OCTOBER 16)")
        print("="*60)
        for sym, s in sorted(main_sym_stats.items()):
            net_sym = s["realized"] + s["commission"] + s["funding"]
            print(f"[{sym}]")
            print(f"  Realized PnL:   ${s['realized']:+.2f} ({s['trades']} closes)")
            print(f"  Commission:     ${s['commission']:+.2f}")
            print(f"  Funding:        ${s['funding']:+.2f}")
            print(f"  Net Total:      ${net_sym:+.2f}")
            if s["pnl_list"]:
                print(f"  Realized Trades (First 3 & Last 3):")
                sample = s["pnl_list"][:3] + s["pnl_list"][-3:] if len(s["pnl_list"]) > 6 else s["pnl_list"]
                for time_str, pnl_val in sample:
                    print(f"    - {time_str}: ${pnl_val:+.2f}")
                
        net_main = main_realized + main_commission + main_funding
        print("\n" + "-"*60)
        print("MAIN PRODUCTION BOT TOTALS (TODAY):")
        print(f"  Gross Realized PnL:  ${main_realized:+.2f}")
        print(f"  Total Commissions:   ${main_commission:+.2f}")
        print(f"  Total Funding Fees:  ${main_funding:+.2f}")
        print(f"  NET MAIN BOT PROFIT: ${net_main:+.2f}")
        print("-" * 60)
        
        print("\n" + "="*60)
        print("EXCLUDED: STABLE-VERSION-3 EXPERIMENT BREAKDOWN")
        print("="*60)
        for sym, s in sorted(sv3_sym_stats.items()):
            net_sv3_sym = s["realized"] + s["commission"] + s["funding"]
            print(f"[{sym}] Realized: ${s['realized']:+.2f} | Fees: ${s['commission']:+.2f} | Net: ${net_sv3_sym:+.2f}")
        net_sv3 = sv3_realized + sv3_commission + sv3_funding
        print(f"  Total SV3 Realized:    ${sv3_realized:+.2f}")
        print(f"  Total SV3 Commission:  ${sv3_commission:+.2f}")
        print(f"  TOTAL SV3 NET IMPACT:  ${net_sv3:+.2f} (EXCLUDED)")
        print("="*60)
        
        acc = await client.get_account_details()
        print(f"\nCURRENT LIVE WALLET BALANCE: ${float(acc.get('balance', 0)):.2f} USDT")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(run_full_daily_audit())
