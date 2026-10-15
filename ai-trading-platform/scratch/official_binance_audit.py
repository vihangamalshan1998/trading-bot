import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def audit_official_binance_periods():
    client = BinanceFuturesAdapter()
    
    # Period 1: 14 Oct 10:30 UTC to 23:30 UTC (4pm 14/10 to 5am 15/10 local)
    p1_start = int(datetime.datetime(2026, 10, 14, 10, 30, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    p1_end   = int(datetime.datetime(2026, 10, 14, 23, 30, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    
    # Period 2: 14 Oct 23:30 UTC to 15 Oct 03:55 UTC (5am 15/10 to ~9:25am local)
    p2_start = p1_end
    p2_end   = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    
    async def get_all_income(start_ms, end_ms):
        all_items = []
        cur_start = start_ms
        while True:
            batch = await client._request(
                "GET", "/fapi/v1/income", signed=True,
                params={"startTime": cur_start, "endTime": end_ms, "limit": 1000}
            )
            if not batch:
                break
            all_items.extend(batch)
            if len(batch) < 1000:
                break
            cur_start = int(batch[-1]["time"]) + 1
            if cur_start >= end_ms:
                break
        return all_items

    p1_income = await get_all_income(p1_start, p1_end)
    p2_income = await get_all_income(p2_start, p2_end)
    
    def process_income(income_list, name):
        pnl_items = [x for x in income_list if x.get("incomeType") == "REALIZED_PNL"]
        comm_items = [x for x in income_list if x.get("incomeType") == "COMMISSION"]
        funding_items = [x for x in income_list if x.get("incomeType") == "FUNDING_FEE"]
        
        # Realized PnL wins / losses (group by trade or info / orderId if possible, or individual PnL records)
        # Note: Binance splits multiple fills into multiple PnL records. Group by (symbol, info, time) within 1 sec
        grouped_trades = {}
        for x in pnl_items:
            key = f"{x.get('symbol')}_{x.get('time')//2000}"
            if key not in grouped_trades:
                grouped_trades[key] = {"symbol": x.get("symbol"), "pnl": 0.0, "time": x.get("time")}
            grouped_trades[key]["pnl"] += float(x.get("income", 0))
            
        trades = list(grouped_trades.values())
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        be = [t for t in trades if t["pnl"] == 0]
        
        gross_pnl = sum(float(x.get("income", 0)) for x in pnl_items)
        comm = sum(float(x.get("income", 0)) for x in comm_items)
        funding = sum(float(x.get("income", 0)) for x in funding_items)
        net_pnl = gross_pnl + comm + funding # comm is negative
        
        wr = (len(wins) / len(trades) * 100) if trades else 0.0
        
        print(f"\n=======================================================")
        print(f"OFFICIAL BINANCE LEDGER AUDIT: {name}")
        print(f"=======================================================")
        print(f"Total Closed Trade Events: {len(trades)}")
        print(f"Win Rate:                  {wr:.1f}% ({len(wins)} Wins / {len(losses)} Losses / {len(be)} BE)")
        print(f"Gross Realized PnL:        ${gross_pnl:+.4f} USDT")
        print(f"Total Commissions Paid:    ${abs(comm):.4f} USDT")
        print(f"Total Funding Paid:        ${funding:+.4f} USDT")
        print(f"Net Bottom-Line PnL:       ${net_pnl:+.4f} USDT")
        
        sym_map = {}
        for t in trades:
            s = t["symbol"]
            if s not in sym_map: sym_map[s] = {"w": 0, "l": 0, "pnl": 0.0, "n": 0}
            sym_map[s]["n"] += 1
            sym_map[s]["pnl"] += t["pnl"]
            if t["pnl"] > 0: sym_map[s]["w"] += 1
            elif t["pnl"] < 0: sym_map[s]["l"] += 1
            
        print("\nSymbol Breakdown (Binance Realized PnL):")
        for s, d in sym_map.items():
            swr = (d["w"] / d["n"] * 100) if d["n"] else 0.0
            print(f"  {s:<10} | Trades: {d['n']:<2} | Win Rate: {swr:5.1f}% | Net PnL: ${d['pnl']:+8.4f}")

    process_income(p1_income, "Period 1 (14 Oct 4:00 PM - 15 Oct 5:00 AM Local)")
    process_income(p2_income, "Period 2 (15 Oct 5:00 AM - NOW)")

    await client.close()

if __name__ == "__main__":
    asyncio.run(audit_official_binance_periods())
