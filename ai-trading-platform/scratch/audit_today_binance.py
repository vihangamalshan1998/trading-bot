import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def audit_today():
    client = BinanceFuturesAdapter()
    
    # 00:00 UTC today (October 17, 2026)
    now = datetime.datetime.now(datetime.timezone.utc)
    start_of_day = datetime.datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=datetime.timezone.utc)
    start_time_ms = int(start_of_day.timestamp() * 1000)
    
    print(f"Auditing Binance Testnet from {start_of_day.isoformat()} to {now.isoformat()}...")
    
    # 1. Fetch Income History (REALIZED_PNL, COMMISSION, FUNDING_FEE)
    income = await client._request("GET", "/fapi/v1/income", signed=True, params={"startTime": start_time_ms, "limit": 1000})
    
    realized_events = []
    commission_events = []
    funding_events = []
    
    for item in income:
        itype = item.get("incomeType")
        amt = float(item.get("income", 0))
        t = datetime.datetime.fromtimestamp(item["time"] / 1000, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        sym = item.get("symbol", "")
        
        if itype == "REALIZED_PNL":
            realized_events.append((t, sym, amt, item.get("tranId")))
        elif itype == "COMMISSION":
            commission_events.append((t, sym, amt))
        elif itype == "FUNDING_FEE":
            funding_events.append((t, sym, amt))
            
    print("\n=== REALIZED PNL EVENTS TODAY ===")
    total_realized = sum(x[2] for x in realized_events)
    for t, sym, amt, tid in realized_events:
        print(f"  {t} | {sym:8s} | Realized PnL: ${amt:+9.4f} (tranId: {tid})")
    print(f"Total Realized PnL Today: ${total_realized:+9.4f}")
    
    print("\n=== COMMISSIONS (FEES) TODAY ===")
    total_fees = sum(x[2] for x in commission_events)
    print(f"Total Fees Paid Today: ${total_fees:9.4f} (Count: {len(commission_events)})")
    
    total_funding = sum(x[2] for x in funding_events)
    print(f"Total Funding Fees Today: ${total_funding:+9.4f}")
    
    net_today = total_realized + total_fees + total_funding
    print(f"\n>>> NET TODAY RESULT: ${net_today:+9.4f} <<<")
    
    # 2. Check Active Positions
    positions = await client.get_positions()
    active_pos = [p for p in positions if float(p.get("positionAmt", 0)) != 0]
    print(f"\n=== ACTIVE POSITIONS RIGHT NOW ({len(active_pos)}) ===")
    for p in active_pos:
        amt = float(p.get("positionAmt"))
        entry = float(p.get("entryPrice", 0))
        mark = float(p.get("markPrice", 0))
        unrealized = float(p.get("unRealizedProfit", 0))
        leverage = p.get("leverage")
        sym = p.get("symbol")
        side = "LONG" if amt > 0 else "SHORT"
        print(f"  {sym:8s} | {side:5s} {abs(amt):.3f} @ {entry:.4f} (Mark: {mark:.4f}) | UnPnL: ${unrealized:+8.2f} | Lev: {leverage}x")
        
    # 3. Check Account Balance
    acc = await client.get_account_details()
    print(f"\n=== WALLET BALANCE ===")
    print(f"  Total Wallet Balance: ${float(acc.get('balance', 0)):.2f}")
    print(f"  Available Balance:    ${float(acc.get('availableBalance', 0)):.2f}")
    print(f"  Total Unrealized PnL: ${float(acc.get('crossUnPnl', 0)):.2f}")

    # 4. For each symbol that had trades today, fetch user trades for fill prices
    traded_symbols = list(set([x[1] for x in realized_events] + [p.get('symbol') for p in active_pos]))
    print(f"\n=== RECENT USER TRADES FOR {traded_symbols} ===")
    for sym in traded_symbols:
        trades = await client._request("GET", "/fapi/v1/userTrades", signed=True, params={"symbol": sym, "startTime": start_time_ms, "limit": 50})
        for tr in trades:
            t = datetime.datetime.fromtimestamp(tr["time"] / 1000, datetime.timezone.utc).strftime("%H:%M:%S UTC")
            side = tr.get("side")
            price = float(tr.get("price"))
            qty = float(tr.get("qty"))
            rpnl = float(tr.get("realizedPnl"))
            comm = float(tr.get("commission"))
            print(f"  {t} | {sym:8s} | {side:4s} {qty} @ {price:.4f} | RealizedPnL: ${rpnl:+8.4f} | Fee: -${comm:.4f}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(audit_today())
