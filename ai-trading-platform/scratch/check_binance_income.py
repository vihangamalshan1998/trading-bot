import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def check_binance_income():
    client = BinanceFuturesAdapter()
    now = datetime.datetime.now(datetime.timezone.utc)
    start_time = int((now - datetime.timedelta(minutes=10)).timestamp() * 1000)
    
    # Query income history (realized pnl, funding fee, commission)
    income = await client._request("GET", "/fapi/v1/income", signed=True, params={"startTime": start_time, "limit": 50})
    print("--- Binance Official Income History (Last 2 Hours) ---")
    for item in income:
        dt = datetime.datetime.fromtimestamp(item["time"] / 1000, datetime.timezone.utc).strftime("%H:%M:%S UTC")
        itype = item.get("incomeType")
        amt = float(item.get("income", 0))
        sym = item.get("symbol", "")
        asset = item.get("asset", "")
        info = item.get("info", "")
        print(f"  {dt} | {sym:<8} | Type: {itype:<16} | Amount: {amt:+10.4f} {asset} | Info: {info}")

    positions = await client.get_positions()
    active_pos = [p for p in positions if float(p.get("positionAmt", 0)) != 0]
    print("\n--- Active Positions on Binance ---")
    for p in active_pos:
        print(f"  {p.get('symbol')} | Amt: {p.get('positionAmt')} | Entry: {p.get('entryPrice')} | Mark: {p.get('markPrice')} | Unrealized PnL: {p.get('unRealizedProfit')} | Leverage: {p.get('leverage')}")

    orders = await client._request("GET", "/fapi/v1/openOrders", signed=True)
    print("\n--- Open Orders on Binance ---")
    for o in orders:
        print(f"  {o.get('symbol')} | Side: {o.get('side')} | Type: {o.get('type')} | Price: {o.get('price')} | OrigQty: {o.get('origQty')} | ExecQty: {o.get('executedQty')}")

    acc = await client.get_account_details()
    print("\n--- Current Binance Account ---")
    print(f"Wallet Balance:       {acc.get('balance')}")
    print(f"Cross Wallet Balance: {acc.get('crossWalletBalance')}")
    print(f"Cross Unrealized PnL: {acc.get('crossUnPnl')}")
    print(f"Available Balance:    {acc.get('availableBalance')}")
    await client.close()

if __name__ == "__main__":
    asyncio.run(check_binance_income())
