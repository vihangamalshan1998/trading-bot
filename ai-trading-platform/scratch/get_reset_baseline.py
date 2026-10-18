import asyncio
import sys
import datetime

sys.path.insert(0, ".")
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    
    acc = await client.get_account_details()
    wallet_balance = float(acc.get('balance', 0.0))
    cross_unpnl = float(acc.get('crossUnPnl', 0.0))
    equity = wallet_balance + cross_unpnl
    
    now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    now_dt = datetime.datetime.fromtimestamp(now_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    positions = await client.get_positions()
    active_pos = [p for p in positions if abs(float(p.get('positionAmt', 0))) > 1e-5]
    
    print("=== CURRENT BINANCE STATUS FOR 72-HOUR RESET ===")
    print(f"Timestamp MS:       {now_ms}")
    print(f"Timestamp UTC:      {now_dt}")
    print(f"Wallet Balance:     {wallet_balance:.4f} USDT")
    print(f"Cross UnPnl:        {cross_unpnl:.4f} USDT")
    print(f"Total Equity:       {equity:.4f} USDT")
    print(f"Active Positions:   {len(active_pos)}")
    for p in active_pos:
        sym = p.get('symbol')
        amt = p.get('positionAmt')
        ep = p.get('entryPrice')
        unpnl = p.get('unRealizedProfit')
        print(f"  - {sym}: {amt} @ {ep} (UnPnl: {unpnl})")

if __name__ == "__main__":
    asyncio.run(main())
