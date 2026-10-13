import asyncio
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    asset = await client.get_account_details()
    positions = await client.get_positions()
    
    print("=" * 65)
    print("BINANCE TESTNET ACCOUNT REALITY")
    print("=" * 65)
    print(f"Total Wallet Balance:      {float(asset.get('balance', 0)):.2f} USDT")
    print(f"Cross Wallet Balance:      {float(asset.get('crossWalletBalance', 0)):.2f} USDT")
    print(f"Cross Unrealized PnL:      {float(asset.get('crossUnPnl', 0)):+.2f} USDT")
    print(f"Available Balance:         {float(asset.get('availableBalance', 0)):.2f} USDT")
    print("-" * 65)
    
    print(f"Active Positions: {len(positions)}")
    for p in positions:
        sym = p['symbol']
        amt = float(p['positionAmt'])
        entry = float(p['entryPrice'])
        unpnl = float(p.get('unRealizedProfit', 0.0))
        lev = p['leverage']
        side = "LONG" if amt > 0 else "SHORT"
        print(f" -> {sym:<8} | {side:<5} | Qty: {amt:<10} | Entry: {entry:<9.4f} | UnPnL: {unpnl:+.2f} USDT | Lev: {lev}x")
    print("=" * 65)
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
