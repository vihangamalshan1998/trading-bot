import asyncio, sys
sys.path.insert(0, '.')
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    b = BinanceFuturesAdapter()
    await b.connect()
    positions = await b.get_positions()
    print("--- ACTIVE POSITIONS ---")
    for p in positions:
        amt = float(p.get('positionAmt', 0))
        if abs(amt) > 0:
            print(f"{p['symbol']}: amt={amt}, entryPx={p.get('entryPrice')}, unPnL={p.get('unRealizedProfit')}")
    await b.close()

if __name__ == '__main__':
    asyncio.run(main())
