import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    positions = await client.get_positions()
    print("Found positions:", len(positions))
    for p in positions:
        sym = p.get('symbol')
        amt = p.get('positionAmt')
        ep = p.get('entryPrice')
        ut = p.get('updateTime', 0)
        dt = datetime.datetime.fromtimestamp(float(ut) / 1000.0) if ut else "N/A"
        print(f"[{sym}] Amt: {amt}, Entry: {ep}, updateTime: {ut} ({dt})")
    await client.close()

if __name__ == '__main__':
    asyncio.run(main())
