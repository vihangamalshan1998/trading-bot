import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    positions = await client.get_positions()
    ada_pos = [p for p in positions if 'ADA' in p.get('symbol', '') and float(p.get('positionAmt', 0)) != 0]
    print('Open ADA Positions:', ada_pos)
    orders = await client.get_all_orders('ADAUSDT', limit=10)
    for o in orders:
        dt = datetime.datetime.fromtimestamp(o.get('time', 0)/1000)
        print(f"Time: {dt} | Side: {o.get('side')} | Type: {o.get('type')} | Status: {o.get('status')} | OrigQty: {o.get('origQty')} | Executed: {o.get('executedQty')} | Price: {o.get('price')} | AvgPrice: {o.get('avgPrice')}")
    await client.close()

if __name__ == '__main__':
    asyncio.run(main())
