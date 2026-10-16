import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    klines = await client._request('GET', '/fapi/v1/klines', params={'symbol': 'ADAUSDT', 'interval': '1m', 'limit': 20})
    for k in klines:
        t = datetime.datetime.fromtimestamp(k[0]/1000)
        o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
        print(f"{t.strftime('%H:%M:%S')} | Open: {o:.4f} | High: {h:.4f} | Low: {l:.4f} | Close: {c:.4f} | Change: {(c-o)/o*100:+.2f}%")
    await client.close()

if __name__ == '__main__':
    asyncio.run(main())
