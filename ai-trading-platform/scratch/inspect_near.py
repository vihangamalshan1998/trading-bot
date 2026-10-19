import asyncio
import datetime
import sys
sys.path.insert(0, ".")
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    incomes = await client._request('GET', '/fapi/v1/income', signed=True, params={'symbol': 'NEARUSDT', 'limit': 10})
    for inc in incomes:
        t_ms = int(inc.get('time', 0))
        dt = datetime.datetime.fromtimestamp(t_ms / 1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        itype = inc.get('incomeType')
        inc_val = inc.get('income')
        print(f"[{dt}] {itype}: {inc_val} USDT")

if __name__ == '__main__':
    asyncio.run(main())
