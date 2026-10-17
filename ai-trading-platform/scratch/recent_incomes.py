import asyncio
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.exchange.binance_client import BinanceFuturesAdapter

async def run():
    client = BinanceFuturesAdapter()
    srv_time_res = await client._request("GET", "/fapi/v1/time")
    srv_ms = srv_time_res.get("serverTime")
    start_ms = srv_ms - (60 * 60 * 1000)
    
    incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={"startTime": start_ms, "limit": 100})
    print(f"Total income records in last 60m: {len(incomes)}")
    for inc in incomes:
        t_ms = inc.get("time", 0)
        t_str = datetime.datetime.fromtimestamp(t_ms / 1000.0, tz=datetime.timezone.utc).strftime("%H:%M:%S UTC")
        print(f"[{t_str}] {inc.get('symbol')} | {inc.get('incomeType')} | {float(inc.get('income', 0)):+.4f} USDT | {inc.get('info')}")
        
    await client.close()

if __name__ == "__main__":
    asyncio.run(run())
