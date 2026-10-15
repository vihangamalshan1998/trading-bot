import asyncio
import sys
sys.path.insert(0, '.')
from core.exchange.binance_client import BinanceFuturesAdapter
from datetime import datetime

async def main():
    b = BinanceFuturesAdapter()
    await b.connect()
    income = await b._request("GET", "/fapi/v1/income", signed=True, params={"limit": 50})
    print(f"{'Time':<20} | {'Symbol':<10} | {'Type':<18} | {'Income':<12} | {'Asset'}")
    print("-" * 75)
    for inc in (income or []):
        t = datetime.fromtimestamp(inc.get("time", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{t:<20} | {inc.get('symbol', ''):<10} | {inc.get('incomeType'):<18} | {inc.get('income'):<12} | {inc.get('asset')}")

if __name__ == "__main__":
    asyncio.run(main())
