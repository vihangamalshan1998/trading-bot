import asyncio
from core.exchange.binance_client import BinanceFuturesAdapter
from datetime import datetime

async def main():
    b = BinanceFuturesAdapter()
    await b.connect()
    income = await b._request("GET", "/fapi/v1/income", signed=True, params={"symbol": "XRPUSDT", "limit": 20})
    print(f"{'Time':<20} | {'Type':<15} | {'Income':<12} | {'Asset'}")
    print("-" * 60)
    for inc in income:
        t = datetime.fromtimestamp(inc.get("time", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{t:<20} | {inc.get('incomeType'):<15} | {inc.get('income'):<12} | {inc.get('asset')}")

if __name__ == "__main__":
    asyncio.run(main())
