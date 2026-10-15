import asyncio
import time
from core.exchange.binance_client import BinanceFuturesAdapter
from datetime import datetime

async def main():
    b = BinanceFuturesAdapter()
    await b.connect()
    start_t = int((time.time() - 3600 * 4) * 1000)
    trades = await b._request("GET", "/fapi/v1/userTrades", signed=True, params={"symbol": "XRPUSDT", "startTime": start_t})
    print(f"{'Time':<20} | {'Side':<6} | {'Price':<10} | {'Qty':<10} | {'Realized PnL':<14} | {'Commission'}")
    print("-" * 80)
    for tr in trades[-10:]:
        t = datetime.fromtimestamp(tr.get("time", 0) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{t:<20} | {tr.get('side'):<6} | {tr.get('price'):<10} | {tr.get('qty'):<10} | {tr.get('realizedPnl'):<14} | {tr.get('commission')}")

if __name__ == "__main__":
    asyncio.run(main())
