import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def check_klines():
    client = BinanceFuturesAdapter()
    try:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "XRPUSDT"]
        print("=== 1-HOUR KLINES (PAST 12 HOURS) ===")
        for sym in symbols:
            klines = await client._request("GET", "/fapi/v1/klines", params={"symbol": sym, "interval": "1h", "limit": 12})
            first_open = float(klines[0][1])
            last_close = float(klines[-1][4])
            high = max(float(k[2]) for k in klines)
            low = min(float(k[3]) for k in klines)
            net_change = ((last_close - first_open) / first_open) * 100.0
            range_pct = ((high - low) / low) * 100.0
            print(f"[{sym}] 12h Net Change: {net_change:+.2f}% | Total Range (High-Low): {range_pct:.2f}% | Current: ${last_close}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_klines())
