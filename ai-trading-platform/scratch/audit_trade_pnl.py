import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import datetime
from core.exchange.binance_client import BinanceFuturesAdapter

async def print_trades(symbol):
    client = BinanceFuturesAdapter()
    try:
        trades = await client._request("GET", "/fapi/v1/userTrades", signed=True, params={"symbol": symbol, "limit": 20})
        print(f"\n=== RECENT TRADES FOR {symbol} ===")
        tot_realized = 0.0
        tot_qty = 0.0
        tot_commission = 0.0
        for t in trades:
            ts = datetime.datetime.fromtimestamp(t.get('time', 0)/1000.0, tz=datetime.timezone.utc)
            realized = float(t.get('realizedPnl', 0))
            commission = float(t.get('commission', 0))
            qty = float(t.get('qty', 0))
            tot_realized += realized
            tot_commission += commission
            tot_qty += qty
            print(f"[{ts.strftime('%H:%M:%S')} UTC] {t.get('side')} {qty} @ ${t.get('price')} | Realized PnL: ${realized:+.4f} | Fee: ${commission:.4f}")
        print(f"Total Realized PnL for {symbol}: ${tot_realized:+.2f} | Total Commission: ${tot_commission:+.2f}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()

async def main():
    await print_trades("ADAUSDT")
    await print_trades("XRPUSDT")
    await print_trades("NEARUSDT")

if __name__ == "__main__":
    asyncio.run(main())
