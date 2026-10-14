import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())

from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    adapter = BinanceFuturesAdapter()
    await adapter.connect()
    
    # 1. Check open orders
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]:
        try:
            open_orders = await adapter.get_open_orders(sym)
            if open_orders:
                print(f"[{sym}] ACTIVE OPEN ORDERS ({len(open_orders)}):")
                for o in open_orders:
                    print(f"  -> OrderId: {o.get('orderId')} | Side: {o.get('side')} | Type: {o.get('type')} | Price: {o.get('price')} | OrigQty: {o.get('origQty')} | Status: {o.get('status')}")
            else:
                pass
        except Exception as e:
            print(f"[{sym}] Error checking orders: {e}")

    # 2. Check positions
    res = await adapter._request("GET", "/fapi/v2/account", signed=True)
    active_positions = [p for p in res.get("positions", []) if abs(float(p.get("positionAmt", 0.0))) > 0.0]
    print(f"\nACTIVE POSITIONS: {len(active_positions)}")
    for p in active_positions:
        print(f"  -> {p.get('symbol')} | Amt: {p.get('positionAmt')} | Entry: {p.get('entryPrice')} | PnL: {p.get('unrealizedProfit')}")

if __name__ == "__main__":
    asyncio.run(main())
