import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    client = BinanceFuturesAdapter()
    try:
        positions = await client.get_positions()
        print("=== BINANCE FUTURES LIVE POSITIONS ===")
        found = False
        for p in positions:
            amt = float(p.get("positionAmt", 0))
            if amt != 0:
                found = True
                print(f"SYMBOL: {p.get('symbol')} | QTY: {amt} | ENTRY: {p.get('entryPrice')} | MARK: {p.get('markPrice')} | UPNL: {p.get('unRealizedProfit')}")
        if not found:
            print("NO ACTIVE POSITIONS")
            
        account = await client.get_account_balance()
        print(f"ACCOUNT BALANCE: {account.get('totalWalletBalance', 'N/A')} USDT | UNREALIZED PNL: {account.get('totalUnrealizedProfit', 'N/A')}")
        
        # Also check open orders
        for sym in ["NEARUSDT", "ADAUSDT", "XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            orders = await client.get_open_orders(sym)
            if orders:
                print(f"OPEN ORDERS for {sym}: {orders}")
    except Exception as e:
        print(f"Error querying Binance: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
