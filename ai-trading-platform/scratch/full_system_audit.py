import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import json
from core.exchange.binance_client import BinanceFuturesAdapter

async def audit():
    client = BinanceFuturesAdapter()
    try:
        # 1. Account details
        acc = await client.get_account_details()
        print("=== BINANCE ACCOUNT AUDIT ===")
        print(f"Asset: {acc.get('asset')}")
        print(f"Wallet Balance: {acc.get('balance')} USDT")
        print(f"Cross Wallet Balance: {acc.get('crossWalletBalance')} USDT")
        print(f"Available Balance: {acc.get('availableBalance')} USDT")
        print(f"Cross Unrealized PnL: {acc.get('crossUnPnl')} USDT")
        
        wallet_bal = float(acc.get('balance', 0))
        cross_pnl = float(acc.get('crossUnPnl', 0))
        total_equity = wallet_bal + cross_pnl
        print(f"CALCULATED TOTAL EQUITY: {total_equity:.2f} USDT")
        
        # 2. Position Risk details
        positions = await client.get_positions()
        print("\n=== OPEN POSITIONS AUDIT ===")
        total_margin_used = 0.0
        for p in positions:
            amt = float(p.get('positionAmt', 0))
            if amt != 0:
                sym = p.get('symbol')
                entry = float(p.get('entryPrice', 0))
                mark = float(p.get('markPrice', 0))
                upnl = float(p.get('unRealizedProfit', 0))
                lev = int(p.get('leverage', 20))
                notional = abs(amt) * mark
                margin = notional / lev
                total_margin_used += margin
                pnl_pct = (upnl / margin) * 100 if margin > 0 else 0.0
                print(f"[{sym}] Side: {'LONG' if amt > 0 else 'SHORT'} | Qty: {amt} | Entry: ${entry:.4f} | Mark: ${mark:.4f} | Notional: ${notional:.2f} | Margin: ${margin:.2f} (Lev: {lev}x) | uPnL: ${upnl:+.2f} ({pnl_pct:+.2f}%)")
        print(f"Total Margin Committed: ${total_margin_used:.2f} USDT")
        
        # 3. Check open orders
        print("\n=== OPEN ORDERS AUDIT ===")
        for sym in ["ADAUSDT", "XRPUSDT", "NEARUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            open_ords = await client.get_open_orders(sym)
            if open_ords:
                print(f"[{sym}] Active Orders: {len(open_ords)} -> {open_ords}")
        print("No hanging open orders detected." if not any(open_ords for sym in ["ADAUSDT", "XRPUSDT", "NEARUSDT"]) else "")
        
    except Exception as e:
        print(f"Audit error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(audit())
