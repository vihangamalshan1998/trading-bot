import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())

from core.exchange.binance_client import BinanceFuturesAdapter

async def main():
    adapter = BinanceFuturesAdapter()
    await adapter.connect()
    
    # 1. Account balances & positions
    res = await adapter._request("GET", "/fapi/v2/account", signed=True)
    total_margin_balance = float(res.get("totalMarginBalance", 0.0))
    total_wallet_balance = float(res.get("totalWalletBalance", 0.0))
    total_unrealized = float(res.get("totalUnrealizedProfit", 0.0))
    
    print(f"\n================ LIVE BINANCE STATUS ================")
    print(f"Total Margin Balance:  ${total_margin_balance:.2f} USDT")
    print(f"Total Wallet Balance:  ${total_wallet_balance:.2f} USDT")
    print(f"Total Unrealized PnL:  ${total_unrealized:+.2f} USDT")
    print("-----------------------------------------------------")
    
    active_positions = [
        p for p in res.get("positions", []) 
        if abs(float(p.get("positionAmt", 0.0))) > 0.0
    ]
    
    if not active_positions:
        print("NO ACTIVE POSITIONS CURRENTLY OPEN.")
    else:
        print(f"ACTIVE POSITIONS ({len(active_positions)}):")
        for p in active_positions:
            sym = p.get("symbol")
            amt = float(p.get("positionAmt"))
            entry = float(p.get("entryPrice"))
            mark = float(p.get("markPrice", entry))
            pnl = float(p.get("unrealizedProfit"))
            side = "LONG" if amt > 0 else "SHORT"
            lev = p.get("leverage")
            price_change_pct = ((mark - entry) / entry * 100.0) if side == "LONG" else ((entry - mark) / entry * 100.0)
            print(f" • {sym:8s} | {side:5s} | Qty: {amt:10.4f} | Entry: ${entry:.4f} | Mark: ${mark:.4f} | Price Move: {price_change_pct:+.2f}% | PnL: ${pnl:+.2f} USDT ({lev}x)")
    
    print("=====================================================\n")

if __name__ == "__main__":
    asyncio.run(main())
