import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import asyncio
import uuid
from core.exchange.binance_client import BinanceFuturesAdapter

async def close_xrp():
    client = BinanceFuturesAdapter()
    try:
        positions = await client.get_positions()
        xrp_pos = next((p for p in positions if p.get("symbol") == "XRPUSDT"), None)
        
        if not xrp_pos or float(xrp_pos.get("positionAmt", 0)) == 0:
            print("No active XRPUSDT position found to close.")
            return

        amt = float(xrp_pos.get("positionAmt"))
        print(f"Found active XRPUSDT position: amt={amt}, entry={xrp_pos.get('entryPrice')}, uPnL={xrp_pos.get('unRealizedProfit')}")
        
        # If amt < 0 (SHORT), we BUY to close. If amt > 0 (LONG), we SELL to close.
        close_side = "BUY" if amt < 0 else "SELL"
        close_qty = abs(amt)
        client_order_id = f"MANUAL_CLOSE_{uuid.uuid4().hex[:12]}"
        
        print(f"Sending MARKET {close_side} {close_qty} XRP with reduce_only=True...")
        res = await client.create_order(
            symbol="XRPUSDT",
            side=close_side,
            quantity=close_qty,
            client_order_id=client_order_id,
            order_type="MARKET",
            reduce_only=True
        )
        print("Close Order Execution Result:", res)
        
        # Verify position is flat
        await asyncio.sleep(1)
        remaining = await client.get_positions()
        xrp_rem = next((p for p in remaining if p.get("symbol") == "XRPUSDT"), None)
        rem_amt = float(xrp_rem.get("positionAmt", 0)) if xrp_rem else 0.0
        print(f"Post-close XRPUSDT position quantity: {rem_amt}")
        
        acc = await client.get_account_details()
        print(f"Final Wallet Balance: {acc.get('balance')} USDT | Available: {acc.get('availableBalance')} USDT")
    except Exception as e:
        print(f"Error closing XRP position: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(close_xrp())
