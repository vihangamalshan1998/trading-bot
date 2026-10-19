import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.db.redis import redis_manager
from core.logging.logger import logger

async def clean_history():
    await redis_manager.connect()
    if not redis_manager.redis:
        print("Could not connect to Redis.")
        return

    raw_items = await redis_manager.redis.lrange("dashboard:trade_history", 0, -1)
    print(f"Total raw trades before deduplication: {len(raw_items)}")
    
    cleaned_items = []
    seen_trades = set()
    removed_count = 0

    for item in raw_items:
        try:
            d = json.loads(item)
            sym = d.get("symbol", "")
            side = d.get("side", "")
            qty = round(float(d.get("quantity", 0)), 2)
            px = round(float(d.get("price", 0)), 4)
            entry_px = round(float(d.get("entry_price", 0)), 4)
            pnl = round(float(d.get("realized_pnl", 0)), 2)
            order_id = str(d.get("order_id", ""))

            # Remove phantom synced open trades if already closed
            if order_id.startswith("synced_") and "OPEN" in side:
                print(f"Removing phantom auto-reconciled OPEN: {sym} {side} (order: {order_id})")
                removed_count += 1
                continue

            # Deduplicate repeated identical CLOSE attempts with identical quantity and entry
            # Fingerprint: (sym, side, qty, entry_px, pnl)
            fingerprint = (sym, side, qty, entry_px, pnl)
            if "CLOSE" in side and fingerprint in seen_trades:
                print(f"Removing duplicate phantom attempt: {sym} {side} Qty: {qty} Px: ${px} PnL: ${pnl}")
                removed_count += 1
                continue

            seen_trades.add(fingerprint)
            cleaned_items.append(item)
        except Exception as e:
            print(f"Error parsing trade: {e}")
            cleaned_items.append(item)

    print(f"\nRemoved {removed_count} duplicate / phantom trades.")
    print(f"Remaining clean trades: {len(cleaned_items)}")

    if removed_count > 0 and cleaned_items:
        await redis_manager.redis.delete("dashboard:trade_history")
        await redis_manager.redis.rpush("dashboard:trade_history", *cleaned_items)
        print("✅ Successfully cleaned and updated 'dashboard:trade_history' in Redis!")
    else:
        print("No changes needed in Redis.")

    await redis_manager.close()

if __name__ == "__main__":
    asyncio.run(clean_history())
