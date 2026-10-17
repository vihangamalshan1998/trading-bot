import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.db.redis import redis_manager

async def clean_history():
    await redis_manager.connect()
    raw_items = await redis_manager.redis.lrange("dashboard:trade_history", 0, -1)
    print(f"Total raw items before cleanup: {len(raw_items)}")
    
    cleaned_items = []
    removed_count = 0
    
    for item in raw_items:
        try:
            d = json.loads(item)
            order_id = str(d.get("order_id", ""))
            side = d.get("side", "")
            
            # If it's a phantom 'synced_' OPEN trade, remove it!
            if order_id.startswith("synced_") and "OPEN" in side:
                print(f"Removing phantom trade: {d.get('symbol')} {side} Qty: {d.get('quantity')} OrderId: {order_id}")
                removed_count += 1
                continue
                
            cleaned_items.append(item)
        except Exception as e:
            print(f"Error parsing item: {e}")
            
    print(f"Removed {removed_count} phantom synced trades.")
    print(f"Total remaining items: {len(cleaned_items)}")
    
    # Overwrite the Redis list with cleaned items
    if cleaned_items:
        await redis_manager.redis.delete("dashboard:trade_history")
        await redis_manager.redis.rpush("dashboard:trade_history", *cleaned_items)
        print("Successfully updated dashboard:trade_history in Redis!")
        
    # Print the top 15 trades now
    print("\n--- CURRENT TOP 15 CLEAN TRADES ---")
    current = await redis_manager.redis.lrange("dashboard:trade_history", 0, 14)
    for i, it in enumerate(current):
        d = json.loads(it)
        print(f"  {i+1}: {d.get('symbol')} | {d.get('side')} | Qty: {d.get('quantity')} | Px: ${d.get('price')} | Net PnL: ${d.get('realized_pnl', 0):+.2f}")
        
    await redis_manager.close()

if __name__ == "__main__":
    asyncio.run(clean_history())
