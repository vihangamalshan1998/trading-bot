import sys, os
sys.path.insert(0, os.path.abspath("."))
import redis, json, asyncio
from core.config.settings import settings
from core.exchange.binance_client import BinanceFuturesAdapter

r = redis.Redis(decode_responses=True)
all_keys = r.keys("*")
print("Total Redis Keys:", len(all_keys))
matching = [k for k in all_keys if any(x in k.lower() for x in ['account', 'balance', 'portfolio', 'wallet', 'dashboard', 'user'])]
print("Matching Redis keys:", matching)

for k in matching:
    t = r.type(k)
    if t == 'string':
        val = r.get(k)
        print(f"  [{k}] ({t}): {val[:150] if val else None}")

async def check_binance():
    b = BinanceFuturesAdapter()
    await b.connect()
    try:
        acc = await b.get_account_balance()
        print("\nBinance Real/Testnet Account Balance:")
        if isinstance(acc, list):
            for a in acc:
                wallet = float(a.get('balance', 0.0))
                if wallet > 0:
                    print(f"  Asset: {a.get('asset')}, Balance: {wallet}, Available: {a.get('availableBalance')}")
        elif isinstance(acc, dict):
            print(acc)
    except Exception as e:
        print("Binance fetch error:", e)
    finally:
        await b.close()

asyncio.run(check_binance())
