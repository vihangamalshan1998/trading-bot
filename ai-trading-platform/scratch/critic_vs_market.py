import sys, os
sys.path.insert(0, os.path.abspath("."))
import asyncio, numpy as np, redis, json
from core.exchange.binance_client import BinanceFuturesAdapter

r = redis.Redis(decode_responses=True)

async def main():
    b = BinanceFuturesAdapter()
    await b.connect()
    try:
        for sym in ["DOGEUSDT", "1000PEPEUSDT", "WIFUSDT", "ADAUSDT", "XRPUSDT", "ETHUSDT", "BTCUSDT"]:
            k5 = await b.get_klines(sym, interval="5m", limit=289)
            c5 = np.array([float(k[4]) for k in k5])
            r5 = np.diff(c5) / c5[:-1] * 100
            k1 = await b.get_klines(sym, interval="1h", limit=25)
            c1 = np.array([float(k[4]) for k in k1])
            ai = json.loads(r.get(f"ai:state:{sym}") or "{}")
            print(f"{sym:12s} 24h 5m-ret: mean={r5.mean():+.3f}% std={r5.std():.3f}% absmax={np.abs(r5).max():.2f}% | last1h={((c1[-1]/c1[-2])-1)*100:+.2f}% 24h={((c1[-1]/c1[0])-1)*100:+.2f}% || CRITIC 5m={ai.get('critic_return',0):+.2f}% 1h={ai.get('critic_1h',0):+.2f}% 4h={ai.get('critic_4h',0):+.2f}% side={ai.get('predicted_side')}")
    finally:
        await b.close()
asyncio.run(main())
