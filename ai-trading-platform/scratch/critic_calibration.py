import sys, os
sys.path.insert(0, os.path.abspath("."))
import asyncio, numpy as np
from sqlalchemy import text
from core.database.session import SessionLocal
from core.exchange.binance_client import BinanceFuturesAdapter

with SessionLocal() as s:
    rows = s.execute(text("""
        SELECT action, COUNT(*), AVG(reward_5m), STDDEV(reward_5m), MIN(reward_5m), MAX(reward_5m),
               AVG(reward_1h), STDDEV(reward_1h), AVG(reward_4h), STDDEV(reward_4h)
        FROM experiences GROUP BY action
    """)).fetchall()
    for r in rows:
        print("[DB]", r[0], "n=", r[1], "r5m avg=%.3f std=%.3f min=%.2f max=%.2f | r1h avg=%.3f std=%.3f | r4h avg=%.3f std=%.3f" % tuple(float(x or 0) for x in r[2:]))
    rows = s.execute(text("SELECT AVG(reward_5m), AVG(reward_1h), AVG(reward_4h) FROM (SELECT * FROM experiences ORDER BY id DESC LIMIT 5000) t")).fetchall()
    print("[DB recent 5000] avg r5m/r1h/r4h:", rows[0])

async def main():
    b = BinanceFuturesAdapter()
    await b.connect()
    try:
        for sym in ["DOGEUSDT", "1000PEPEUSDT", "WIFUSDT", "ADAUSDT", "XRPUSDT", "ETHUSDT"]:
            k5 = await b.get_klines(sym, interval="5m", limit=13)
            c5 = np.array([float(k[4]) for k in k5])
            r5 = np.diff(c5) / c5[:-1] * 100
            k1 = await b.get_klines(sym, interval="1h", limit=25)
            c1 = np.array([float(k[4]) for k in k1])
            print(f"[MKT] {sym:12s} last 12x5m returns avg={r5.mean():+.3f}% absmax={np.abs(r5).max():.3f}% | 1h chg={((c1[-1]/c1[-2])-1)*100:+.2f}% 24h chg={((c1[-1]/c1[0])-1)*100:+.2f}%")
    finally:
        await b.close()
asyncio.run(main())
