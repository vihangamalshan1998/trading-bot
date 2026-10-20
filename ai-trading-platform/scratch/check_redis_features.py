import redis, json
r = redis.Redis()
symbols = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "LINKUSDT",
    "ADAUSDT", "AVAXUSDT", "NEARUSDT", "LTCUSDT", "DOGEUSDT", "RENDERUSDT",
    "1000PEPEUSDT", "APTUSDT", "TAOUSDT", "WIFUSDT", "INJUSDT"
]
for sym in symbols:
    raw = r.get(f"market:state:{sym}")
    if not raw:
        print(f"[{sym:12s}] NO MARKET STATE IN REDIS!")
        continue
    m = json.loads(raw)
    feats = m.get("features", [])
    mid = m.get("mid_price", 0)
    print(f"[{sym:12s}] mid={mid:10.4f} | features len={len(feats)} | sum={sum(feats):10.4f} | min={min(feats) if feats else 0:8.4f} | max={max(feats) if feats else 0:8.4f}")
