import redis, json
r = redis.Redis()
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'RENDERUSDT']:
    val = r.get(f'ai:state:{sym}')
    print(f"[{sym}]: {val.decode() if val else None}")
