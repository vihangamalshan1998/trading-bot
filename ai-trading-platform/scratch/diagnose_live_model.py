import sys, os
import redis, json

r = redis.Redis(decode_responses=True)
for sym in ["1000PEPEUSDT", "DOGEUSDT", "NEARUSDT", "BTCUSDT", "WIFUSDT", "ADAUSDT", "SOLUSDT"]:
    ai_raw = r.get(f"ai:state:{sym}")
    print(f"[{sym}] Current Redis ai:state -> {ai_raw}")
