import redis
import json
import time

r = redis.Redis(host="localhost", port=6379, db=0)
keys = sorted(r.keys("ai:state:*"))
print(f"Total ai:state keys found: {len(keys)}")
now = time.time()
for k in keys:
    val = r.get(k)
    if val:
        data = json.loads(val.decode())
        ts = data.get("timestamp")
        age = (now - ts) if ts else None
        side = data.get("predicted_side")
        conf = data.get("confidence")
        target = data.get("target_size")
        c5m = data.get("critic_return")
        c1h = data.get("critic_1h")
        c4h = data.get("critic_4h")
        print(f"[{k.decode():18s}] side={side} conf={conf:.4f} target={target:.4f} c5m={c5m:+.5f} c1h={c1h:+.5f} c4h={c4h:+.5f} age={age:.1f}s" if (conf is not None and c5m is not None) else f"[{k.decode():18s}] {data}")
