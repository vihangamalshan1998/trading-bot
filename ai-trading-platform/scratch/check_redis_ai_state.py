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
        age_str = f"{age:.1f}s" if age is not None else "N/A"
        side = data.get("predicted_side")
        conf = data.get("confidence")
        target = data.get("target_size")
        c5m = data.get("critic_return")
        c1h = data.get("critic_1h")
        c4h = data.get("critic_4h")
        if conf is not None and c5m is not None and target is not None:
            c1h_val = c1h if c1h is not None else 0.0
            c4h_val = c4h if c4h is not None else 0.0
            print(f"[{k.decode():18s}] side={side:5s} conf={conf*100:5.1f}% target={target*100:5.1f}% c5m={c5m:+.5f} c1h={c1h_val:+.5f} c4h={c4h_val:+.5f} age={age_str}")
        else:
            print(f"[{k.decode():18s}] {data}")
