import redis, json

r = redis.Redis()
keys = sorted(r.keys('ai:state:*'))
for k in keys:
    val = json.loads(r.get(k))
    sym = k.decode().replace('ai:state:', '')
    print(f"[{sym:10s}] Side: {val.get('predicted_side', 'N/A'):5s} | Conf: {val.get('confidence', 0)*100:5.1f}% | Size: {val.get('target_size', 0)*100:5.1f}% | Critic: {val.get('critic_return', 0):+.3f}%")
