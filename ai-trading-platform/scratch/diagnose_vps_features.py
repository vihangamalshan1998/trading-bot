import redis
import json
import numpy as np
import torch
from apps.research.model import SingleSymbolActorCritic

r = redis.Redis(host='localhost', port=6379, db=0)

symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', '1000PEPEUSDT', 'ADAUSDT', 'RENDERUSDT']

print("="*60)
print("1. RAW MARKET FEATURES IN REDIS")
print("="*60)
for sym in symbols:
    raw = r.get(f"market:state:{sym}")
    if not raw:
        print(f"[{sym}] No market state in Redis!")
        continue
    data = json.loads(raw)
    feats = data.get("features", [])
    if not feats:
        print(f"[{sym}] Empty features!")
        continue
    arr = np.array(feats, dtype=float)
    top_indices = np.argsort(np.abs(arr))[-5:][::-1]
    top_str = ", ".join([f"slot {i}: {arr[i]:.2f}" for i in top_indices])
    print(f"[{sym:12s}] len={len(feats)} | min={np.min(arr):.2f} | max={np.max(arr):.2f} | mean={np.mean(arr):.2f} | std={np.std(arr):.2f}")
    print(f"    Top magnitudes -> {top_str}")

print("\n" + "="*60)
print("2. LIVE AI INFERENCE CHECK (model_v1.pt)")
print("="*60)
model = SingleSymbolActorCritic(input_dim=200)
checkpoint = torch.load("models/production/model_v1.pt", map_location="cpu")
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Check what the model outputs for BTC vs DOGE using their actual 90-dim features
for sym in symbols:
    raw = r.get(f"market:state:{sym}")
    if not raw:
        continue
    data = json.loads(raw)
    feats = data.get("features", [])
    if len(feats) != 90:
        continue
    
    # Reconstruct the 200-dim vector as _build_state_vector does
    # 10 portfolio, 10 position, 90 market, 20 macro, 70 canvas
    obs = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] # 10 portfolio
    obs += [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 10 position
    obs += list(feats) # 90 market
    obs += [0.0] * 20 # 20 macro
    obs += [0.0] * 70 # 70 canvas
    
    t = torch.tensor([[obs] * 30], dtype=torch.float32) # [1, 30, 200]
    with torch.no_grad():
        act, crit = model(t)
    
    act_dir = act[0, 0].item()
    conf = (act[0, 1].item() + 1.0) / 2.0
    size = max(0.25, (act[0, 2].item() + 1.0) / 2.0)
    c5m = crit[0, 0].item()
    c1h = crit[0, 1].item()
    c4h = crit[0, 2].item()
    side = "LONG" if act_dir > 0.2 else ("SHORT" if act_dir < -0.2 else "HOLD")
    print(f"[{sym:12s}] Model Output -> Side: {side:5s} (dir: {act_dir:+.3f}) | Conf: {conf*100:5.1f}% | Size: {size*100:5.1f}% | Critic: {c5m:+.4f}% (1h: {c1h:+.4f}%, 4h: {c4h:+.4f}%)")
