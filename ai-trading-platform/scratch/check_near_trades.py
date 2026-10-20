import urllib.request, json

data = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/state").read())
states = data.get("market_states", {})

print(f"=== EVALUATING ALL {len(states)} SYMBOLS FOR TRADE READINESS ===")
candidates = []
for sym, v in states.items():
    conf = round(v.get("ai_confidence", 0) * 100, 1)
    size = round(v.get("ai_target_size", 0) * 100, 1)
    c5m = round(v.get("ai_critic_return", 0), 3)
    c1h = round(v.get("ai_critic_1h", 0), 3)
    side = v.get("ai_predicted_side", "N/A")
    spread = round(v.get("spread", 0) / v.get("mid_price", 1) * 100, 3) if v.get("mid_price", 0) > 0 else 0
    
    score = conf * abs(c5m) if (conf >= 70 and side in ["LONG", "SHORT"]) else 0
    candidates.append({
        "symbol": sym,
        "side": side,
        "conf": conf,
        "size": size,
        "c5m": c5m,
        "c1h": c1h,
        "spread": spread,
        "score": round(score, 2)
    })

# Sort by readiness / opportunity score
candidates.sort(key=lambda x: x["score"], reverse=True)

print(f"{'Symbol':<12} | {'Side':<5} | {'Conf %':<6} | {'Size %':<6} | {'Critic 5m':<9} | {'Critic 1h':<9} | {'Opportunity Score'}")
print("-" * 75)
for c in candidates:
    print(f"{c['symbol']:<12} | {c['side']:<5} | {c['conf']:>5.1f}% | {c['size']:>5.1f}% | {c['c5m']:>+8.3f}% | {c['c1h']:>+8.3f}% | Score: {c['score']}")
