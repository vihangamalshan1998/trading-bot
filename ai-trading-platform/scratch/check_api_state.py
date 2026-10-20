import urllib.request, json

data = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/state").read())
states = data.get("market_states", {})
print(f"Total market states from API: {len(states)}")
for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "DOGEUSDT", "ADAUSDT", "XRPUSDT", "WIFUSDT", "1000PEPEUSDT"]:
    if sym in states:
        v = states[sym]
        conf = round(v.get("ai_confidence", 0) * 100, 1)
        size = round(v.get("ai_target_size", 0) * 100, 1)
        c5m = round(v.get("ai_critic_return", 0), 3)
        side = v.get("ai_predicted_side", "N/A")
        print(f"[{sym:12s}] Decision: {side:5s} | Conf: {conf:4.1f}% | Size: {size:4.1f}% | Critic 5m: {c5m:+.3f}%")
