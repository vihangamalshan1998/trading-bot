import json, re, collections, time
counts = collections.Counter()
per_sym = collections.defaultdict(collections.Counter)
lines = open("/root/.pm2/logs/trading-bot-out.log", errors="ignore").readlines()[-6000:]
for ln in lines:
    i = ln.find("{")
    if i < 0: continue
    try: d = json.loads(ln[i:])
    except Exception: continue
    m = d.get("message", "")
    if d.get("timestamp", "") < "2026-10-21T01:38": continue
    sm = re.match(r"\[(\w+)\]\s*(.*)", m)
    if not sm: continue
    sym, rest = sm.groups()
    key = re.sub(r"[-+]?\d[\d,.]*%?", "#", rest)[:90]
    counts[key] += 1
    per_sym[sym][key] += 1
for k, v in counts.most_common(30):
    print(v, k)
print("-----")
for s in ["DOGEUSDT", "1000PEPEUSDT", "WIFUSDT", "ADAUSDT", "XRPUSDT", "NEARUSDT", "ETHUSDT"]:
    print(s, per_sym[s].most_common(3))
