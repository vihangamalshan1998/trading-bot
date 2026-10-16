import json
from collections import Counter

skips = Counter()
orders = []
today_str = "2026-10-16"

with open('/root/.pm2/logs/trading-bot-out.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            d = json.loads(line.strip())
            ts = d.get('timestamp', '')
            if not ts.startswith(today_str):
                continue
            msg = d.get('message', '')
            if 'EXECUTING' in msg or 'Placing Order' in msg:
                orders.append((ts, msg))
            if msg.startswith('[') and ']' in msg:
                sym = msg.split(']')[0][1:]
                rest = msg.split(']')[1].strip()
                cat = rest.split(':')[0]
                skips[(sym, cat)] += 1
        except Exception:
            pass

print(f"=== ALL EXECUTIONS / ORDERS PLACED TODAY ({today_str}) ===")
for ts, o in orders:
    print(f"{ts} : {o}")

print("\n=== TOP 3 GATES / REASONS PER SYMBOL TODAY ===")
symbols = set(s for s, _ in skips.keys())
for sym in sorted(list(symbols)):
    print(f"\n--- {sym} ---")
    sym_skips = [(cat, cnt) for (s, cat), cnt in skips.items() if s == sym]
    sym_skips.sort(key=lambda x: x[1], reverse=True)
    for cat, cnt in sym_skips[:3]:
        print(f"  {cnt:5d}x : {cat}")
