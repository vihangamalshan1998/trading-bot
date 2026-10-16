import json
from collections import Counter

skips = Counter()
symbols = set()

with open('/root/.pm2/logs/trading-bot-out.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            d = json.loads(line.strip())
            msg = d.get('message', '')
            if msg.startswith('[') and ']' in msg:
                sym = msg.split(']')[0][1:]
                symbols.add(sym)
                rest = msg.split(']')[1].strip()
                # Categorize
                if 'SKIPPING OPEN_LONG' in rest or 'SKIPPING OPEN_SHORT' in rest:
                    cat = rest.split(':')[0]
                elif 'HOLDING' in rest:
                    cat = 'HOLDING'
                elif 'EXECUTING' in rest:
                    cat = 'EXECUTING'
                elif '🛑' in rest or '🔒' in rest:
                    cat = rest.split(':')[0]
                else:
                    cat = rest[:40]
                skips[(sym, cat)] += 1
        except Exception:
            pass

print("=== SYMBOLS ACTIVE IN BOT ===")
print(sorted(list(symbols)))

print("\n=== TOP ACTIVITY PER SYMBOL (LAST LOG CHUNK) ===")
for sym in sorted(list(symbols)):
    print(f"\n--- {sym} ---")
    sym_skips = [(cat, cnt) for (s, cat), cnt in skips.items() if s == sym]
    sym_skips.sort(key=lambda x: x[1], reverse=True)
    for cat, cnt in sym_skips[:5]:
        print(f"  {cnt:5d}x : {cat}")
