import redis
import json

r = redis.Redis(host='127.0.0.1', port=6379)
raw = r.lrange('dashboard:trade_history', 0, -1)
items = [json.loads(x) for x in raw]
items.sort(key=lambda x: x.get('timestamp', 0))

# Group by symbol and find cases where multiple OPENs occurred before a CLOSE
symbol_events = {}
for x in items:
    s = x.get('symbol')
    if s not in symbol_events:
        symbol_events[s] = []
    symbol_events[s].append(x)

print("=====================================================")
print("📊 MULTI-ENTRY (PYRAMIDING / ADD-ON) AUDIT")
print("=====================================================")
total_trades = 0
multi_open_trades = 0

for sym, events in symbol_events.items():
    current_opens = []
    for ev in events:
        side = ev.get('side', '')
        if 'OPEN' in side:
            current_opens.append(ev)
        elif 'CLOSE' in side:
            total_trades += 1
            if len(current_opens) > 1:
                multi_open_trades += 1
                qtys = [float(o.get('quantity', 0)) for o in current_opens]
                prices = [float(o.get('price', 0)) for o in current_opens]
                pnl = float(ev.get('realized_pnl', 0))
                print(f"[{sym}] {len(current_opens)} OPENs before CLOSE! Qtys: {qtys} | Prices: {prices} | Net PnL: ${pnl:.2f}")
            current_opens = []

pct = (multi_open_trades / total_trades * 100) if total_trades else 0
print("-----------------------------------------------------")
print(f"Total Completed Trades: {total_trades}")
print(f"Trades with Multiple OPENs: {multi_open_trades} ({pct:.1f}%)")
