import redis
import json
import datetime

r = redis.Redis(host='127.0.0.1', port=6379)
raw = r.lrange('dashboard:trade_history', 0, -1)
items = [json.loads(x) for x in raw]
items.sort(key=lambda x: x.get('timestamp', 0))

print(f"Total raw items: {len(items)}")
for sym in ['ADAUSDT', 'XRPUSDT']:
    print(f"\n==================== {sym} RAW EVENTS ====================")
    sym_items = [x for x in items if x.get('symbol') == sym]
    for x in sym_items:
        dt = datetime.datetime.fromtimestamp(x.get('timestamp', 0)).strftime('%m/%d %H:%M:%S')
        side = x.get('side', '')
        px = float(x.get('price', 0.0))
        qty = float(x.get('quantity', 0.0))
        notional = px * qty
        pnl = x.get('realized_pnl')
        gross = x.get('gross_pnl')
        fee = x.get('fees_paid')
        reason = x.get('exit_reason', '')
        roi = x.get('roi_pct')
        print(f"{dt} | {side:<12} | Px: {px:.4f} | Qty: {qty:>7.0f} (${notional:>8.1f}) | Net: {str(pnl):>7} | Gross: {str(gross):>7} | Fee: {str(fee):>5} | Reason: {reason}")
