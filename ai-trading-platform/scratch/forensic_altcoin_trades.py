import redis
import json
import datetime

r = redis.Redis(host='127.0.0.1', port=6379)
raw_items = r.lrange('dashboard:trade_history', 0, -1)
items = [json.loads(x) for x in raw_items]
items.sort(key=lambda x: x.get('timestamp', 0))

print("=========================================================================================")
print("🔬 DEEP FORENSIC BREAKDOWN: ADA & XRP TRADES")
print("=========================================================================================")

# Group trades by pair of OPEN and CLOSE
opens = {x.get('symbol'): [] for x in items}
trades = []

for item in items:
    s = item.get('symbol')
    side = item.get('side', '')
    if 'OPEN' in side:
        opens[s].append(item)
    elif 'CLOSE' in side:
        # find matching open
        matched_open = opens[s].pop(0) if opens[s] else None
        trades.append((matched_open, item))

for sym in ['ADAUSDT', 'XRPUSDT']:
    sym_trades = [t for t in trades if t[1].get('symbol') == sym]
    print(f"\n>>> SYMBOL: {sym} (Total Closes: {len(sym_trades)}) <<<")
    for open_t, close_t in sym_trades:
        c_time = datetime.datetime.fromtimestamp(close_t.get('timestamp', 0)).strftime('%m/%d %H:%M:%S')
        duration_s = (close_t.get('timestamp', 0) - open_t.get('timestamp', 0)) if open_t else 0
        o_price = float(open_t.get('price', 0.0)) if open_t else 0.0
        c_price = float(close_t.get('price', 0.0))
        pnl = float(close_t.get('realized_pnl', 0.0))
        gross = float(close_t.get('gross_pnl', 0.0))
        fee = float(close_t.get('fees_paid', 0.0))
        qty = float(close_t.get('quantity', 0.0))
        side = close_t.get('side', '')
        exit_reason = close_t.get('exit_reason', 'N/A')
        roi = float(close_t.get('roi_pct', 0.0))
        
        o_action = open_t.get('action_val', 'N/A') if open_t else 'N/A'
        o_conf = open_t.get('confidence', 'N/A') if open_t else 'N/A'
        
        print(f"[{c_time}] {side:<12} | Dur: {duration_s:>4}s | In: {o_price:.4f} -> Out: {c_price:.4f} | Qty: {qty:,.0f} | Gross: ${gross:>6.2f} | Fee: ${fee:>5.2f} | Net: ${pnl:>6.2f} | Reason: {exit_reason}")
