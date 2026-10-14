import redis
import json
from datetime import datetime

r = redis.Redis(host='127.0.0.1', port=6379)
raw_trades = r.lrange('dashboard:trade_history', 0, 300)
print(f'Total trades in redis: {len(raw_trades)}')
parsed = []
for t in raw_trades:
    try:
        d = json.loads(t)
        parsed.append(d)
    except:
        pass

parsed.sort(key=lambda x: x.get('timestamp', 0))

print(f"{'SL Time':<18} | {'Symbol':<8} | {'Side':<12} | {'Price':<10} | {'EntryPx':<10} | {'NetPnL':<8} | {'ROI':<8} | {'Result'}")
print("-" * 90)
for d in parsed:
    ts = d.get('timestamp', 0)
    # If timestamp is already unix epoch
    # Sri Lanka is UTC+5.5 -> 19800s
    dt = datetime.fromtimestamp(ts + 19800).strftime('%m/%d %H:%M:%S')
    side = d.get('side', '')
    sym = d.get('symbol', '')
    price = d.get('price', 0.0)
    ep = d.get('entry_price', 0.0)
    realized = d.get('realized_pnl', 0.0)
    roi = d.get('roi_pct', 0.0)
    if 'CLOSE' in side:
        res = 'WIN' if realized > 0 else 'LOSS'
        print(f"{dt:<18} | {sym:<8} | {side:<12} | {price:<10.4f} | {ep:<10.4f} | {realized:>8.2f} | {roi:>7.2f}% | {res}")
    else:
        print(f"{dt:<18} | {sym:<8} | {side:<12} | {price:<10.4f} | {'-':<10} | {'-':>8} | {'-':>8} | OPEN")
