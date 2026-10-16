import redis
import json

r = redis.Redis(host='127.0.0.1', port=6379)
raw_items = r.lrange('dashboard:trade_history', 0, -1)
items = [json.loads(x) for x in raw_items]
items.sort(key=lambda x: x.get('timestamp', 0))

closes = [x for x in items if 'CLOSE' in x.get('side', '')]

total_gross = sum(float(x.get('gross_pnl', 0.0)) for x in closes)
total_fees = sum(float(x.get('fees_paid', 0.0)) for x in closes)
total_net = sum(float(x.get('realized_pnl', 0.0)) for x in closes)

wins = [x for x in closes if float(x.get('realized_pnl', 0.0)) > 0]
losses = [x for x in closes if float(x.get('realized_pnl', 0.0)) < 0]

print("====================================================")
print("📊 COMPLETE TRADE PERFORMANCE AUDIT")
print("====================================================")
print(f"Total Completed Trades: {len(closes)}")
print(f"Wins:                   {len(wins)} ({len(wins)/len(closes)*100:.1f}%)" if closes else "No trades")
print(f"Losses:                 {len(losses)} ({len(losses)/len(closes)*100:.1f}%)" if closes else "No trades")
print(f"Total Gross PnL:        ${total_gross:+.2f} USDT")
print(f"Total Fees Paid:        ${total_fees:.2f} USDT")
print(f"Total Net Realized PnL: ${total_net:+.2f} USDT")
if wins:
    avg_win = sum(float(x.get('realized_pnl', 0.0)) for x in wins) / len(wins)
    print(f"Avg Win:                +${avg_win:.2f}")
if losses:
    avg_loss = sum(float(x.get('realized_pnl', 0.0)) for x in losses) / len(losses)
    print(f"Avg Loss:               ${avg_loss:.2f}")

print("\n--- Breakdown By Coin ---")
syms = {}
for c in closes:
    s = c.get('symbol', 'UNKNOWN')
    if s not in syms:
        syms[s] = {'wins': 0, 'losses': 0, 'gross': 0.0, 'fees': 0.0, 'net': 0.0, 'trades': 0}
    syms[s]['trades'] += 1
    pnl = float(c.get('realized_pnl', 0.0))
    gross = float(c.get('gross_pnl', 0.0))
    fee = float(c.get('fees_paid', 0.0))
    syms[s]['gross'] += gross
    syms[s]['fees'] += fee
    syms[s]['net'] += pnl
    if pnl > 0:
        syms[s]['wins'] += 1
    elif pnl < 0:
        syms[s]['losses'] += 1

for s, d in sorted(syms.items(), key=lambda x: x[1]['net']):
    wr = (d['wins'] / d['trades'] * 100) if d['trades'] else 0.0
    print(f"{s:<10} | Trades: {d['trades']:>2} | WR: {wr:>5.1f}% ({d['wins']}W/{d['losses']}L) | Gross: ${d['gross']:>7.2f} | Fees: ${d['fees']:>6.2f} | Net: ${d['net']:>7.2f}")

print("\n--- Last 10 Trades Chronological ---")
for c in closes[-10:]:
    import datetime
    dt = datetime.datetime.fromtimestamp(c.get('timestamp', 0)).strftime('%m/%d %H:%M:%S')
    pnl = float(c.get('realized_pnl', 0.0))
    s = c.get('symbol')
    side = c.get('side')
    roi = float(c.get('roi_pct', 0.0))
    fee = float(c.get('fees_paid', 0.0))
    print(f"{dt} | {s:<8} | {side:<12} | Net: ${pnl:>7.2f} (ROI: {roi:>6.2f}%) | Fee: ${fee:.2f}")
