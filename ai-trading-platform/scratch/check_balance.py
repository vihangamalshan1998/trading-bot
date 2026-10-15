import json
import redis
import datetime

r = redis.Redis(host="localhost", port=6379, db=0)

# Check portfolio state in Redis
portfolio_raw = r.get("dashboard:portfolio")
if portfolio_raw:
    p = json.loads(portfolio_raw)
    print("Current Dashboard Portfolio Equity:", p.get("equity"))
    print("Positions:", p.get("positions"))

# Check all trades in Redis after 23:48:04 UTC (5:18:04 AM local)
t_reset = datetime.datetime(2026, 10, 14, 23, 48, 4, tzinfo=datetime.timezone.utc).timestamp()
raw_trades = r.lrange("dashboard:trade_history", 0, -1)

trades_since_reset = []
for x in raw_trades:
    t = json.loads(x)
    ts = t.get("timestamp", 0)
    if "CLOSE" in t.get("side", "") and ts >= t_reset:
        trades_since_reset.append(t)

print(f"\nClosed trades since 5:18 AM reset ({len(trades_since_reset)} trades):")
net_sum = 0.0
total_fees = 0.0
wins = 0
losses = 0

for t in trades_since_reset:
    dt = datetime.datetime.fromtimestamp(t.get("timestamp", 0), datetime.timezone.utc).strftime("%H:%M:%S UTC")
    pnl = float(t.get("realized_pnl", 0))
    fees = float(t.get("fees_paid", 0))
    net_sum += pnl
    total_fees += fees
    if pnl > 0: wins += 1
    elif pnl < 0: losses += 1
    sym = t.get("symbol", "")
    side = t.get("side", "")
    print(f"  {dt} | {sym:<8} {side:<11} | PnL: ${pnl:+8.4f} | Fees: ${fees:6.4f}")

print(f"\nTotal Realized Net PnL since 5:18 AM: ${net_sum:+.4f} USDT")
print(f"Total Fees Paid since 5:18 AM:         ${total_fees:.4f} USDT")
print(f"Wins: {wins}, Losses: {losses}")
