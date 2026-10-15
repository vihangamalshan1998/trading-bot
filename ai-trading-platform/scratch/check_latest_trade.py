import json
import redis
import datetime

r = redis.Redis(host="localhost", port=6379, db=0)
raw = r.lrange("dashboard:trade_history", 0, 9)

print("--- Latest 10 Trade Events in Redis ---")
for x in raw:
    t = json.loads(x)
    dt = datetime.datetime.fromtimestamp(t.get("timestamp", 0), datetime.timezone.utc).strftime("%H:%M:%S UTC")
    sym = t.get("symbol", "")
    side = t.get("side", "")
    gross = float(t.get("gross_pnl", 0))
    fees = float(t.get("fees_paid", 0))
    net = float(t.get("realized_pnl", 0))
    roi = float(t.get("roi_pct", 0))
    price = float(t.get("price", 0))
    entry_px = float(t.get("entry_price", 0))
    qty = float(t.get("quantity", 0))
    print(f"{dt} | {sym:<8} | {side:<11} | Entry: {entry_px:.4f} | Exit: {price:.4f} | Qty: {qty:.1f} | Gross: ${gross:+7.4f} | Fees: ${fees:6.4f} | Net: ${net:+7.4f} (ROI: {roi:+5.2f}%)")

portfolio_raw = r.get("dashboard:portfolio")
if portfolio_raw:
    p = json.loads(portfolio_raw)
    print("\n--- Current Dashboard Portfolio ---")
    print("Equity:", p.get("equity"))
    print("Active Positions:", p.get("positions"))
