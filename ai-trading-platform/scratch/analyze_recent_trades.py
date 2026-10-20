import sys, os
sys.path.insert(0, os.path.abspath("."))
import datetime, time
from sqlalchemy import create_engine, text
from core.config.settings import settings

engine = create_engine(settings.database_url)

# Look at last 7 days (after Oct 15)
cutoff_time = int(time.time()) - (5 * 24 * 3600)

with engine.connect() as conn:
    query = text(f"""
        SELECT symbol, action, realized_pnl, fees, reward, reward_5m, entry_price, exit_price, timestamp
        FROM experiences
        WHERE timestamp >= {cutoff_time}
          AND (action IN ('CLOSE_LONG', 'CLOSE_SHORT', 'OPEN_LONG', 'OPEN_SHORT') 
               OR (realized_pnl IS NOT NULL AND realized_pnl != 0.0))
        ORDER BY timestamp ASC
    """)
    rows = conn.execute(query).fetchall()

print(f"Trades in last 5 days (since Oct 15): {len(rows)}")

wins = 0
losses = 0
total_pnl = 0.0
total_fees = 0.0
trades_summary = []

for r in rows:
    sym, act, pnl, fee, rew, r5, entry, exit_p, ts = r
    pnl = pnl or 0.0
    fee = fee or 0.0
    net = pnl - fee
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else 'N/A'
    
    # If it is a closing trade or has realized pnl
    if 'CLOSE' in act or pnl != 0.0:
        if net > 0:
            wins += 1
        elif net < 0:
            losses += 1
        total_pnl += net
        total_fees += fee
        trades_summary.append((dt, sym, act, net, fee, entry, exit_p))

print(f"Recent Closed Positions: {len(trades_summary)}")
print(f"Wins: {wins} | Losses: {losses} | Net PnL: ${total_pnl:+.4f} | Fees: ${total_fees:.4f}")
if wins + losses > 0:
    print(f"Win Rate: {(wins / (wins + losses) * 100):.1f}%")

print("\n--- RECENT COMPLETED TRADES ---")
for t in trades_summary[-25:]:
    print(f"[{t[0]}] {t[1]:<12} {t[2]:<12} Net: ${t[3]:+7.2f} (Fee: ${t[4]:.2f}, Entry: {t[5]}, Exit: {t[6]})")
