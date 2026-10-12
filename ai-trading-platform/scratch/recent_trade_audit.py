import pymysql
from datetime import datetime

conn = pymysql.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="ai_trading",
    cursorclass=pymysql.cursors.DictCursor
)

with conn.cursor() as c:
    c.execute("""
        SELECT id, symbol, action, entry_price, exit_price, realized_pnl, margin, fees, holding_duration, reward, timestamp 
        FROM experiences 
        WHERE action LIKE '%CLOSE%' 
        ORDER BY timestamp DESC 
        LIMIT 15;
    """)
    rows = c.fetchall()
    print("Latest 15 Closed Trades:")
    for r in rows:
        dt = datetime.fromtimestamp(r['timestamp']).strftime('%H:%M:%S')
        dur = round((r['holding_duration'] or 0) / 60, 1)
        roi = round((r['realized_pnl'] / r['margin']) * 100, 2) if r['margin'] and r['margin'] > 0 else 0
        sign = "+" if r['realized_pnl'] > 0 else ""
        print(f"[{dt}] {r['symbol']} {r['action']} | PnL: {sign}${r['realized_pnl']:.2f} | ROI: {roi}% | Fees: ${r['fees']:.2f} | Hold: {dur}m | Reward: {r['reward']:.2f}")

conn.close()
