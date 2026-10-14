import pymysql
import os
import sys
from datetime import datetime

# Connect to database (local or via tunnel)
try:
    conn = pymysql.connect(
        host=os.getenv('MYSQL_HOST', '127.0.0.1'),
        port=int(os.getenv('MYSQL_PORT', 3307)),
        user=os.getenv('MYSQL_USER', 'ai_trading'),
        password=os.getenv('MYSQL_PASSWORD', 'ai_trading_password'),
        database=os.getenv('MYSQL_DATABASE', 'ai_trading')
    )
except Exception:
    conn = pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password='',
        database='ai_trading'
    )

with conn.cursor() as cur:
    print("=== TOP WINNING TRADES (LAST 24 HOURS) ===")
    cur.execute("""
        SELECT symbol, action, FROM_UNIXTIME(timestamp) as trade_time, realized_pnl, fees, (realized_pnl - fees) as net_pnl
        FROM experiences
        WHERE action != 'HOLD' AND realized_pnl > 0
        ORDER BY realized_pnl DESC
        LIMIT 25
    """)
    rows = cur.fetchall()
    if rows:
        for r in rows:
            sym, act, t, gross, fees, net = r
            print(f" {t} | {sym:8s} | {act:12s} | Gross: +${gross:.2f} | Fees: -${fees:.2f} | Net: ${net:+.2f}")
    else:
        print("No winning records found in experiences table.")

    print("\n=== TOTAL GROSS PNL BY SYMBOL (LAST 24 HOURS) ===")
    cur.execute("""
        SELECT symbol, COUNT(*) as trades, SUM(realized_pnl) as total_gross, SUM(fees) as total_fees
        FROM experiences
        WHERE action != 'HOLD'
        GROUP BY symbol
        ORDER BY total_gross DESC
    """)
    for r in cur.fetchall():
        sym, cnt, gross, fees = r
        gross = gross or 0.0
        fees = fees or 0.0
        print(f" {sym:8s} | Trades: {cnt:5d} | Gross: ${gross:+9.2f} | Fees: ${fees:8.2f} | Net: ${gross - fees:+9.2f}")

conn.close()
