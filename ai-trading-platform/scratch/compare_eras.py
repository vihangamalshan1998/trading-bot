import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='ai_trading', cursorclass=pymysql.cursors.DictCursor)
with conn.cursor() as cur:
    # 1. Early Golden Era (Sep 29 - Oct 04)
    cur.execute('''
        SELECT 
            COUNT(*) as total_closed,
            AVG(holding_duration) as avg_hold_sec,
            AVG(margin) as avg_margin,
            AVG(fees) as avg_fee,
            AVG(CASE WHEN (realized_pnl + fees) > 0 THEN (realized_pnl + fees) ELSE NULL END) as avg_gross_win,
            AVG(CASE WHEN (realized_pnl + fees) < 0 THEN (realized_pnl + fees) ELSE NULL END) as avg_gross_loss,
            SUM(realized_pnl + fees) as total_gross_pnl,
            SUM(fees) as total_fees,
            SUM(realized_pnl) as net_pnl
        FROM experiences
        WHERE DATE(FROM_UNIXTIME(timestamp)) BETWEEN '2026-09-29' AND '2026-10-04'
          AND realized_pnl IS NOT NULL AND realized_pnl != 0
    ''')
    early = cur.fetchone()

    # 2. Today (Oct 12)
    cur.execute('''
        SELECT 
            COUNT(*) as total_closed,
            AVG(holding_duration) as avg_hold_sec,
            AVG(margin) as avg_margin,
            AVG(fees) as avg_fee,
            AVG(CASE WHEN (realized_pnl + fees) > 0 THEN (realized_pnl + fees) ELSE NULL END) as avg_gross_win,
            AVG(CASE WHEN (realized_pnl + fees) < 0 THEN (realized_pnl + fees) ELSE NULL END) as avg_gross_loss,
            SUM(realized_pnl + fees) as total_gross_pnl,
            SUM(fees) as total_fees,
            SUM(realized_pnl) as net_pnl
        FROM experiences
        WHERE DATE(FROM_UNIXTIME(timestamp)) = '2026-10-12'
          AND realized_pnl IS NOT NULL AND realized_pnl != 0
    ''')
    today = cur.fetchone()

    print("--- COMPARISON: GOLDEN ERA (SEP 29 - OCT 04) VS TODAY (OCT 12) ---")
    print(f"Metric                        | Golden Era (Sep 29-Oct 04) | Today (Oct 12)")
    print(f"------------------------------+----------------------------+----------------")
    print(f"Total Closed Trades           | {early['total_closed']:26d} | {today['total_closed']:14d}")
    print(f"Avg Holding Duration          | {early['avg_hold_sec'] or 0:23.1f}s | {today['avg_hold_sec'] or 0:11.1f}s")
    print(f"Avg Margin per Trade          | ${early['avg_margin']:25.2f} | ${today['avg_margin']:13.2f}")
    print(f"Avg Fee per Trade             | ${early['avg_fee']:25.2f} | ${today['avg_fee']:13.2f}")
    print(f"Avg Gross Win (When Winning)  | ${early['avg_gross_win']:25.2f} | ${today['avg_gross_win']:13.2f}")
    print(f"Avg Gross Loss (When Losing)  | ${early['avg_gross_loss']:25.2f} | ${today['avg_gross_loss']:13.2f}")
    print(f"Gross PnL (Market Direction)  | ${early['total_gross_pnl']:25.2f} | ${today['total_gross_pnl']:13.2f}")
    print(f"Total Fees Paid               | ${early['total_fees']:25.2f} | ${today['total_fees']:13.2f}")
    print(f"Net Realized PnL              | ${early['net_pnl']:25.2f} | ${today['net_pnl']:13.2f}")

conn.close()
