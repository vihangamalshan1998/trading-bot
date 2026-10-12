import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='ai_trading', cursorclass=pymysql.cursors.DictCursor)
with conn.cursor() as cur:
    # 1. Experiences check
    cur.execute('''
        SELECT 
            COUNT(*) as total_records,
            SUM(CASE WHEN realized_pnl IS NOT NULL AND realized_pnl != 0 THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN (realized_pnl IS NOT NULL AND realized_pnl != 0) AND (fees IS NULL OR fees = 0) THEN 1 ELSE 0 END) as missing_fees_closed,
            SUM(fees) as total_fees_paid,
            SUM(realized_pnl) as net_realized_pnl
        FROM experiences
    ''')
    print('EXPERIENCES SUMMARY:', cur.fetchone())

    # 2. Golden experiences check
    cur.execute('''
        SELECT 
            COUNT(*) as total_golden,
            SUM(CASE WHEN fees IS NULL OR fees = 0 THEN 1 ELSE 0 END) as missing_fees_golden,
            SUM(fees) as golden_fees,
            SUM(realized_pnl) as golden_net_pnl
        FROM golden_experiences
    ''')
    print('GOLDEN VAULT SUMMARY:', cur.fetchone())

    # 3. Day by day summary
    cur.execute('''
        SELECT 
            DATE(FROM_UNIXTIME(timestamp)) as dt,
            COUNT(*) as total_records,
            SUM(CASE WHEN realized_pnl IS NOT NULL AND realized_pnl != 0 THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) / SUM(CASE WHEN realized_pnl IS NOT NULL AND realized_pnl != 0 THEN 1 ELSE 0 END) * 100, 1) as win_rate_pct,
            ROUND(SUM(fees), 2) as fees_paid,
            ROUND(SUM(realized_pnl), 2) as net_realized_pnl
        FROM experiences
        GROUP BY DATE(FROM_UNIXTIME(timestamp))
        ORDER BY dt ASC
    ''')
    print('\nDAY BY DAY AUDIT:')
    for r in cur.fetchall():
        print(f"{r['dt']} | Rec: {r['total_records']} | Closed: {r['closed_trades']} | W: {r['wins']} | L: {r['losses']} | WR: {r['win_rate_pct']}% | Fees: ${r['fees_paid']} | Net: ${r['net_realized_pnl']}")
conn.close()
