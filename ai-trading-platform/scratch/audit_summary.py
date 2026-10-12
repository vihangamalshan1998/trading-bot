import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='ai_trading', cursorclass=pymysql.cursors.DictCursor)
with conn.cursor() as cur:
    # 1. Overall Database Totals
    cur.execute('''
        SELECT 
            COUNT(*) as total_records,
            SUM(CASE WHEN realized_pnl IS NOT NULL AND realized_pnl != 0 THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) as gross_wins,
            SUM(CASE WHEN realized_pnl < 0 THEN realized_pnl ELSE 0 END) as gross_losses,
            SUM(fees) as total_fees,
            SUM(realized_pnl) as net_pnl,
            AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE NULL END) as avg_win,
            AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl ELSE NULL END) as avg_loss,
            MAX(realized_pnl) as max_win,
            MIN(realized_pnl) as max_loss
        FROM experiences
    ''')
    db_totals = cur.fetchone()
    print("--- LIFETIME GRAND TOTALS ---")
    for k, v in db_totals.items():
        print(f"{k}: {v}")

    # 2. Golden stats
    cur.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
            SUM(realized_pnl) as net_pnl,
            AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE NULL END) as avg_win,
            AVG(CASE WHEN realized_pnl < 0 THEN realized_pnl ELSE NULL END) as avg_loss,
            SUM(fees) as total_fees
        FROM golden_experiences
    ''')
    g_stats = cur.fetchone()
    print("\n--- GOLDEN VAULT STATS ---")
    for k, v in g_stats.items():
        print(f"{k}: {v}")
    
    # 3. Top 10 wins in golden
    cur.execute('''
        SELECT id, symbol, action, realized_pnl, fees, FROM_UNIXTIME(timestamp) as t
        FROM golden_experiences
        ORDER BY realized_pnl DESC
        LIMIT 10
    ''')
    print("\n--- TOP 10 WINS IN GOLDEN VAULT ---")
    for row in cur.fetchall():
        fee_str = f"{row['fees']:.2f}" if row['fees'] is not None else "0.00"
        print(f"{row['symbol']} (id: {row['id']}): +${row['realized_pnl']:.2f} (Fee: ${fee_str}) — {row['t']}")
        
    # 4. By coin in golden
    cur.execute('''
        SELECT symbol, COUNT(*) as trades, SUM(realized_pnl) as net_pnl
        FROM golden_experiences
        GROUP BY symbol
        ORDER BY net_pnl DESC
    ''')
    print("\n--- GOLDEN VAULT BY COIN ---")
    for row in cur.fetchall():
        print(f"{row['symbol']}: {row['trades']} trades | Net: ${row['net_pnl']:.2f}")

    # 5. Day by day summary
    cur.execute('''
        SELECT 
            DATE(FROM_UNIXTIME(timestamp)) as dt,
            COUNT(*) as total_records,
            SUM(CASE WHEN realized_pnl IS NOT NULL AND realized_pnl != 0 THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
            SUM(fees) as fees_paid,
            SUM(realized_pnl) as net_pnl
        FROM experiences
        GROUP BY DATE(FROM_UNIXTIME(timestamp))
        ORDER BY dt ASC
    ''')
    print("\n--- DAY BY DAY SUMMARY ---")
    for row in cur.fetchall():
        wr = (row['wins'] / row['closed_trades'] * 100) if row['closed_trades'] and row['closed_trades'] > 0 else 0
        fees = row['fees_paid'] if row['fees_paid'] is not None else 0
        pnl = row['net_pnl'] if row['net_pnl'] is not None else 0
        print(f"{row['dt']} | Rec: {row['total_records']} | Closed: {row['closed_trades']} | W: {row['wins']} | L: {row['losses']} | WR: {wr:.1f}% | Fees: ${fees:.2f} | Net: ${pnl:.2f}")

conn.close()
