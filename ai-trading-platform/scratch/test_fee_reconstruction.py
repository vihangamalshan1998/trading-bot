import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='ai_trading', cursorclass=pymysql.cursors.DictCursor)
with conn.cursor() as cur:
    # 1. Check fee distribution in modern closed trades
    cur.execute('''
        SELECT 
            symbol,
            COUNT(*) as count,
            AVG(fees) as avg_fee,
            AVG(margin) as avg_margin,
            AVG(leverage) as avg_leverage,
            AVG(fees / (margin * leverage)) as avg_fee_rate,
            AVG(ABS(realized_pnl)) as avg_abs_pnl,
            AVG(fees / ABS(realized_pnl)) as fee_to_abs_pnl
        FROM experiences
        WHERE id >= 25174 AND (realized_pnl IS NOT NULL AND realized_pnl != 0) AND fees > 0
        GROUP BY symbol
    ''')
    print("--- MODERN CLOSED TRADES BY SYMBOL ---")
    for r in cur.fetchall():
        print(r)

    # 2. Check early closed trades by symbol
    cur.execute('''
        SELECT 
            symbol,
            COUNT(*) as count,
            AVG(ABS(realized_pnl)) as avg_abs_pnl,
            SUM(realized_pnl) as total_pnl
        FROM experiences
        WHERE id < 25174 AND (realized_pnl IS NOT NULL AND realized_pnl != 0)
        GROUP BY symbol
    ''')
    print("\n--- EARLY CLOSED TRADES BY SYMBOL (MISSING FEES) ---")
    for r in cur.fetchall():
        print(r)

conn.close()
