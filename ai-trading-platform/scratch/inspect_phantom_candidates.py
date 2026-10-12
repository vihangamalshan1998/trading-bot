import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='',
    database='ai_trading',
    cursorclass=pymysql.cursors.DictCursor
)

with conn.cursor() as cur:
    cur.execute('''
        SELECT id, timestamp, symbol, action, entry_price, exit_price, realized_pnl, fees, position_before, position_after
        FROM experiences
        WHERE id IN (1, 2, 3, 4, 5, 14, 15, 16, 20, 21, 22)
        ORDER BY id ASC
    ''')
    for r in cur.fetchall():
        print(r)

    print("\n--- LATEST SAMPLES (47735 - 47750) ---")
    cur.execute('''
        SELECT id, timestamp, symbol, action, entry_price, exit_price, realized_pnl, fees, position_before, position_after
        FROM experiences
        WHERE id >= 47735
        ORDER BY id ASC
    ''')
    for r in cur.fetchall():
        print(r)

conn.close()
