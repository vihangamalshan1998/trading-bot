import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='ai_trading', cursorclass=pymysql.cursors.DictCursor)
with conn.cursor() as cur:
    cur.execute('''
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN realized_pnl IS NOT NULL AND realized_pnl != 0 THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN (realized_pnl IS NOT NULL AND realized_pnl != 0) AND (fees IS NULL OR fees = 0) THEN 1 ELSE 0 END) as closed_missing_fees,
            SUM(CASE WHEN margin IS NULL OR margin = 0 THEN 1 ELSE 0 END) as missing_margin,
            SUM(CASE WHEN leverage IS NULL OR leverage = 0 THEN 1 ELSE 0 END) as missing_leverage,
            SUM(CASE WHEN entry_price IS NULL OR entry_price = 0 THEN 1 ELSE 0 END) as missing_entry_px,
            SUM(CASE WHEN exit_price IS NULL OR exit_price = 0 THEN 1 ELSE 0 END) as missing_exit_px,
            SUM(CASE WHEN holding_duration IS NULL THEN 1 ELSE 0 END) as missing_holding_duration,
            SUM(CASE WHEN slippage IS NULL THEN 1 ELSE 0 END) as missing_slippage,
            SUM(CASE WHEN funding_cost IS NULL THEN 1 ELSE 0 END) as missing_funding_cost
        FROM experiences
    ''')
    res = cur.fetchone()
    print("--- MISSING COLUMNS AUDIT IN EXPERIENCES ---")
    for k, v in res.items():
        print(f"{k}: {v}")

    # Inspect a few closed rows with missing fees
    cur.execute('''
        SELECT id, symbol, action, entry_price, exit_price, leverage, margin, realized_pnl, fees, holding_duration, FROM_UNIXTIME(timestamp) as t
        FROM experiences
        WHERE (realized_pnl IS NOT NULL AND realized_pnl != 0) AND (fees IS NULL OR fees = 0)
        LIMIT 5
    ''')
    print("\n--- SAMPLE ROWS WITH MISSING FEES ---")
    for row in cur.fetchall():
        print(row)

    # Compare with closed rows with populated fees
    cur.execute('''
        SELECT id, symbol, action, entry_price, exit_price, leverage, margin, realized_pnl, fees, holding_duration, FROM_UNIXTIME(timestamp) as t
        FROM experiences
        WHERE (realized_pnl IS NOT NULL AND realized_pnl != 0) AND fees > 0
        LIMIT 5
    ''')
    print("\n--- SAMPLE ROWS WITH POPULATED FEES ---")
    for row in cur.fetchall():
        notional = (row['margin'] or 0) * (row['leverage'] or 1)
        ratio = (row['fees'] / notional) if notional > 0 else 0
        print(f"{row} | Notional: {notional} | Fee/Notional: {ratio:.5f}")

conn.close()
