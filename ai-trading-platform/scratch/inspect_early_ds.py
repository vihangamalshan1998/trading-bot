import pymysql, json

conn = pymysql.connect(host='localhost', user='root', password='', database='ai_trading', cursorclass=pymysql.cursors.DictCursor)
with conn.cursor() as cur:
    cur.execute('''
        SELECT id, symbol, action, entry_price, realized_pnl, derivatives_state
        FROM experiences
        WHERE id < 25174 AND (realized_pnl IS NOT NULL AND realized_pnl != 0) AND derivatives_state IS NOT NULL
        LIMIT 10
    ''')
    for r in cur.fetchall():
        ds = r['derivatives_state']
        if isinstance(ds, str): ds = json.loads(ds)
        print(f"id: {r['id']} | {r['symbol']} | {r['action']} | entry: {r['entry_price']} | pnl: {r['realized_pnl']} | ds: {ds}")
conn.close()
