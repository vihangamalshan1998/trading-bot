import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

REMOTE_USER = os.getenv("VPS_MYSQL_USER", "root")
REMOTE_PASS = os.getenv("VPS_MYSQL_PASSWORD", "")
REMOTE_DB = os.getenv("VPS_MYSQL_DATABASE", "ai_trading")

conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user=REMOTE_USER,
    password=REMOTE_PASS,
    database=REMOTE_DB,
    cursorclass=pymysql.cursors.DictCursor
)

with conn.cursor() as cur:
    cur.execute("DESCRIBE experiences")
    cols = [c['Field'] for c in cur.fetchall()]
    print("COLUMNS IN experiences:", cols)
    
    # Query with available columns
    pnl_col = 'realized_pnl' if 'realized_pnl' in cols else ('pnl' if 'pnl' in cols else None)
    fee_col = 'fees' if 'fees' in cols else ('fee' if 'fee' in cols else None)
    time_col = 'timestamp' if 'timestamp' in cols else ('created_at' if 'created_at' in cols else None)
    
    print(f"Detected cols: pnl={pnl_col}, fee={fee_col}, time={time_col}")
    
    cur.execute(f'''
        SELECT id, {time_col}, symbol, action, {pnl_col}, {fee_col}
        FROM experiences
        ORDER BY id DESC
        LIMIT 20
    ''')
    for r in cur.fetchall():
        print(r)

