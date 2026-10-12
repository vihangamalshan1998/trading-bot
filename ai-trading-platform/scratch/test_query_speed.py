import pymysql
import time

conn = pymysql.connect(host='127.0.0.1', user='root', password='', database='ai_trading')
cursor = conn.cursor()

cursor.execute("SELECT id FROM experiences ORDER BY id DESC LIMIT 500")
ids = [str(r[0]) for r in cursor.fetchall()]
ids_str = ",".join(ids)

t0 = time.time()
cursor.execute(f"SELECT id, symbol, reward FROM experiences WHERE id IN ({ids_str}) ORDER BY id DESC")
res1 = cursor.fetchall()
t1 = time.time()
print(f"ORDER BY id DESC: {len(res1)} rows in {t1 - t0:.4f} seconds")

conn.close()
