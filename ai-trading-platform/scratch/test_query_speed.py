import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', password='', database='ai_trading')
cursor = conn.cursor()

# Get 50 recent IDs
cursor.execute("SELECT id FROM experiences ORDER BY id DESC LIMIT 50")
ids = [str(r[0]) for r in cursor.fetchall()]
ids_str = ",".join(ids)

cursor.execute(f"EXPLAIN SELECT * FROM experiences WHERE id IN ({ids_str}) ORDER BY timestamp DESC")
print("EXPLAIN WITH ORDER BY timestamp DESC:")
for row in cursor.fetchall():
    print(row)

cursor.execute(f"EXPLAIN SELECT * FROM experiences WHERE id IN ({ids_str}) ORDER BY id DESC")
print("\nEXPLAIN WITH ORDER BY id DESC:")
for row in cursor.fetchall():
    print(row)

conn.close()
