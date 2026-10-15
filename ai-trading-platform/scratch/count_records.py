import sqlite3
conn = sqlite3.connect("trading_bot.db")
c = conn.cursor()
c.execute("SELECT COUNT(1) FROM experience_replay")
print("Total records:", c.fetchone()[0])
c.execute("SELECT MIN(created_at), MAX(created_at) FROM experience_replay")
r = c.fetchone()
print(f"Date range: {r[0]} to {r[1]}")
conn.close()
