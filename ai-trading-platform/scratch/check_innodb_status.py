import pymysql

conn = pymysql.connect(host='127.0.0.1', user='root', password='', database='ai_trading')
cursor = conn.cursor()
cursor.execute("SHOW CREATE TABLE experiences")
print(cursor.fetchone()[1])

cursor.execute("SHOW ENGINE INNODB STATUS")
status = cursor.fetchone()[2]
# print the TRANSACTIONS and WAITING locks part
print("\n=== INNODB STATUS LOCKS ===")
for line in status.split("\n"):
    if "TRANSACTIONS" in line or "ACTIVE" in line or "lock struct" in line or "WAITING" in line or "RECORD LOCKS" in line:
        print(line)

conn.close()
