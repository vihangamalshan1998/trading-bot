import pymysql
conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', database='ai_trading')
cursor = conn.cursor(pymysql.cursors.DictCursor)

cursor.execute('SELECT MIN(id), MAX(id) FROM experiences;')
print("Experiences table:", cursor.fetchone())

cursor.execute('SELECT MIN(id), MAX(id) FROM golden_experiences;')
print("Golden Experiences table:", cursor.fetchone())
