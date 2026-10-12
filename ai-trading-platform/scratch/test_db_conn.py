import pymysql

for host in ['127.0.0.1', 'localhost', '::1']:
    try:
        conn = pymysql.connect(host=host, port=3306, user='root', password='', database='ai_trading')
        print(f"SUCCESS connecting with host={host}")
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM experiences")
            print("Total experiences:", cur.fetchone())
        conn.close()
        break
    except Exception as e:
        print(f"Failed host={host}: {e}")
