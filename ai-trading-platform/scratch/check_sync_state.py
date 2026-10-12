import pymysql

try:
    conn = pymysql.connect(host='127.0.0.1', user='root', password='', database='ai_trading')
    cursor = conn.cursor()
    cursor.execute('SHOW PROCESSLIST')
    processes = cursor.fetchall()
    print("=== MySQL PROCESSLIST ===")
    for p in processes:
        print(p)
        
    cursor.execute('SELECT MAX(id), COUNT(*) FROM experiences')
    max_id, total = cursor.fetchone()
    print(f"Local experiences: total={total}, max_id={max_id}")
    
    cursor.execute('SELECT COUNT(*) FROM experiences WHERE reward IS NULL OR reward_4h IS NULL')
    missing_count = cursor.fetchone()[0]
    print(f"Missing rewards count: {missing_count}")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
