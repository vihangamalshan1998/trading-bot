import pymysql
from datetime import datetime

try:
    conn = pymysql.connect(host="127.0.0.1", port=3306, user="root", password="", database="ai_trading")
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, timestamp FROM experiences ORDER BY id ASC")
        rows = cursor.fetchall()
        
        missing_ranges = []
        last_id = 0
        last_ts = 0
        
        for row in rows:
            current_id = row[0]
            current_ts = row[1]
            
            if last_id != 0 and current_id != last_id + 1:
                # Gap found
                missing_ranges.append({
                    "start_id": last_id + 1,
                    "end_id": current_id - 1,
                    "count": current_id - last_id - 1,
                    "start_time": datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": datetime.fromtimestamp(current_ts).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            last_id = current_id
            last_ts = current_ts
            
        print(f"Total missing rows: {sum(r['count'] for r in missing_ranges)}")
        print("\nMissing Ranges:")
        for r in missing_ranges:
            print(f"- Missing {r['count']} rows between ID {r['start_id']} and {r['end_id']}")
            print(f"  Estimated Timeframe: {r['start_time']} to {r['end_time']}\n")
            
except Exception as e:
    print(f"Error: {e}")
