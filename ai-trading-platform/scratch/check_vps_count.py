import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

LOCAL_USER = os.getenv("MYSQL_USER", "root")
REMOTE_USER = os.getenv("VPS_MYSQL_USER", LOCAL_USER)
REMOTE_PASS = os.getenv("VPS_MYSQL_PASSWORD", "")
REMOTE_DB = os.getenv("VPS_MYSQL_DATABASE", "ai_trading")

print("Checking VPS database for any hidden events...")
try:
    remote_conn = pymysql.connect(host="127.0.0.1", port=3307, user=REMOTE_USER, password=REMOTE_PASS, database=REMOTE_DB)
    with remote_conn.cursor() as cursor:
        cursor.execute("SHOW EVENTS;")
        events = cursor.fetchall()
        
        if len(events) == 0:
            print("[CONFIRMED] 0 events found. The database is completely clean!")
        else:
            print(f"[WARNING] Found {len(events)} events:")
            for e in events:
                print(e)
except Exception as e:
    print(f"Error: {e}")
