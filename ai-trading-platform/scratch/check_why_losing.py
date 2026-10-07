import pymysql
import os

LOCAL_USER = os.getenv("DB_USER", "root")
LOCAL_PASS = os.getenv("DB_PASSWORD", "")
LOCAL_DB = os.getenv("DB_NAME", "ai_trading")

try:
    conn = pymysql.connect(
        host="127.0.0.1", 
        port=3306, 
        user=LOCAL_USER, 
        password=LOCAL_PASS, 
        database=LOCAL_DB,
        cursorclass=pymysql.cursors.DictCursor
    )
    
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT id, created_at, symbol, action_type, pnl, reward 
            FROM experiences 
            ORDER BY id DESC 
            LIMIT 20
        """)
        rows = cursor.fetchall()
        
        print("\n" + "="*80)
        print("🚨 LAST 20 TRADES/ACTIONS 🚨")
        print("="*80)
        print(f"{'ID':<8} | {'Date':<20} | {'Symbol':<10} | {'Action':<15} | {'PNL':<10} | {'Reward':<10}")
        print("-"*80)
        for r in rows:
            print(f"{r['id']:<8} | {str(r['created_at'])[:19]:<20} | {r['symbol']:<10} | {r['action_type']:<15} | {str(r['pnl']):<10} | {r['reward']:<10.2f}")
            
    conn.close()
        
except Exception as e:
    print(f"Error checking database: {e}")
