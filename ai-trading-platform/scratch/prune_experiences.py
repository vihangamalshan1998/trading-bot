import pymysql
import time
from datetime import datetime, timedelta

# ==========================================
# --- CONFIGURATION ---
# ==========================================
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "ai_trading"

DAYS_TO_KEEP = 30
# ==========================================

def prune_old_experiences():
    print(f"Connecting to {DB_NAME} database...")
    try:
        connection = pymysql.connect(
            host=DB_HOST, 
            user=DB_USER, 
            password=DB_PASSWORD, 
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection.cursor() as cursor:
            cutoff_date = datetime.now() - timedelta(days=DAYS_TO_KEEP)
            cutoff_ts_ms = int(cutoff_date.timestamp() * 1000)
            
            print(f"Targeting normal trades older than {DAYS_TO_KEEP} days (Before {cutoff_date.strftime('%Y-%m-%d')})...")
            
            # Delete old trades from `experiences`
            # SAFETY NET: We do NOT delete anything from the `golden_data` table!
            delete_query = """
                DELETE FROM experiences 
                WHERE timestamp < %s;
            """
            
            print(f"Executing massive DELETE query to free up disk space...")
            start_time = time.time()
            cursor.execute(delete_query, (cutoff_ts_ms,))
            connection.commit()
            
            rows_deleted = cursor.rowcount
            elapsed = time.time() - start_time
            
            print("\n=================================")
            print("        PRUNING SUMMARY          ")
            print("=================================")
            print(f"Execution Time: {elapsed:.2f} seconds")
            print(f"Old Trades Deleted: {rows_deleted}")
            print("=================================")
            print("\n[SUCCESS] Your active database is now incredibly fast and clean!")
            print("The AI will now focus ONLY on the current 30-day market regime + your permanent Golden Batch.")

    except Exception as e:
        print(f"Error: {e}")
        connection.rollback()
    finally:
        if 'connection' in locals() and connection.open:
            connection.close()
            print("Database connection closed.")

if __name__ == "__main__":
    prune_old_experiences()
