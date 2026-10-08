import pymysql
import time
from datetime import datetime

# ==========================================
# --- CONFIGURATION ---
# ==========================================
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "ai_trading"

START_DATE = "2023-10-01" 
END_DATE = "2026-11-01"

MIN_WIN_PCT = 0.20    
MAX_LOSS_PCT = -0.10  
# ==========================================

def extract_golden_batch():
    print(f"Connecting to {DB_NAME} database...")
    try:
        connection = pymysql.connect(
            host=DB_HOST, 
            user=DB_USER, 
            password=DB_PASSWORD, 
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    try:
        with connection.cursor() as cursor:
            # 1. Create the Golden Data table identical to experiences
            print("Creating 'golden_experiences' table if it doesn't exist...")
            cursor.execute("CREATE TABLE IF NOT EXISTS golden_experiences LIKE experiences;")
            connection.commit()

            start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp())
            end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%d").timestamp())

            print(f"Scanning experiences and inserting extreme trades natively inside MySQL (Ultra Fast)...")
            start_time = time.time()
            
            # Prevent Locking! The Live Bot is constantly writing to `experiences`. 
            # We must use READ UNCOMMITTED so we don't cause a lock timeout!
            cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;")
            
            # 2. Fetch ONLY the IDs first (Ultra Fast)
            print("Querying for extreme trade IDs...")
            select_ids_query = """
                SELECT id FROM experiences 
                WHERE id NOT IN (SELECT id FROM golden_experiences)
                AND ((timestamp >= %s AND timestamp <= %s) OR (timestamp >= %s AND timestamp <= %s))
                AND reward_5m IS NOT NULL /* GUARANTEE: Only extract fully graded/marked data */
                AND (
                    /* The Whales: Massive absolute dollar amounts */
                    (realized_pnl >= 150 OR realized_pnl <= -150) 
                    OR 
                    /* The Snipers: High percentage ROI with a minimum $50 floor */
                    (
                        (realized_pnl >= 50 OR realized_pnl <= -50) 
                        AND (
                            (margin > 0 AND (realized_pnl / margin) >= %s) OR 
                            (margin > 0 AND (realized_pnl / margin) <= %s) OR 
                            (margin > 0 AND (reward_5m / margin) >= %s) OR 
                            (margin > 0 AND (reward_5m / margin) <= %s)
                        )
                    )
                );
            """
            
            params = [
                start_ts, end_ts, 
                start_ts * 1000, end_ts * 1000, 
                MIN_WIN_PCT, MAX_LOSS_PCT,
                MIN_WIN_PCT, MAX_LOSS_PCT
            ]
            
            cursor.execute(select_ids_query, params)
            extreme_ids = [row['id'] for row in cursor.fetchall()]
            
            print(f"Found {len(extreme_ids)} extreme trades. Moving them one by one...")
            
            # 3. Pull each row into Python, then insert it (Bypasses MySQL Source Table Locks!)
            rows_inserted = 0
            for i, exp_id in enumerate(extreme_ids):
                # Fetch row directly into Python memory
                cursor.execute("SELECT * FROM experiences WHERE id = %s;", (exp_id,))
                row_data = cursor.fetchone()
                
                if row_data:
                    # Construct INSERT query dynamically based on the dictionary keys
                    import json
                    columns = ', '.join(row_data.keys())
                    placeholders = ', '.join(['%s'] * len(row_data))
                    
                    insert_vals = []
                    for val in row_data.values():
                        if isinstance(val, (dict, list)):
                            insert_vals.append(json.dumps(val))
                        else:
                            insert_vals.append(val)
                    
                    insert_query = f"INSERT IGNORE INTO golden_experiences ({columns}) VALUES ({placeholders});"
                    cursor.execute(insert_query, tuple(insert_vals))
                    rows_inserted += cursor.rowcount
                
                # Commit every 100 rows and print progress
                if (i + 1) % 100 == 0:
                    connection.commit()
                    print(f"Progress: {i + 1} / {len(extreme_ids)} moved...")
                    
            connection.commit()
            
            elapsed = time.time() - start_time
            
            print("\n=================================")
            print("        RESULTS SUMMARY          ")
            print("=================================")
            print(f"Execution Time: {elapsed:.2f} seconds")
            print(f"New Golden Trades Extracted: {rows_inserted}")
            print("=================================")
            
            print(f"\n[SUCCESS] Extracted new extreme trades to the 'golden_experiences' SQL table!")
            print("You can now safely run the 30-day auto-pruning script on the 'experiences' table.")

    except Exception as e:
        print(f"Error: {e}")
        connection.rollback()
    finally:
        connection.close()

if __name__ == "__main__":
    extract_golden_batch()
