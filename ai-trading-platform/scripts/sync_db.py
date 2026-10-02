import os
import time
import pymysql
from dotenv import load_dotenv
from core.logging.logger import logger

load_dotenv()

# The bat file will automatically tunnel the VPS database to localhost:3307
LOCAL_USER = os.getenv("MYSQL_USER", "root")
LOCAL_PASS = os.getenv("MYSQL_PASSWORD", "")
LOCAL_DB = os.getenv("MYSQL_DATABASE", "ai_trading")

# Remote DB (VPS) Credentials
REMOTE_USER = os.getenv("VPS_MYSQL_USER", LOCAL_USER)
REMOTE_PASS = os.getenv("VPS_MYSQL_PASSWORD", "")  # <--- We will add this to your .env
REMOTE_DB = os.getenv("VPS_MYSQL_DATABASE", LOCAL_DB)

def sync_experiences():
    logger.info("-----------------------------------")
    logger.info("Connecting to databases to sync new live trades...")
    try:
        # Local Laragon Database
        local_conn = pymysql.connect(host="127.0.0.1", port=3306, user=LOCAL_USER, password=LOCAL_PASS, database=LOCAL_DB, read_timeout=30, connect_timeout=10)
    except Exception as e:
        logger.error(f"LOCAL Database Connection error: {e}")
        logger.error("Make sure Laragon MySQL is running!")
        return

    try:
        # Remote VPS Database (via SSH Tunnel on port 3307)
        remote_conn = pymysql.connect(host="127.0.0.1", port=3307, user=REMOTE_USER, password=REMOTE_PASS, database=REMOTE_DB, read_timeout=30, connect_timeout=10)
    except Exception as e:
        logger.error(f"REMOTE VPS Database Connection error: {e}")
        logger.error("Make sure the SSH tunnel is open and the VPS database password matches!")
        return

    with local_conn.cursor() as local_cursor, remote_conn.cursor(pymysql.cursors.DictCursor) as remote_cursor:
        while True:
            # Get the highest ID we currently have on the laptop
            local_cursor.execute("SELECT MAX(id) FROM experiences")
            max_id = local_cursor.fetchone()[0] or 0
            
            try:
                logger.info(f"Asking VPS for up to 10 records after ID {max_id}...")
                remote_cursor.execute("SELECT * FROM experiences WHERE id > %s ORDER BY id ASC LIMIT 10", (max_id,))
                new_rows = remote_cursor.fetchall()
            except Exception as e:
                logger.error(f"FATAL FETCH ERROR: {e}")
                raise
            
            if not new_rows:
                logger.info("Data is perfectly in sync! No new trades to download.")
                break # BREAK OUT OF THE LOOP AND GO TO PHASE 2!
                
            # Dynamically build the insert query based on the exact columns returned from the VPS
            columns = list(new_rows[0].keys())
            cols_str = ", ".join(columns)
            vals_str = ", ".join([f"%({col})s" for col in columns])
            insert_query = f"INSERT INTO experiences ({cols_str}) VALUES ({vals_str})"
            
            # Insert them into the local Laragon database
            try:
                local_cursor.executemany(insert_query, new_rows)
                local_conn.commit()
                logger.info(f"Successfully synced {len(new_rows)} new rows to the laptop! Checking for more...")
            except Exception as e:
                logger.error(f"Failed to insert VPS rows into Laragon: {e}")
                local_conn.rollback()
                return
                
        # Phase 2: Sync Delayed Rewards!
        # The VPS updates `reward_4h` hours after the trade. We need to pull those updates down.
        local_cursor.execute("SELECT id FROM experiences WHERE reward IS NULL OR reward_4h IS NULL")
        missing_rewards = local_cursor.fetchall()
        
        if missing_rewards:
            missing_ids = [row[0] for row in missing_rewards]
            # Chunk the IDs to prevent massive queries
            chunk_size = 100
            updated_count = 0
            
            for i in range(0, len(missing_ids), chunk_size):
                chunk_ids = missing_ids[i:i+chunk_size]
                format_strings = ','.join(['%s'] * len(chunk_ids))
                
                # Ask VPS if it has calculated the rewards for these rows yet
                remote_cursor.execute(f"SELECT id, reward, reward_5m, reward_1h, reward_4h FROM experiences WHERE id IN ({format_strings}) AND (reward IS NOT NULL OR reward_4h IS NOT NULL)", tuple(chunk_ids))
                vps_updates = remote_cursor.fetchall()
                
                for vps_row in vps_updates:
                    local_cursor.execute("""
                        UPDATE experiences 
                        SET reward = %s, reward_5m = %s, reward_1h = %s, reward_4h = %s 
                        WHERE id = %s
                    """, (vps_row['reward'], vps_row['reward_5m'], vps_row['reward_1h'], vps_row['reward_4h'], vps_row['id']))
                    updated_count += 1
            
            if updated_count > 0:
                local_conn.commit()
                logger.info(f"Successfully synced {updated_count} delayed rewards from VPS to Laptop!")

    local_conn.close()
    remote_conn.close()

if __name__ == "__main__":
    print("===================================")
    print("AI Database Auto-Syncer Started!")
    print("===================================")
    while True:
        try:
            sync_experiences()
        except Exception as e:
            print(f"Sync failed: {e}")
            
        print("Sleeping for 5 minutes before checking for new trades...")
        time.sleep(300)
