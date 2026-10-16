import os
import sys
import time

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pymysql
from core.config.settings import settings

def prune_vps_database():
    """
    Cleans out experiences from the VPS MySQL database to reclaim disk space.
    Since all training data is stored locally on Laragon PC via sync_db.py,
    the VPS only needs to act as a lightweight trading engine.

    Usage:
        python scripts/maintenance/prune_excess_holds.py        -> Deletes ALL 'HOLD' records (frees ~95% disk)
        python scripts/maintenance/prune_excess_holds.py --all  -> Deletes ALL experiences & old features (frees 100% disk)
    """
    delete_all = "--all" in sys.argv or "-a" in sys.argv or "all" in sys.argv

    print("==================================================================")
    print("           VPS DATABASE DISK SPACE RECLAMATION TOOL               ")
    print("==================================================================")
    mode_text = "DELETE ALL EXPERIENCES (Full VPS Purge)" if delete_all else "DELETE ALL HOLD RECORDS"
    print(f"Mode: {mode_text}")
    print(f"Connecting to MySQL database '{settings.mysql_database}' at {settings.mysql_host}:{settings.mysql_port}...\n")

    try:
        conn = pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with conn.cursor() as cur:
            # 1. Experiences Table Check
            cur.execute("SELECT COUNT(*) as total, SUM(CASE WHEN action = 'HOLD' THEN 1 ELSE 0 END) as holds FROM experiences")
            stats = cur.fetchone() or {"total": 0, "holds": 0}
            total_exp = stats["total"] or 0
            total_holds = stats["holds"] or 0
            
            print(f"📊 Current Experiences Table Status:")
            print(f"   - Total Records: {total_exp:,}")
            print(f"   - HOLD Records:  {total_holds:,}")
            print(f"   - Trade Records: {total_exp - total_holds:,}\n")
            
            if delete_all:
                print("🗑️ Executing Full Purge: Deleting ALL experiences from VPS...")
                # Note: We use DELETE instead of TRUNCATE to preserve AUTO_INCREMENT
                # so that sync_db.py (WHERE id > max_id) on your local laptop continues working smoothly!
                cur.execute("DELETE FROM experiences")
                conn.commit()
                print(f"   ✅ Successfully deleted {cur.rowcount:,} experiences!")
            else:
                if total_holds > 0:
                    print("🗑️ Executing HOLD Purge: Deleting ALL HOLD records from VPS...")
                    cur.execute("DELETE FROM experiences WHERE action = 'HOLD'")
                    conn.commit()
                    print(f"   ✅ Successfully deleted {cur.rowcount:,} HOLD records!")
                else:
                    print("   ℹ️ No HOLD records to delete.")

            # 2. Market Features Cleanup (Keep only last 24h)
            try:
                cutoff_feature_ts = int((time.time() - (24 * 3600)) * 1000)
                cur.execute("DELETE FROM market_features WHERE timestamp < %s", (cutoff_feature_ts,))
                conn.commit()
                print(f"   ✅ Pruned {cur.rowcount:,} old market feature snapshots (older than 24h).")
            except Exception as feat_err:
                pass # Table might not exist or be empty

            # 3. Shrink physical files on disk via OPTIMIZE TABLE
            print("\n⚙️ Running OPTIMIZE TABLE to reclaim physical disk space back to Linux OS...")
            cur.execute("OPTIMIZE TABLE experiences")
            try:
                cur.execute("OPTIMIZE TABLE market_features")
            except Exception:
                pass
            print("   ✅ Tables optimized successfully! Disk space returned to OS.")

        conn.close()
        print("\n🎉 VPS Database Pruning Complete! Storage has been freed.")
        print("==================================================================")
    except Exception as e:
        print(f"❌ Pruning failed with error: {e}")

if __name__ == "__main__":
    prune_vps_database()
