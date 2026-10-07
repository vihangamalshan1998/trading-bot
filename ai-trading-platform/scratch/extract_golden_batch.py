import pymysql
import pandas as pd
from datetime import datetime

# ==========================================
# --- CONFIGURATION ---
# ==========================================
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "ai_trading"

# Set the time period you want to scan (in YYYY-MM-DD format)
# You can change these dates before running!
START_DATE = "2023-10-01" 
END_DATE = "2026-11-01"

# Define what an "Extreme" trade is (Using PERCENTAGES to scale perfectly!)
# 0.10 means a 10% gain on the margin risked
# -0.05 means a 5% loss on the margin risked
MIN_WIN_PCT = 0.10    
MAX_LOSS_PCT = -0.05  
# ==========================================

def extract_golden_batch():
    print(f"Connecting to {DB_NAME} database...")
    try:
        connection = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("\nNote: Make sure your venv is activated and has the required packages:")
        print("pip install pymysql pandas sqlalchemy")
        return

    # Build the query
    query = "SELECT * FROM experiences WHERE 1=1"
    params = []
    
    if START_DATE and END_DATE:
        # Convert dates to unix timestamps
        start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(END_DATE, "%Y-%m-%d").timestamp())
        
        # We check both standard timestamps (seconds) and JS timestamps (milliseconds)
        query += " AND ((timestamp >= %s AND timestamp <= %s) OR (timestamp >= %s AND timestamp <= %s))"
        params.extend([start_ts, end_ts, start_ts * 1000, end_ts * 1000])

    print(f"Scanning experiences from {START_DATE} to {END_DATE}...")
    
    try:
        df = pd.read_sql_query(query, connection, params=params)
    except Exception as e:
        print(f"Error reading from database: {e}")
        connection.close()
        return
        
    connection.close()

    total_rows = len(df)
    if total_rows == 0:
        print("No trades found in this specific time period!")
        return

    # Protect against divide-by-zero if margin is 0
    # Create a safe ROI column (realized_pnl / margin)
    df['roi_pct'] = df.apply(
        lambda row: (row['realized_pnl'] / row['margin']) if pd.notnull(row['margin']) and row['margin'] > 0 else 0, 
        axis=1
    )

    # Filter for Extreme Wins and Extreme Losses based on PERCENTAGE
    extreme_wins = df[df['roi_pct'] >= MIN_WIN_PCT]
    extreme_losses = df[df['roi_pct'] <= MAX_LOSS_PCT]
    
    # Combine them to create the Golden Batch
    golden_batch = pd.concat([extreme_wins, extreme_losses]).drop_duplicates(subset=['id'])

    print("\n=================================")
    print("        RESULTS SUMMARY          ")
    print("=================================")
    print(f"Total Trades Analyzed: {total_rows}")
    print(f"Extreme Wins (>{int(MIN_WIN_PCT*100)}% ROI): {len(extreme_wins)}")
    print(f"Extreme Losses (<{int(MAX_LOSS_PCT*100)}% ROI): {len(extreme_losses)}")
    print(f"Total Golden Batch Candidates: {len(golden_batch)}")
    print("=================================")
    
    if len(golden_batch) > 0:
        # Save to CSV
        output_file = f"data/golden_batch/golden_batch_extraction_{START_DATE}_to_{END_DATE}.csv"
        
        # Ensure the directory exists
        import os
        os.makedirs("data/golden_batch", exist_ok=True)
        
        golden_batch.to_csv(output_file, index=False)
        print(f"\n[SUCCESS] Saved {len(golden_batch)} extreme trades to:")
        print(f"-> {output_file}")
        print("\nYou can now safely delete the boring everyday data from this period!")

if __name__ == "__main__":
    extract_golden_batch()
