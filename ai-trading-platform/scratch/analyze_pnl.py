import sqlite3

def analyze_pnl():
    db_path = "data/trading_data.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query the last 1000 trades that have a calculated reward
        query = """
        SELECT symbol, action, confidence, price, reward 
        FROM trading_experience 
        WHERE reward IS NOT NULL
        ORDER BY id DESC 
        LIMIT 1000
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            print("No completed trades with rewards found in the local database.")
            return

        total_records = len(rows)
        wins = [r for r in rows if r[4] > 0]
        losses = [r for r in rows if r[4] < 0]
        
        print(f"--- RECENT PNL ANALYSIS (Last 1000 records) ---")
        print(f"Total Evaluated Records: {total_records}")
        print(f"Winning Records: {len(wins)} ({len(wins)/total_records*100:.1f}%)")
        print(f"Losing Records: {len(losses)} ({len(losses)/total_records*100:.1f}%)")
        
        if losses:
            print("\n--- DEEP DIVE ON LOSING TRADES ---")
            avg_loss = sum(r[4] for r in losses) / len(losses)
            print(f"Average loss penalty: {avg_loss:.2f}")
            
            # Action distribution on losses
            action_counts = {}
            for r in losses:
                action_counts[r[1]] = action_counts.get(r[1], 0) + 1
                
            print("\nWhat actions cause the most losses?")
            for action, count in action_counts.items():
                print(f"Action {action}: {count} times ({count/len(losses)*100:.1f}%)")
                
            # Confidence on losses
            avg_conf = sum(r[2] for r in losses) / len(losses)
            print(f"\nAverage Bot Confidence during losses: {avg_conf:.1f}%")
            
            if avg_conf < 40:
                print("CONCLUSION: The bot is losing because it is EXPLORING (Low Confidence). It is intentionally guessing.")
            else:
                print("CONCLUSION: The bot is highly confident but still losing. The Reward Function or timeframe might be misaligned.")
                
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_pnl()
