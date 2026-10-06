from sqlalchemy import create_engine
import json

def verify_fixes():
    # Load database URL from .env or construct it
    db_url = "mysql+pymysql://root:@localhost:3306/ai_trading"
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Query the last 15 experiences
            result = conn.execute("SELECT id, action, exit_price, fees, holding_duration, position_after, leverage, margin, portfolio_state, action_probability FROM experiences ORDER BY id DESC LIMIT 15")
            rows = result.fetchall()
            
            if not rows:
                print("No data found in the database.")
                return
                
            print("--- VERIFYING BUG FIXES ---\n")
            all_good = True
            close_tested = False
            
            # Map column names (SQLAlchemy returns tuples)
            columns = result.keys()
            
            for r in rows:
                row = dict(zip(columns, r))
                action = row['action']
                print(f"Checking Trade ID: {row['id']} | Action: {action}")
                
                # Check Universal Fixes
                if row['position_after'] is None:
                    print("  [ERROR] position_after is NULL!")
                    all_good = False
                if row['leverage'] is None:
                    print("  [ERROR] leverage is NULL!")
                    all_good = False
                if row['margin'] is None:
                    print("  [ERROR] margin is NULL!")
                    all_good = False
                if row['portfolio_state'] is None:
                    print("  [ERROR] portfolio_state is NULL!")
                    all_good = False
                if row['action_probability'] is None:
                    print("  [ERROR] action_probability is NULL!")
                    all_good = False
                    
                # Check CLOSE fixes
                if action and "CLOSE" in action:
                    close_tested = True
                    if row['exit_price'] is None:
                        print("  [ERROR] exit_price is NULL on a CLOSE action!")
                        all_good = False
                    if row['fees'] is None:
                        print("  [ERROR] fees is NULL on a CLOSE action!")
                        all_good = False
                    if row['holding_duration'] is None:
                        print("  [ERROR] holding_duration is NULL on a CLOSE action!")
                        all_good = False
                        
                print("  [OK] Data looks complete.")
                print("-" * 40)
                
            print("\n--- FINAL VERIFICATION RESULT ---")
            if all_good:
                print("✅ SUCCESS: All new data points are saving correctly!")
                if not close_tested:
                    print("   (Note: The bot hasn't closed a trade yet, so I couldn't test the 'exit_price' fix. Check again later!)")
            else:
                print("❌ ERROR: Some columns are still showing up as NULL. The fix may not have applied correctly.")
                
    except Exception as e:
        print(f"Database Connection Error: {e}")

if __name__ == "__main__":
    verify_fixes()
