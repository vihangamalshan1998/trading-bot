import pymysql
import json
from collections import defaultdict

DB_HOST = "127.0.0.1"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "ai_trading"

def analyze_golden():
    print("Connecting to Database to analyze golden_experiences...")
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

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT id, experience_id, symbol, action, realized_pnl, margin, 
                   market_state, derivatives_state
            FROM golden_experiences 
        """)
        trades = cursor.fetchall()

    if not trades:
        print("No trades found in golden_experiences.")
        return

    total = len(trades)
    print(f"\nLoaded {total} trades from the Golden Vault.\n")

    # Metrics
    null_states = 0
    small_trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    symbol_counts = defaultdict(int)
    action_counts = defaultdict(int)
    unique_exp_ids = set()
    duplicates = 0

    for t in trades:
        # Check duplicates
        eid = t['experience_id']
        if eid in unique_exp_ids:
            duplicates += 1
        else:
            unique_exp_ids.add(eid)

        # Check missing states (corrupted data)
        if not t['market_state'] or not t['derivatives_state']:
            null_states += 1

        pnl = t['realized_pnl'] or 0.0
        
        # Check rule violation (pnl between -50 and 50)
        if -50 <= pnl <= 50:
            small_trades += 1

        if pnl > 0:
            wins += 1
            total_profit += pnl
        elif pnl < 0:
            losses += 1
            total_loss += pnl

        symbol_counts[t['symbol']] += 1
        action_counts[t['action']] += 1

    print("=========================================")
    print("      GOLDEN BATCH QUALITY REPORT        ")
    print("=========================================")
    print(f"Total Golden Trades: {total}")
    print(f"Duplicates: {duplicates} (Should be 0)")
    print(f"Corrupted/Missing States: {null_states} (Should be 0)")
    print(f"Rule Violations (-50 to +50 PnL): {small_trades} (Should be 0)")
    print("-----------------------------------------")
    print(f"Massive Winners (> +50): {wins} trades (Total: +${total_profit:,.2f})")
    print(f"Massive Losers (< -50): {losses} trades (Total: -${abs(total_loss):,.2f})")
    print("-----------------------------------------")
    print("Distribution by Symbol:")
    for sym, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {sym}: {count} trades")
    print("-----------------------------------------")
    print("Distribution by Action:")
    for act, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {act}: {count} trades")
    print("=========================================\n")
    
    if duplicates == 0 and null_states == 0 and small_trades == 0:
        print("[RESULT] 100% PERFECT DATA. The Golden Vault is flawlessly clean.")
    else:
        print("[RESULT] WARNING: Unnecessary or corrupted data found. Review metrics above.")

if __name__ == "__main__":
    analyze_golden()
