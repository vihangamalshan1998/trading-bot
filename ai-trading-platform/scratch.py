import pymysql
import json
import statistics

def analyze_db():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='', database='ai_trading', cursorclass=pymysql.cursors.DictCursor)
    
    with conn.cursor() as cursor:
        # 1. Analyze the general experiences
        cursor.execute("SELECT COUNT(*) as count, SUM(realized_pnl) as total_pnl, AVG(realized_pnl) as avg_pnl FROM experiences WHERE realized_pnl IS NOT NULL AND realized_pnl != 0")
        exp_stats = cursor.fetchone()
        
        # 2. Analyze the golden experiences
        cursor.execute("SELECT COUNT(*) as count, SUM(realized_pnl) as total_pnl, AVG(realized_pnl) as avg_pnl, MAX(realized_pnl) as max_pnl, MIN(realized_pnl) as min_pnl FROM golden_experiences WHERE realized_pnl IS NOT NULL")
        golden_stats = cursor.fetchone()
        
        # 3. Analyze recent performance (last 1000 trades)
        cursor.execute("SELECT realized_pnl FROM experiences WHERE realized_pnl IS NOT NULL AND realized_pnl != 0 ORDER BY timestamp DESC LIMIT 1000")
        recent_trades = [r['realized_pnl'] for r in cursor.fetchall()]
        
        if recent_trades:
            wins = len([t for t in recent_trades if t > 0])
            losses = len([t for t in recent_trades if t < 0])
            win_rate = (wins / len(recent_trades)) * 100
            recent_pnl = sum(recent_trades)
        else:
            win_rate = 0
            recent_pnl = 0

        print(f"=== DB ANALYSIS ===")
        print(f"Total Completed Trades: {exp_stats['count']}")
        print(f"All-Time Net PnL (including fees): ${exp_stats['total_pnl'] or 0:.2f}")
        
        print(f"\n=== GOLDEN RECORDS (The Proof of Edge) ===")
        print(f"Total Golden Trades Found: {golden_stats['count']}")
        print(f"Average Golden Trade PnL: ${golden_stats['avg_pnl'] or 0:.2f}")
        print(f"Biggest Single Win: ${golden_stats['max_pnl'] or 0:.2f}")
        
        print(f"\n=== RECENT PERFORMANCE (Last 1000 Trades) ===")
        print(f"Recent Win Rate: {win_rate:.2f}%")
        print(f"Recent Net PnL: ${recent_pnl:.2f}")

analyze_db()
