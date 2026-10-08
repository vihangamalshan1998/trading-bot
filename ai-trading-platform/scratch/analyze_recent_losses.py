import pymysql
import json
from datetime import datetime
from collections import defaultdict

DB_HOST = "127.0.0.1"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "ai_trading"

def analyze():
    print("Connecting to Database...")
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
        print("Querying daily PnL summaries from the database...")
        cursor.execute("""
            SELECT 
                DATE(FROM_UNIXTIME(
                    CASE 
                        WHEN timestamp > 9999999999 THEN timestamp / 1000 
                        ELSE timestamp 
                    END
                )) as trade_date,
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                SUM(realized_pnl) as total_pnl,
                SUM(margin) as total_margin
            FROM experiences 
            GROUP BY trade_date
            ORDER BY trade_date DESC 
            LIMIT 10
        """)
        daily_stats = cursor.fetchall()
        
    print("=== DAILY TRADING SUMMARY (SQL GROUP BY) ===")
    for s in daily_stats:
        if s['trade_date'] is None:
            continue
        total_trades = s['total_trades']
        wins = s['wins'] or 0
        losses = s['losses'] or 0
        pnl = s['total_pnl'] or 0.0
        margin = s['total_margin'] or 0.0
        
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        roi = (pnl / margin) * 100 if margin > 0 else 0
        
        print(f"[{s['trade_date']}] Trades: {total_trades} | Wins: {wins} | Losses: {losses} | WinRate: {win_rate:.1f}% | PnL: ${pnl:.2f} | ROI: {roi:.2f}%")



if __name__ == "__main__":
    analyze()
