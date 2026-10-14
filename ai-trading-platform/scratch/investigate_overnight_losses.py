import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.getcwd())

from core.database.session import SessionLocal
from core.database.models import Experience
from sqlalchemy import func

def investigate():
    session = SessionLocal()
    try:
        # Check trades from the last 12 hours
        twelve_hours_ago = int(time.time() - 12 * 3600)
        
        trades = session.query(Experience).filter(
            Experience.timestamp >= twelve_hours_ago,
            Experience.action.in_(["OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"])
        ).order_by(Experience.timestamp.asc()).all()
        
        print(f"Total trades in last 12 hours: {len(trades)}")
        
        total_fees = 0.0
        total_realized_pnl = 0.0
        wins = 0
        losses = 0
        trades_by_symbol = {}
        
        for t in trades:
            sym = t.symbol
            if sym not in trades_by_symbol:
                trades_by_symbol[sym] = {"count": 0, "pnl": 0.0, "fees": 0.0}
            trades_by_symbol[sym]["count"] += 1
            trades_by_symbol[sym]["fees"] += float(t.fees or 0.0)
            trades_by_symbol[sym]["pnl"] += float(t.realized_pnl or 0.0)
            
            total_fees += float(t.fees or 0.0)
            total_realized_pnl += float(t.realized_pnl or 0.0)
            
            if "CLOSE" in t.action:
                pnl = float(t.realized_pnl or 0.0)
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                    
        print(f"Total Fees Paid: {total_fees:.2f} USDT")
        print(f"Total Realized PnL: {total_realized_pnl:.2f} USDT")
        print(f"Closed Trades: Wins={wins}, Losses={losses}")
        print("\nBreakdown by Symbol:")
        for sym, data in trades_by_symbol.items():
            print(f" -> {sym:10s} | Trades: {data['count']:4d} | PnL: {data['pnl']:+8.2f} USDT | Fees: {data['fees']:6.2f} USDT")
            
        print("\nTop 10 Biggest Losses:")
        big_losses = session.query(Experience).filter(
            Experience.timestamp >= twelve_hours_ago,
            Experience.realized_pnl < -10.0
        ).order_by(Experience.realized_pnl.asc()).limit(10).all()
        
        for l in big_losses:
            dt = datetime.fromtimestamp(l.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f" -> {dt} | {l.symbol:8s} | {l.action:12s} | PnL: {l.realized_pnl:8.2f} USDT | Fees: {l.fees:.2f} USDT | Dur: {l.holding_duration}s")

    finally:
        session.close()

if __name__ == "__main__":
    investigate()
