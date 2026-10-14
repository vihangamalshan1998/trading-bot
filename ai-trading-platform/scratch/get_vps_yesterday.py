import asyncio
import os
import sys
import time
from datetime import datetime
import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    user='ai_trading',
    password='ai_trading_password',
    database='ai_trading'
)

with conn.cursor() as cur:
    print("=== ACTUAL OCTOBER 13 (YESTERDAY) TRADING PERFORMANCE ===")
    
    # 1. Total PnL and trades on Oct 13
    cur.execute("""
        SELECT 
            symbol,
            COUNT(*) as trade_count,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as win_count,
            SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as loss_count,
            MAX(realized_pnl) as max_win,
            MIN(realized_pnl) as max_loss,
            SUM(realized_pnl) as total_gross_pnl,
            SUM(fees) as total_fees,
            SUM(realized_pnl - fees) as net_pnl
        FROM experiences
        WHERE action != 'HOLD' 
          AND timestamp >= UNIX_TIMESTAMP('2026-10-13 00:00:00')
          AND timestamp < UNIX_TIMESTAMP('2026-10-14 00:00:00')
        GROUP BY symbol
        ORDER BY total_gross_pnl DESC
    """)
    rows = cur.fetchall()
    
    total_gross = 0.0
    total_fees = 0.0
    total_trades = 0
    
    print(f"{'SYMBOL':8s} | {'TRADES':6s} | {'WINS':5s} | {'LOSSES':6s} | {'MAX WIN':10s} | {'GROSS PNL':12s} | {'FEES':10s} | {'NET PNL':12s}")
    print("-" * 85)
    for r in rows:
        sym, trades, wins, losses, max_w, max_l, gross, fees, net = r
        gross = gross or 0.0
        fees = fees or 0.0
        net = net or 0.0
        max_w = max_w or 0.0
        total_gross += gross
        total_fees += fees
        total_trades += trades
        print(f"{sym:8s} | {trades:6d} | {wins:5d} | {losses:6d} | +${max_w:8.2f} | ${gross:+10.2f} | -${fees:8.2f} | ${net:+10.2f}")
        
    print("-" * 85)
    print(f"TOTAL:   | {total_trades:6d} |       |        |            | ${total_gross:+10.2f} | -${total_fees:8.2f} | ${total_gross - total_fees:+10.2f}")

    # 2. Hourly breakdown of Oct 13 to see when wins happened
    print("\n=== HOURLY BREAKDOWN FOR OCT 13 ===")
    cur.execute("""
        SELECT 
            DATE_FORMAT(FROM_UNIXTIME(timestamp), '%H:00') as hour_slot,
            COUNT(*) as trades,
            SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(realized_pnl) as gross_pnl,
            SUM(fees) as fees_paid
        FROM experiences
        WHERE action != 'HOLD'
          AND timestamp >= UNIX_TIMESTAMP('2026-10-13 00:00:00')
          AND timestamp < UNIX_TIMESTAMP('2026-10-14 00:00:00')
        GROUP BY hour_slot
        ORDER BY hour_slot ASC
    """)
    for r in cur.fetchall():
        h, cnt, w, g, f = r
        g = g or 0.0
        f = f or 0.0
        print(f" {h} | Trades: {cnt:5d} | Wins: {w:4d} | Gross: ${g:+8.2f} | Fees: -${f:7.2f} | Net: ${g - f:+8.2f}")

conn.close()
