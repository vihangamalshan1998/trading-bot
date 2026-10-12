import pymysql

conn = pymysql.connect(host='localhost', user='root', password='', database='ai_trading', cursorclass=pymysql.cursors.DictCursor)
with conn.cursor() as cur:
    cur.execute('''
        SELECT 
            id, symbol, action, entry_price, exit_price, margin, leverage, fees, realized_pnl, holding_duration,
            (realized_pnl + fees) as gross_pnl
        FROM experiences
        WHERE DATE(FROM_UNIXTIME(timestamp)) = '2026-10-12'
          AND realized_pnl IS NOT NULL AND realized_pnl != 0
    ''')
    rows = cur.fetchall()
    
    total = len(rows)
    gross_positive = [r for r in rows if r['gross_pnl'] > 0]
    gross_negative = [r for r in rows if r['gross_pnl'] < 0]
    gross_zero = [r for r in rows if r['gross_pnl'] == 0]
    
    net_positive = [r for r in rows if r['realized_pnl'] > 0]
    net_negative = [r for r in rows if r['realized_pnl'] < 0]
    
    # Trades that were DIRECTIONALLY WINNING (gross > 0) but turned into NET LOSSES by fees
    killed_by_fees = [r for r in rows if r['gross_pnl'] > 0 and r['realized_pnl'] <= 0]
    
    total_gross = sum(r['gross_pnl'] for r in rows)
    total_fees = sum(r['fees'] for r in rows)
    total_net = sum(r['realized_pnl'] for r in rows)
    
    print("==========================================")
    print("TODAY'S FORENSIC TRADING AUDIT (2026-10-12)")
    print("==========================================")
    print(f"Total Closed Trades Today: {total}")
    print(f"Gross PnL (Market Directional Result): ${total_gross:.2f}")
    print(f"Total Exchange Fees Paid:               ${total_fees:.2f}")
    print(f"Net Realized PnL:                       ${total_net:.2f}")
    print("------------------------------------------")
    print(f"Gross Directional Wins:  {len(gross_positive)} ({len(gross_positive)/total*100:.1f}%)")
    print(f"Gross Directional Losses:{len(gross_negative)} ({len(gross_negative)/total*100:.1f}%)")
    print(f"Net Reported Wins:       {len(net_positive)} ({len(net_positive)/total*100:.1f}%)")
    print(f"Net Reported Losses:     {len(net_negative)} ({len(net_negative)/total*100:.1f}%)")
    print("------------------------------------------")
    print(f"TRADES KILLED ENTIRELY BY FEES: {len(killed_by_fees)} trades!")
    print(f"These {len(killed_by_fees)} trades predicted the market RIGHT, but the fee turned them into losses.")
    
    # Check by symbol
    from collections import defaultdict
    by_sym = defaultdict(lambda: {'count': 0, 'gross': 0.0, 'fees': 0.0, 'net': 0.0, 'wins': 0})
    for r in rows:
        s = r['symbol']
        by_sym[s]['count'] += 1
        by_sym[s]['gross'] += r['gross_pnl']
        by_sym[s]['fees'] += r['fees']
        by_sym[s]['net'] += r['realized_pnl']
        if r['realized_pnl'] > 0:
            by_sym[s]['wins'] += 1
            
    print("\nBY COIN TODAY:")
    for sym, data in sorted(by_sym.items(), key=lambda x: x[1]['net']):
        wr = data['wins'] / data['count'] * 100
        print(f"  {sym:8s} | Trades: {data['count']:4d} | WR: {wr:4.1f}% | Gross: ${data['gross']:8.2f} | Fees: ${data['fees']:8.2f} | Net: ${data['net']:8.2f}")

conn.close()
