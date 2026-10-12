import pymysql
import time
from collections import defaultdict

conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='',
    database='ai_trading',
    cursorclass=pymysql.cursors.DictCursor
)

print("="*75)
print("[*] COMPLETE FORENSIC LIFETIME AUDIT: MATCHING OPENS TO REAL CLOSES")
print("="*75)

with conn.cursor() as cur:
    cur.execute('''
        SELECT id, timestamp, symbol, action, entry_price, exit_price, realized_pnl, fees, position_before, position_after
        FROM experiences
        ORDER BY id ASC
    ''')
    rows = cur.fetchall()

print(f"Total Database Rows: {len(rows):,}\n")

# Separate into categories
holds = [r for r in rows if r['action'] == 'HOLD']
closes = [r for r in rows if r['action'] in ('CLOSE_LONG', 'CLOSE_SHORT')]
opens = [r for r in rows if r['action'] in ('OPEN_LONG', 'OPEN_SHORT')]

print(f"Total HOLD Experiences:   {len(holds):,}")
print(f"Total CLOSED Trades:       {len(closes):,}")
print(f"Total OPEN Records:        {len(opens):,}")

# Group by symbol
by_symbol = defaultdict(list)
for r in rows:
    by_symbol[r['symbol']].append(r)

matched_opens = set()
phantom_opens = []

for sym, sym_rows in by_symbol.items():
    # Track the active open records
    pending_opens = []
    
    for r in sym_rows:
        if r['action'].startswith('OPEN'):
            pending_opens.append(r)
        elif r['action'].startswith('CLOSE'):
            # This is a real close that occurred on Binance.
            # Which OPEN record was the one that actually executed?
            # It's the open record closest in entry_price and timestamp!
            close_entry_px = r['entry_price'] or 0.0
            
            if pending_opens:
                # Find the pending open whose entry_price best matches the close's entry_price
                best_match = None
                min_diff = float('inf')
                
                for o in pending_opens:
                    o_px = o['entry_price'] or 0.0
                    diff = abs(o_px - close_entry_px)
                    if diff < min_diff:
                        min_diff = diff
                        best_match = o
                
                if best_match:
                    matched_opens.add(best_match['id'])
                    # All other pending opens before this close were unfilled limit order timeout attempts!
                    for o in pending_opens:
                        if o['id'] != best_match['id']:
                            phantom_opens.append(o)
                
                pending_opens = []
                
    # Any trailing pending opens at the very end (currently open positions vs expired limits)
    # The last one might be currently open, earlier ones are expired limits
    if len(pending_opens) > 1:
        for o in pending_opens[:-1]:
            phantom_opens.append(o)

print("\n" + "="*75)
print("[*] FORENSIC AUDIT RESULTS:")
print("="*75)
print(f"1. Verified Real Executed OPENs (Matched to Real Closes): {len(matched_opens):,} records")
print(f"2. Verified Real Executed CLOSEs:                         {len(closes):,} records")
print(f"3. Valid HOLD Observations (RL Policy Anchors):           {len(holds):,} records")
print(f"---------------------------------------------------------------------------")
print(f"4. TOTAL PHANTOM UNFILLED ATTEMPTS IDENTIFIED:             {len(phantom_opens):,} records")
print("="*75)

# Date breakdown of phantom records
phantom_by_date = defaultdict(int)
for p in phantom_opens:
    d = time.strftime('%Y-%m-%d', time.localtime(p['timestamp']))
    phantom_by_date[d] += 1

print("\n--- PHANTOM RECORDS BY DATE ---")
for d in sorted(phantom_by_date.keys()):
    print(f"  {d}: {phantom_by_date[d]:,d} phantom records")

print(f"\nSample of first 5 Phantom IDs: {[p['id'] for p in phantom_opens[:5]]}")
print(f"Sample of latest 5 Phantom IDs: {[p['id'] for p in phantom_opens[-5:]]}")

conn.close()
