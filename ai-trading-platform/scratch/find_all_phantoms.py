import pymysql
import time
from collections import defaultdict

conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    password='',
    database='ai_trading',
    cursorclass=pymysql.cursors.DictCursor,
    connect_timeout=10
)

print("="*70)
print("[*] AUDITING ALL 47,000+ EXPERIENCES FOR PHANTOM RECORDS")
print("="*70)

t0 = time.time()
with conn.cursor() as cur:
    # Notice: DO NOT SELECT market_state! Selecting metadata columns is 100x faster!
    cur.execute('''
        SELECT id, timestamp, symbol, action, entry_price, exit_price, 
               realized_pnl, fees, position_before, position_after
        FROM experiences
        ORDER BY id ASC
    ''')
    rows = cur.fetchall()

print(f"Loaded {len(rows):,} rows in {time.time()-t0:.2f}s.\n")

# Categories of Phantoms:
# 1. Unfilled Limit Order Spams (Consecutive OPENs without a fill / position change)
# 2. Duplicate timestamp spams (identical symbol and action within 2 seconds)
# 3. Ghost Closures (CLOSE actions with entry_price <= 0 or exit_price <= 0)
# 4. Zero-movement fake records

duplicate_timestamp_spams = []
ghost_closures = []
unfilled_open_spams = []

# Group by symbol to analyze the chronological life cycle of every coin
by_symbol = defaultdict(list)
for r in rows:
    by_symbol[r['symbol']].append(r)

for sym, sym_rows in by_symbol.items():
    last_row = None
    for r in sym_rows:
        # Check 1: Duplicate timestamp spam (< 2 seconds apart with same action)
        if last_row and (r['timestamp'] - last_row['timestamp']) <= 2 and r['action'] == last_row['action']:
            duplicate_timestamp_spams.append(r['id'])
            
        # Check 2: Ghost Closures (Close action but entry_price is 0 or exit is 0)
        if r['action'].startswith('CLOSE') and ((r['entry_price'] or 0) <= 0 or (r['exit_price'] or 0) <= 0):
            ghost_closures.append(r['id'])
            
        last_row = r

    # Check 3: Consecutive OPEN spams where previous OPEN never had a fill or price change
    # e.g., OPEN_SHORT -> OPEN_SHORT -> OPEN_SHORT before any CLOSE or without position increase
    pending_opens = []
    for r in sym_rows:
        if r['action'].startswith('OPEN'):
            pending_opens.append(r)
        elif r['action'].startswith('CLOSE'):
            # A close occurred. If there were multiple pending opens, only the last filled one was real!
            # The earlier ones were timed-out limit order replacement attempts!
            if len(pending_opens) > 1:
                # All except the one that actually established the position are phantoms
                for p in pending_opens[:-1]:
                    unfilled_open_spams.append(p['id'])
            pending_opens = []
            
    # Trailing pending opens at the very end that were never closed
    if len(pending_opens) > 1:
        for p in pending_opens[:-1]:
            unfilled_open_spams.append(p['id'])

all_phantoms = sorted(list(set(duplicate_timestamp_spams + ghost_closures + unfilled_open_spams)))

print(f"--- PHANTOM AUDIT BREAKDOWN ---")
print(f"1. Unfilled Limit Order Spams (Replacement Attempts): {len(unfilled_open_spams):,} records")
print(f"2. Duplicate Rapid Timestamp Spams (<= 2s spam):       {len(duplicate_timestamp_spams):,} records")
print(f"3. Ghost Closures (Zero Entry/Exit Price Closures):    {len(ghost_closures):,} records")
print(f"-------------------------------------------------------")
print(f"TOTAL UNIQUE PHANTOM RECORDS FOUND:                    {len(all_phantoms):,} records")
print(f"REAL VALID EXPERIENCES REMAINING:                      {len(rows) - len(all_phantoms):,} records")
print(f"-------------------------------------------------------")

if all_phantoms:
    print(f"\nSample of first 10 Phantom IDs: {all_phantoms[:10]}")
    print(f"Sample of latest 10 Phantom IDs: {all_phantoms[-10:]}")
    
conn.close()
