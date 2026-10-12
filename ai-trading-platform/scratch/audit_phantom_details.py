import pymysql
import time
import os
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

def audit_db(name, port, user=None, password=None):
    print("=" * 70)
    print(f"[*] AUDITING {name} (Port {port})")
    print("=" * 70)
    u = user or os.getenv('MYSQL_USER', 'root')
    p = password if password is not None else os.getenv('MYSQL_PASSWORD', '')
    if port == 3307:
        u = os.getenv('VPS_MYSQL_USER', u)
        p = os.getenv('VPS_MYSQL_PASSWORD', p)
        
    conn = pymysql.connect(
        host='127.0.0.1',
        port=port,
        user=u,
        password=p,
        database='ai_trading',
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cur:
        cur.execute('''
            SELECT id, timestamp, symbol, action, entry_price, exit_price, realized_pnl, fees, position_before, position_after
            FROM experiences
            ORDER BY id ASC
        ''')
        rows = cur.fetchall()
    conn.close()

    total = len(rows)
    holds = [r for r in rows if r['action'] == 'HOLD']
    closes = [r for r in rows if r['action'].startswith('CLOSE')]
    opens = [r for r in rows if r['action'].startswith('OPEN')]
    print(f"Total rows: {total:,} | Opens: {len(opens):,} | Closes: {len(closes):,} | Holds: {len(holds):,}")

    by_sym = defaultdict(list)
    for r in rows:
        by_sym[r['symbol']].append(r)

    matched_open_ids = set()
    phantom_open_ids = set()

    for sym, sym_rows in by_sym.items():
        pending_opens = []
        for r in sym_rows:
            if r['action'].startswith('OPEN'):
                pending_opens.append(r)
            elif r['action'].startswith('CLOSE'):
                close_entry_px = r['entry_price'] or 0.0
                if pending_opens:
                    # Best match by price & timestamp proximity
                    best = min(pending_opens, key=lambda o: (abs((o['entry_price'] or 0.0) - close_entry_px), abs(r['timestamp'] - o['timestamp'])))
                    matched_open_ids.add(best['id'])
                    for o in pending_opens:
                        if o['id'] != best['id']:
                            phantom_open_ids.add(o['id'])
                    pending_opens = []

        # If any pending opens remain at the end (current open positions)
        if pending_opens:
            # The very last one could be currently active open position
            matched_open_ids.add(pending_opens[-1]['id'])
            for o in pending_opens[:-1]:
                phantom_open_ids.add(o['id'])

    print(f"Verified Real Opens: {len(matched_open_ids):,}")
    print(f"Verified Real Closes: {len(closes):,}")
    print(f"Verified Real Holds:  {len(holds):,}")
    print(f"PHANTOM Records Identified: {len(phantom_open_ids):,} ({len(phantom_open_ids)/total*100:.1f}% of DB)")
    print(f"Real Total after cleanup: {total - len(phantom_open_ids):,}")
    return phantom_open_ids

print("Testing Audit on Local and VPS:")
local_phantoms = audit_db("LOCAL DB", 3306)
vps_phantoms = audit_db("VPS DB", 3307)
