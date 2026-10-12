import pymysql
import json
import time
from collections import defaultdict
import bisect

conn = pymysql.connect(
    host='localhost', 
    user='root', 
    password='', 
    database='ai_trading', 
    cursorclass=pymysql.cursors.DictCursor
)

BASE_MARGINS = {
    'BNBUSDT': 500.0,
    'BTCUSDT': 500.0,
    'ETHUSDT': 480.0,
    'XRPUSDT': 660.0,
    'DOGEUSDT': 690.0,
    'SOLUSDT': 1000.0
}

def fast_reconstruct():
    t0 = time.time()
    print("==================================================")
    print("[*] HIGH-SPEED RECONSTRUCTION OF FEES & MISSING COLUMNS")
    print("==================================================")
    
    with conn.cursor() as cur:
        # 1. Fetch pre-existing OPEN timestamps to compute holding durations accurately
        print("[1/5] Indexing OPEN timestamps for duration calculation...")
        cur.execute("SELECT symbol, timestamp FROM experiences WHERE id < 25174 AND action LIKE 'OPEN%' ORDER BY timestamp ASC")
        open_ts_by_sym = defaultdict(list)
        for row in cur.fetchall():
            open_ts_by_sym[row['symbol']].append(row['timestamp'])
            
        # 2. Fetch all closed trades before ID 25174 with missing fees
        print("[2/5] Fetching closed trades with missing fees...")
        cur.execute('''
            SELECT id, symbol, action, entry_price, realized_pnl, timestamp, derivatives_state
            FROM experiences
            WHERE id < 25174 
              AND realized_pnl IS NOT NULL 
              AND realized_pnl != 0 
              AND (fees IS NULL OR fees = 0)
            ORDER BY id ASC
        ''')
        closed_trades = cur.fetchall()
        print(f"      Loaded {len(closed_trades)} closed trades to reconstruct.")

        recon_values = []
        for r in closed_trades:
            sym = r['symbol']
            pnl = float(r['realized_pnl'])
            entry_px = float(r['entry_price']) if r['entry_price'] and float(r['entry_price']) > 0 else 1.0
            t_close = r['timestamp']
            
            target_size = 0.5
            if r['derivatives_state']:
                try:
                    ds = json.loads(r['derivatives_state']) if isinstance(r['derivatives_state'], str) else r['derivatives_state']
                    target_size = float(ds.get('target_size', 0.5))
                except Exception:
                    pass
                    
            base_m = BASE_MARGINS.get(sym, 500.0)
            margin = base_m * (0.5 + 0.5 * target_size)
            
            # If large PnL (Whale trade), scale margin so ROI is realistic
            if abs(pnl) > margin * 0.5:
                margin = abs(pnl) / 0.25 # 25% ROI
                
            leverage = 20
            notional = margin * leverage
            fee = round(notional * 0.0008, 4) # 0.04% entry + 0.04% exit
            
            qty = notional / entry_px
            if 'LONG' in r['action']:
                exit_px = entry_px + (pnl / qty)
            else:
                exit_px = entry_px - (pnl / qty)
            exit_px = max(0.00001, round(exit_px, 6))
            
            # Holding duration
            hold_dur = None
            sym_opens = open_ts_by_sym.get(sym, [])
            if sym_opens:
                idx = bisect.bisect_right(sym_opens, t_close) - 1
                if idx >= 0:
                    dur = t_close - sym_opens[idx]
                    if 0 < dur <= 14400:
                        hold_dur = dur
                        
            recon_values.append((r['id'], round(margin, 2), leverage, fee, exit_px, hold_dur))

        # 3. Use temporary table to bulk insert and update experiences in 1 step!
        print("[3/5] Applying bulk updates via temporary staging table...")
        cur.execute('''
            CREATE TEMPORARY TABLE temp_recon (
                id INT PRIMARY KEY,
                margin FLOAT,
                leverage INT,
                fees FLOAT,
                exit_price FLOAT,
                holding_duration INT
            );
        ''')
        
        batch_size = 2000
        for i in range(0, len(recon_values), batch_size):
            chunk = recon_values[i:i+batch_size]
            cur.executemany('''
                INSERT INTO temp_recon (id, margin, leverage, fees, exit_price, holding_duration)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', chunk)
            
        # Perform single joined update
        cur.execute('''
            UPDATE experiences e
            JOIN temp_recon t ON e.id = t.id
            SET e.margin = t.margin,
                e.leverage = t.leverage,
                e.fees = t.fees,
                e.exit_price = t.exit_price,
                e.holding_duration = COALESCE(t.holding_duration, e.holding_duration);
        ''')
        conn.commit()
        print(f"      Updated {len(recon_values)} closed rows in experiences!")

        # 4. Update OPEN trades (missing margin/leverage)
        print("[4/5] Updating OPEN trades with margin and leverage...")
        cur.execute('''
            UPDATE experiences
            SET leverage = 20,
                margin = CASE 
                    WHEN symbol = 'SOLUSDT' THEN 1000.0
                    WHEN symbol = 'DOGEUSDT' THEN 690.0
                    WHEN symbol = 'XRPUSDT' THEN 660.0
                    WHEN symbol = 'BNBUSDT' THEN 500.0
                    WHEN symbol = 'BTCUSDT' THEN 500.0
                    WHEN symbol = 'ETHUSDT' THEN 480.0
                    ELSE 500.0
                END,
                fees = 0.0
            WHERE id < 25174 
              AND action LIKE 'OPEN%' 
              AND (margin IS NULL OR leverage IS NULL OR fees IS NULL);
        ''')
        open_count = cur.rowcount
        conn.commit()
        print(f"      Updated {open_count} OPEN rows in experiences!")

        # 5. Sync to golden_experiences
        print("[5/5] Syncing reconstructed fields into golden_experiences...")
        cur.execute('''
            UPDATE golden_experiences g
            JOIN experiences e ON g.id = e.id
            SET g.fees = e.fees,
                g.margin = e.margin,
                g.leverage = e.leverage,
                g.exit_price = e.exit_price,
                g.holding_duration = e.holding_duration
            WHERE g.fees IS NULL OR g.fees = 0;
        ''')
        golden_count = cur.rowcount
        conn.commit()
        print(f"      Updated {golden_count} rows in golden_experiences!")

    conn.close()
    elapsed = time.time() - t0
    print(f"\n[SUCCESS] ALL DATA RECONSTRUCTED & SAVED IN {elapsed:.2f}s!")

if __name__ == "__main__":
    fast_reconstruct()
