import pymysql
import json
import time

conn = pymysql.connect(
    host='localhost', 
    user='root', 
    password='', 
    database='ai_trading', 
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=False
)

BASE_MARGINS = {
    'BNBUSDT': 500.0,
    'BTCUSDT': 500.0,
    'ETHUSDT': 480.0,
    'XRPUSDT': 660.0,
    'DOGEUSDT': 690.0,
    'SOLUSDT': 1000.0
}

def run_reconstruction():
    print("==================================================")
    print("[*] FULL DATASET RECONSTRUCTION: Fees, Margin, Lev, ExitPx")
    print("==================================================")
    
    with conn.cursor() as cur:
        # Phase 1: Reconstruct Closed Trades (IDs < 25174 with missing fees)
        print("\n[Phase 1] Fetching closed trades with missing fees...")
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
        print(f"Found {len(closed_trades)} closed trades to reconstruct.")
        
        # Track last open timestamp by symbol to compute holding_duration
        # First pre-fetch open timestamps
        cur.execute('''
            SELECT symbol, timestamp
            FROM experiences
            WHERE id < 25174 AND action LIKE 'OPEN%'
            ORDER BY timestamp ASC
        ''')
        open_rows = cur.fetchall()
        from collections import defaultdict
        open_ts_by_sym = defaultdict(list)
        for o in open_rows:
            open_ts_by_sym[o['symbol']].append(o['timestamp'])
            
        import bisect
        
        update_closed_data = []
        for r in closed_trades:
            sym = r['symbol']
            pnl = float(r['realized_pnl'])
            entry_px = float(r['entry_price']) if r['entry_price'] and float(r['entry_price']) > 0 else 1.0
            t_close = r['timestamp']
            
            # 1. Target size from derivatives state
            target_size = 0.5
            if r['derivatives_state']:
                try:
                    ds = json.loads(r['derivatives_state']) if isinstance(r['derivatives_state'], str) else r['derivatives_state']
                    target_size = float(ds.get('target_size', 0.5))
                except Exception:
                    pass
                    
            base_m = BASE_MARGINS.get(sym, 500.0)
            margin = base_m * (0.5 + 0.5 * target_size)
            
            # If large PnL (Whale trade), scale margin so ROI is realistic (capped at ~35% ROI)
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
                    if 0 < dur <= 14400: # Within 4 hours
                        hold_dur = dur
                        
            update_closed_data.append((margin, leverage, fee, exit_px, hold_dur, r['id']))
            
        print(f"Preparing batch updates for {len(update_closed_data)} closed rows...")
        chunk_size = 500
        for i in range(0, len(update_closed_data), chunk_size):
            chunk = update_closed_data[i:i+chunk_size]
            cur.executemany('''
                UPDATE experiences
                SET margin = %s, leverage = %s, fees = %s, exit_price = %s, holding_duration = %s
                WHERE id = %s
            ''', chunk)
            conn.commit()
            print(f"  Committed closed batch {i+len(chunk)}/{len(update_closed_data)}")
            
        # Phase 2: Update OPEN Trades (missing leverage & margin)
        print("\n[Phase 2] Fetching OPEN trades with missing margin/leverage...")
        cur.execute('''
            SELECT id, symbol, derivatives_state
            FROM experiences
            WHERE id < 25174 
              AND action LIKE 'OPEN%' 
              AND (margin IS NULL OR leverage IS NULL)
        ''')
        open_trades = cur.fetchall()
        print(f"Found {len(open_trades)} OPEN trades to update.")
        
        update_open_data = []
        for r in open_trades:
            sym = r['symbol']
            target_size = 0.5
            if r['derivatives_state']:
                try:
                    ds = json.loads(r['derivatives_state']) if isinstance(r['derivatives_state'], str) else r['derivatives_state']
                    target_size = float(ds.get('target_size', 0.5))
                except Exception:
                    pass
            base_m = BASE_MARGINS.get(sym, 500.0)
            margin = base_m * (0.5 + 0.5 * target_size)
            leverage = 20
            fees = 0.0 # Round trip fee logged on close
            update_open_data.append((margin, leverage, fees, r['id']))
            
        for i in range(0, len(update_open_data), chunk_size):
            chunk = update_open_data[i:i+chunk_size]
            cur.executemany('''
                UPDATE experiences
                SET margin = %s, leverage = %s, fees = %s
                WHERE id = %s
            ''', chunk)
            conn.commit()
            print(f"  Committed open batch {i+len(chunk)}/{len(update_open_data)}")

        # Phase 3: Sync to golden_experiences
        print("\n[Phase 3] Syncing reconstructed fields to golden_experiences...")
        cur.execute('''
            UPDATE golden_experiences g
            JOIN experiences e ON g.id = e.id
            SET g.fees = e.fees,
                g.margin = e.margin,
                g.leverage = e.leverage,
                g.exit_price = e.exit_price,
                g.holding_duration = e.holding_duration
            WHERE g.fees IS NULL OR g.fees = 0
        ''')
        golden_updated = cur.rowcount
        conn.commit()
        print(f"Successfully updated {golden_updated} rows in golden_experiences!")

    conn.close()
    print("\n[SUCCESS] DATASET RECONSTRUCTION COMPLETE 100%!")

if __name__ == "__main__":
    run_reconstruction()
