import pymysql, json

conn = pymysql.connect(host='localhost', user='root', password='', database='ai_trading', cursorclass=pymysql.cursors.DictCursor)

BASE_MARGINS = {
    'BNBUSDT': 500.0,
    'BTCUSDT': 500.0,
    'ETHUSDT': 480.0,
    'XRPUSDT': 660.0,
    'DOGEUSDT': 690.0,
    'SOLUSDT': 1000.0
}

with conn.cursor() as cur:
    cur.execute('''
        SELECT id, symbol, action, entry_price, realized_pnl, derivatives_state
        FROM experiences
        WHERE id < 25174 AND (realized_pnl IS NOT NULL AND realized_pnl != 0) AND (fees IS NULL OR fees = 0)
        LIMIT 10
    ''')
    sample_rows = cur.fetchall()
    
    print("--- SAMPLE RECONSTRUCTION TEST ---")
    for r in sample_rows:
        sym = r['symbol']
        pnl = float(r['realized_pnl'])
        entry_px = float(r['entry_price']) if r['entry_price'] else 1.0
        
        target_size = 0.5
        if r['derivatives_state']:
            try:
                ds = json.loads(r['derivatives_state']) if isinstance(r['derivatives_state'], str) else r['derivatives_state']
                target_size = float(ds.get('target_size', 0.5))
            except Exception:
                pass
                
        base_m = BASE_MARGINS.get(sym, 500.0)
        # Margin scaling
        margin = base_m * (0.5 + 0.5 * target_size)
        
        # If huge win/loss (Whale trade), margin must scale to support the PnL reasonably (e.g. max 50% ROI)
        if abs(pnl) > margin * 0.5:
            margin = abs(pnl) / 0.25 # 25% ROI
            
        leverage = 20
        notional = margin * leverage
        fee = notional * 0.0008 # 0.04% entry + 0.04% exit
        
        qty = notional / entry_px
        if 'LONG' in r['action']:
            exit_px = entry_px + (pnl / qty)
        else:
            exit_px = entry_px - (pnl / qty)
            
        roi = (pnl / margin) * 100
        print(f"ID {r['id']} | {sym} {r['action']} | Entry: {entry_px:.4f} -> Exit: {exit_px:.4f} | Margin: ${margin:.2f} | Lev: {leverage} | Fee: ${fee:.2f} | PnL: ${pnl:.2f} (ROI: {roi:.1f}%)")

conn.close()
