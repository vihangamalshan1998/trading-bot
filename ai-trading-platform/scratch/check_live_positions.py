import redis, json

try:
    r = redis.Redis(host='127.0.0.1', port=6379, db=0, socket_timeout=3)
    pos = r.get('portfolio:state')
    if pos:
        pos_data = json.loads(pos)
        print("PORTFOLIO STATE:")
        print("  Equity:", pos_data.get('equity'))
        print("  Wallet Balance:", pos_data.get('wallet_balance'))
        print("  Free Margin:", pos_data.get('free_margin'))
        print("  Total Unrealized PnL:", pos_data.get('total_unrealized_pnl'))
        print("\nACTIVE POSITIONS:")
        for sym, p in pos_data.get('positions', {}).items():
            qty = p.get('quantity', 0)
            if qty != 0:
                print(f"  {sym:8s} | Qty: {qty:10.4f} | Entry: {p.get('entry_price', 0):10.2f} | UnPnL: ${p.get('unrealized_pnl', 0):8.2f} | Leverage: {p.get('leverage')}")
    else:
        print("portfolio:state key not found in Redis.")
except Exception as e:
    print(f"Redis error: {e}")
