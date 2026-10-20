import sys, os
sys.path.insert(0, os.path.abspath("."))
import datetime
from sqlalchemy import create_engine, text
from core.config.settings import settings

engine = create_engine(settings.database_url)

with engine.connect() as conn:
    # Query closed trades with PnL or reward
    query = text("""
        SELECT symbol, action, realized_pnl, fees, reward, reward_5m, reward_1h, reward_4h, entry_price, exit_price, timestamp
        FROM experiences
        WHERE action IN ('CLOSE_LONG', 'CLOSE_SHORT')
           OR (realized_pnl IS NOT NULL AND realized_pnl != 0.0)
        ORDER BY timestamp ASC
    """)
    rows = conn.execute(query).fetchall()
    
print(f"Total Closed Trade Records Found: {len(rows)}")

if not rows:
    # Check if trades are logged in another table or format
    print("No direct closed trade rows with realized_pnl. Checking reward distributions...")
else:
    total_trades = len(rows)
    wins = 0
    losses = 0
    breakeven = 0
    total_pnl = 0.0
    total_fees = 0.0
    gross_profits = 0.0
    gross_losses = 0.0
    
    coin_stats = {}
    
    trade_list = []
    
    for r in rows:
        sym = r[0]
        act = r[1]
        pnl = r[2] if r[2] is not None else 0.0
        fee = r[3] if r[3] is not None else 0.0
        net_pnl = pnl - fee
        rew = r[4] or 0.0
        ts = r[10]
        
        # If realized_pnl is 0.0 but reward is nonzero, record reward as proxy percentage return
        effective_return = net_pnl if (pnl != 0.0 or fee != 0.0) else rew
        
        if effective_return > 0.0001:
            wins += 1
            gross_profits += effective_return
        elif effective_return < -0.0001:
            losses += 1
            gross_losses += abs(effective_return)
        else:
            breakeven += 1
            
        total_pnl += effective_return
        total_fees += fee
        
        if sym not in coin_stats:
            coin_stats[sym] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "fees": 0.0}
        coin_stats[sym]["trades"] += 1
        if effective_return > 0.0001: coin_stats[sym]["wins"] += 1
        elif effective_return < -0.0001: coin_stats[sym]["losses"] += 1
        coin_stats[sym]["pnl"] += effective_return
        coin_stats[sym]["fees"] += fee
        
        trade_list.append({
            "symbol": sym,
            "action": act,
            "pnl": effective_return,
            "fees": fee,
            "entry": r[8],
            "exit": r[9],
            "ts": ts
        })

    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
    profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else float("inf")
    
    print("\n" + "="*50)
    print("HISTORICAL TRADING SUMMARY")
    print("="*50)
    print(f"Total Closed Positions: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses} | Breakeven: {breakeven}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Cumulative Return / Net PnL: {total_pnl:+.4f}")
    print(f"Total Fees Incurred: ${total_fees:.4f}")
    print(f"Gross Profits: +{gross_profits:.4f} | Gross Losses: -{gross_losses:.4f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    print("\n--- PERFORMANCE BY SYMBOL ---")
    sorted_coins = sorted(coin_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)
    for sym, s in sorted_coins:
        c_wr = (s['wins'] / (s['wins'] + s['losses']) * 100) if (s['wins'] + s['losses']) > 0 else 0.0
        print(f"  {sym:<12} | Trades: {s['trades']:<4} | WinRate: {c_wr:5.1f}% | Net PnL: {s['pnl']:+8.4f} | Fees: ${s['fees']:.2f}")
        
    print("\n--- TOP 5 BIGGEST WINS ---")
    sorted_by_win = sorted(trade_list, key=lambda x: x["pnl"], reverse=True)[:5]
    for w in sorted_by_win:
        dt = datetime.datetime.fromtimestamp(w['ts']).strftime('%Y-%m-%d %H:%M') if w['ts'] else 'N/A'
        print(f"  [{dt}] {w['symbol']:<10} {w['action']:<12} Net: {w['pnl']:+8.4f} (Entry: {w['entry']}, Exit: {w['exit']})")

    print("\n--- TOP 5 BIGGEST LOSSES ---")
    sorted_by_loss = sorted(trade_list, key=lambda x: x["pnl"])[:5]
    for l in sorted_by_loss:
        dt = datetime.datetime.fromtimestamp(l['ts']).strftime('%Y-%m-%d %H:%M') if l['ts'] else 'N/A'
        print(f"  [{dt}] {l['symbol']:<10} {l['action']:<12} Net: {l['pnl']:+8.4f} (Entry: {l['entry']}, Exit: {l['exit']})")
