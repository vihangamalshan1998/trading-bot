import pymysql
import datetime
import time
import json
import redis

# Local time is UTC+5:30
# Period 1: 14 Oct 4:00 PM (+5:30) = 14 Oct 10:30 UTC to 15 Oct 5:00 AM (+5:30) = 14 Oct 23:30 UTC
# Period 2: 15 Oct 5:00 AM (+5:30) = 14 Oct 23:30 UTC to Now (15 Oct ~03:33 UTC)

p1_start = datetime.datetime(2026, 10, 14, 10, 30, tzinfo=datetime.timezone.utc).timestamp()
p2_start = datetime.datetime(2026, 10, 14, 23, 30, tzinfo=datetime.timezone.utc).timestamp()
p2_end = time.time()

print(f"Current server time: {datetime.datetime.now(datetime.timezone.utc)}")
print(f"Period 1: {datetime.datetime.fromtimestamp(p1_start, datetime.timezone.utc)} to {datetime.datetime.fromtimestamp(p2_start, datetime.timezone.utc)}")
print(f"Period 2: {datetime.datetime.fromtimestamp(p2_start, datetime.timezone.utc)} to {datetime.datetime.fromtimestamp(p2_end, datetime.timezone.utc)}")

# 1. Connect to MySQL
conn = pymysql.connect(
    host="localhost",
    user="trading_user",
    password="trading_password",
    database="ai_trading",
    cursorclass=pymysql.cursors.DictCursor
)

def query_mysql_trades(start, end, label):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol, action, confidence, realized_pnl, fees, reward, holding_duration, timestamp, entry_price, exit_price "
            "FROM experiences "
            "WHERE timestamp >= %s AND timestamp < %s AND (action LIKE '%%CLOSE%%' OR realized_pnl != 0) "
            "ORDER BY timestamp ASC",
            (start, end)
        )
        trades = cur.fetchall()
        print(f"\n=======================================================")
        print(f"🗄️ MYSQL DATABASE: {label} (Total: {len(trades)})")
        print(f"=======================================================")
        if not trades:
            print("No trades found in MySQL experiences for this window.")
            return trades
        wins = [t for t in trades if float(t.get("realized_pnl", 0)) > 0]
        losses = [t for t in trades if float(t.get("realized_pnl", 0)) < 0]
        be = [t for t in trades if float(t.get("realized_pnl", 0)) == 0]
        wr = (len(wins) / len(trades)) * 100 if trades else 0.0
        net_pnl = sum(float(t.get("realized_pnl", 0)) for t in trades)
        fees = sum(float(t.get("fees", 0)) for t in trades)
        gross_profit = sum(float(t.get("realized_pnl", 0)) for t in wins)
        gross_loss = abs(sum(float(t.get("realized_pnl", 0)) for t in losses))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
        avg_dur = sum(int(t.get("holding_duration", 0)) for t in trades) / len(trades) if trades else 0.0
        avg_conf = sum(float(t.get("confidence", 0)) for t in trades) / len(trades) if trades else 0.0

        print(f"Win Rate:            {wr:.1f}% ({len(wins)}W / {len(losses)}L / {len(be)}BE)")
        print(f"Total Net PnL:       ${net_pnl:+.4f} USDT")
        print(f"Total Fees:          ${fees:.4f} USDT")
        print(f"Profit Factor:       {pf:.2f}")
        print(f"Avg Hold Duration:   {avg_dur:.1f}s ({avg_dur/60:.1f} mins)")
        print(f"Avg Confidence:      {avg_conf*100:.1f}%")

        syms = {}
        for t in trades:
            s = t.get("symbol", "UNKNOWN")
            if s not in syms:
                syms[s] = {"w": 0, "l": 0, "pnl": 0.0, "fees": 0.0, "n": 0}
            syms[s]["n"] += 1
            pnl = float(t.get("realized_pnl", 0))
            syms[s]["pnl"] += pnl
            syms[s]["fees"] += float(t.get("fees", 0))
            if pnl > 0: syms[s]["w"] += 1
            elif pnl < 0: syms[s]["l"] += 1

        print("\nSymbol Breakdown:")
        for s, d in syms.items():
            swr = (d["w"] / d["n"]) * 100 if d["n"] else 0.0
            print(f"  {s:<10} | Trades: {d['n']:<2} | WR: {swr:5.1f}% | Net PnL: ${d['pnl']:+8.4f} | Fees: ${d['fees']:6.4f}")
        return trades

mysql_p1 = query_mysql_trades(p1_start, p2_start, "Period 1: 14 Oct 4:00 PM to 15 Oct 5:00 AM Local")
mysql_p2 = query_mysql_trades(p2_start, p2_end, "Period 2: 15 Oct 5:00 AM Local to NOW")

# 2. Check Redis dashboard trade history
r = redis.Redis(host='localhost', port=6379, db=0)
raw_history = r.lrange("dashboard:trade_history", 0, -1)
print(f"\n[Redis] dashboard:trade_history items: {len(raw_history)}")
parsed_history = []
for item in raw_history:
    try:
        parsed_history.append(json.loads(item))
    except Exception:
        pass

p1_trades = []
p2_trades = []
other_trades = []

for t in parsed_history:
    ts = t.get("timestamp", 0)
    if "CLOSE" in t.get("side", ""):
        if p1_start <= ts < p2_start:
            p1_trades.append(t)
        elif p2_start <= ts <= p2_end:
            p2_trades.append(t)
        else:
            other_trades.append(t)

print(f"Redis P1 closed trades: {len(p1_trades)}")
print(f"Redis P2 closed trades: {len(p2_trades)}")
print(f"Redis Other closed trades: {len(other_trades)}")

def analyze_trade_set(trades, name):
    print(f"\n=======================================================")
    print(f"📊 ANALYSIS: {name} (Total Trades: {len(trades)})")
    print(f"=======================================================")
    if not trades:
        print("No completed trades in this window.")
        return

    wins = [t for t in trades if float(t.get("realized_pnl", 0)) > 0]
    losses = [t for t in trades if float(t.get("realized_pnl", 0)) < 0]
    breakeven = [t for t in trades if float(t.get("realized_pnl", 0)) == 0]

    win_rate = (len(wins) / len(trades)) * 100 if trades else 0.0
    total_realized_pnl = sum(float(t.get("realized_pnl", 0)) for t in trades)
    total_gross_pnl = sum(float(t.get("gross_pnl", 0)) for t in trades)
    total_fees = sum(float(t.get("fees_paid", 0)) for t in trades)
    avg_roi = sum(float(t.get("roi_pct", 0)) for t in trades) / len(trades) if trades else 0.0
    avg_conf = sum(float(t.get("confidence", 0)) for t in trades) / len(trades) if trades else 0.0

    gross_profit = sum(float(t.get("realized_pnl", 0)) for t in wins)
    gross_loss = abs(sum(float(t.get("realized_pnl", 0)) for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    print(f"Win Rate:            {win_rate:.1f}% ({len(wins)}W / {len(losses)}L / {len(breakeven)}BE)")
    print(f"Total Net PnL:       ${total_realized_pnl:+.4f} USDT")
    print(f"Total Gross PnL:     ${total_gross_pnl:+.4f} USDT")
    print(f"Total Fees Paid:     ${total_fees:.4f} USDT")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print(f"Average ROI:         {avg_roi:+.2f}%")
    print(f"Average Confidence:  {avg_conf*100:.1f}%")

    # Symbol breakdown
    sym_map = {}
    for t in trades:
        s = t.get("symbol", "UNKNOWN")
        if s not in sym_map:
            sym_map[s] = {"wins": 0, "losses": 0, "pnl": 0.0, "fees": 0.0, "trades": 0}
        sym_map[s]["trades"] += 1
        pnl = float(t.get("realized_pnl", 0))
        sym_map[s]["pnl"] += pnl
        sym_map[s]["fees"] += float(t.get("fees_paid", 0))
        if pnl > 0:
            sym_map[s]["wins"] += 1
        elif pnl < 0:
            sym_map[s]["losses"] += 1

    print("\nSymbol Breakdown:")
    for s, data in sym_map.items():
        wr = (data["wins"] / data["trades"]) * 100 if data["trades"] else 0.0
        print(f"  {s:<10} | Trades: {data['trades']} | WR: {wr:5.1f}% | Net PnL: ${data['pnl']:+8.4f} | Fees: ${data['fees']:6.4f}")

    # Detailed log omitted to keep summary visible
    pass

analyze_trade_set(p1_trades, "Period 1: 14 Oct 4:00 PM to 15 Oct 5:00 AM Local")
analyze_trade_set(p2_trades, "Period 2: 15 Oct 5:00 AM Local to NOW")

# Also check PM2 log files for any trades if Redis only has 100
print("\n--- PM2 Log grep for EXECUTING / ORDER SUCCESS ---")
import subprocess
try:
    res = subprocess.run(["grep", "-E", "EXECUTING:|ORDER SUCCESS:|MARKET INTENT", "/root/.pm2/logs/trading-bot-out.log"], capture_output=True, text=True)
    lines = res.stdout.strip().split("\n")
    print(f"Total execution lines in current PM2 log: {len(lines)}")
    # Print last 20
    for l in lines[-20:]:
        print(" ", l)
except Exception as e:
    print("Could not grep PM2 logs:", e)
