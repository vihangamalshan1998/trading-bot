import re
from collections import Counter

counts = Counter()
try:
    with open('/root/.pm2/logs/trading-bot-out.log', 'r', errors='ignore') as f:
        for line in f:
            m = re.search(r'\[([A-Z]+USDT)\] EXECUTING: (OPEN_[A-Z]+|CLOSE_[A-Z]+)', line)
            if m:
                counts[(m.group(1), m.group(2))] += 1
except Exception as e:
    print(f"Error reading log: {e}")

print("=====================================================")
print("📊 EXECUTING COUNTS BY SYMBOL AND SIDE")
print("=====================================================")
for (sym, side), cnt in counts.most_common():
    print(f"{sym:<10} {side:<12}: {cnt:>4} times")
