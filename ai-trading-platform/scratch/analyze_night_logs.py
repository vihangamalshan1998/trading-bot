import sys
import json
import collections

counts = collections.Counter()
recent_skips_by_sym = collections.defaultdict(list)

with open("/root/.pm2/logs/trading-bot-out.log", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        try:
            d = json.loads(line)
            msg = d.get("message", "")
            if "SKIPPING" in msg or "BLOCKING" in msg or "EXECUTING" in msg or "GUARD" in msg:
                if "Low Confidence" in msg:
                    # extract hurdle
                    counts["Low Confidence"] += 1
                elif "ANTI-COUNTERTREND" in msg:
                    counts["Anti-Countertrend Guard (Fighting Trend)"] += 1
                elif "OVER-EXTENSION" in msg:
                    counts["Over-Extension Guard (Ceiling/Floor)"] += 1
                elif "fee hurdle" in msg:
                    counts["Fee Hurdle (Predicted Move Too Small)"] += 1
                elif "mandatory post-trade rest" in msg:
                    counts["Post-Trade Cooldown"] += 1
                elif "EXECUTING" in msg:
                    counts["EXECUTED: " + msg] += 1
                elif "PORTFOLIO BETA" in msg:
                    counts["Beta Governor"] += 1
                elif "CIRCUIT BREAKER" in msg:
                    counts["Circuit Breaker Active"] += 1
                else:
                    counts[msg[:60]] += 1
        except Exception:
            pass

print("=== 11-HOUR LOG AUDIT SUMMARY ===")
for k, v in counts.most_common(15):
    print(f"  {k}: {v:,} occurrences")
