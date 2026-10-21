import sys, os
sys.path.insert(0, os.path.abspath("."))
from sqlalchemy import text
from core.database.session import SessionLocal
with SessionLocal() as s:
    q = lambda sql: s.execute(text(sql)).fetchall()
    print("total rows:", q("SELECT COUNT(*), SUM(reward_5m IS NULL), MIN(timestamp), MAX(timestamp) FROM experiences")[0])
    for r in q("""SELECT DATE(FROM_UNIXTIME(timestamp)) d, COUNT(*), SUM(reward_5m IS NOT NULL),
                  AVG(reward_5m), STDDEV(reward_5m), MIN(reward_5m), MAX(reward_5m), SUM(action LIKE '%SHORT%'), SUM(action LIKE '%LONG%')
                  FROM experiences GROUP BY d ORDER BY d DESC LIMIT 10"""):
        print(r)
    print("distinct r5m values (top):", q("SELECT ROUND(reward_5m,3) v, COUNT(*) c FROM experiences WHERE reward_5m IS NOT NULL GROUP BY v ORDER BY c DESC LIMIT 12"))
    print("has next_state:", q("SELECT SUM(next_state IS NULL), SUM(next_state IS NOT NULL) FROM experiences")[0])
