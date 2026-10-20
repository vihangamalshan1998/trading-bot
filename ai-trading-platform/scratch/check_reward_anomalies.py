from sqlalchemy import create_engine, text
from core.config.settings import settings
e = create_engine(settings.database_url)
with e.connect() as c:
    q = lambda s: c.execute(text(s)).fetchall()
    print("5m < -2%:", q("SELECT COUNT(*) FROM experiences WHERE reward_5m < -2")[0][0])
    print("5m > 2%:", q("SELECT COUNT(*) FROM experiences WHERE reward_5m > 2")[0][0])
    print("1h |x|>10%:", q("SELECT COUNT(*) FROM experiences WHERE ABS(reward_1h) > 10")[0][0])
    print("4h |x|>15%:", q("SELECT COUNT(*) FROM experiences WHERE ABS(reward_4h) > 15")[0][0])
    print("5m exactly 0:", q("SELECT COUNT(*) FROM experiences WHERE reward_5m = 0")[0][0])
    print("5m positive:", q("SELECT COUNT(*) FROM experiences WHERE reward_5m > 0")[0][0], " negative:", q("SELECT COUNT(*) FROM experiences WHERE reward_5m < 0")[0][0])
    print("\nworst 5m rows:")
    for r in q("SELECT id, symbol, action, FROM_UNIXTIME(timestamp), entry_price, reward_5m, reward_1h, reward_4h FROM experiences WHERE reward_5m IS NOT NULL ORDER BY reward_5m ASC LIMIT 8"):
        print("  ", r)
    print("\nper-day graded stats:")
    for r in q("SELECT DATE(FROM_UNIXTIME(timestamp)) d, COUNT(*), MIN(reward_5m), MAX(reward_5m), AVG(ABS(reward_5m)), SUM(reward_5m=0) FROM experiences WHERE reward_5m IS NOT NULL GROUP BY d ORDER BY d"):
        print("  ", r)
    print("\nper-symbol entry_price vs mid sample (HOLD rows with entry_price>0):", q("SELECT COUNT(*) FROM experiences WHERE action='HOLD' AND entry_price > 0")[0][0])
