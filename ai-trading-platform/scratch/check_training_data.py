"""Inspect training data health: dones, grading, reward units, action/size label distribution."""
from sqlalchemy import create_engine, text
from core.config.settings import settings

e = create_engine(settings.database_url)
with e.connect() as c:
    q = lambda s: c.execute(text(s)).fetchall()
    print("total rows:", q("SELECT COUNT(*) FROM experiences")[0][0])
    print("next_state NOT NULL:", q("SELECT COUNT(*) FROM experiences WHERE next_state IS NOT NULL")[0][0])
    print("graded (reward_5m not null):", q("SELECT COUNT(*) FROM experiences WHERE reward_5m IS NOT NULL")[0][0])
    print("reward_5m stats (min,max,avg,std):", q("SELECT MIN(reward_5m),MAX(reward_5m),AVG(reward_5m),STDDEV(reward_5m) FROM experiences WHERE reward_5m IS NOT NULL")[0])
    print("reward_1h stats:", q("SELECT MIN(reward_1h),MAX(reward_1h),AVG(reward_1h),STDDEV(reward_1h) FROM experiences WHERE reward_1h IS NOT NULL")[0])
    print("reward_4h stats:", q("SELECT MIN(reward_4h),MAX(reward_4h),AVG(reward_4h),STDDEV(reward_4h) FROM experiences WHERE reward_4h IS NOT NULL")[0])
    print("actions:", q("SELECT action, COUNT(*) FROM experiences GROUP BY action ORDER BY 2 DESC LIMIT 10"))
    print("time range:", q("SELECT FROM_UNIXTIME(MIN(timestamp)), FROM_UNIXTIME(MAX(timestamp)) FROM experiences"))
    try:
        print("golden rows:", q("SELECT COUNT(*) FROM golden_experiences")[0][0])
    except Exception as ex:
        print("golden err", ex)
