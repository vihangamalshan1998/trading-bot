from sqlalchemy import create_engine, text
from core.config.settings import settings

e = create_engine(settings.database_url)
with e.connect() as c:
    row = c.execute(text('SELECT id, symbol, action, entry_price, reward_5m FROM experiences WHERE action="HOLD" AND reward_5m IS NOT NULL LIMIT 5')).fetchall()
    print("Graded HOLD rows sample:")
    for r in row:
        print("  ", r)
