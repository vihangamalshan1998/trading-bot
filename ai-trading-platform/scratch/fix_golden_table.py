from sqlalchemy import create_engine, text
from core.config.settings import settings

e = create_engine(settings.database_url)
with e.connect() as c:
    c.execute(text('CREATE TABLE IF NOT EXISTS golden_experiences LIKE experiences;'))
    c.commit()
    tables = c.execute(text('SHOW TABLES LIKE "golden_experiences";')).fetchall()
    print('golden_experiences table status:', tables)

    # Clean the 9 extreme negative outlier rows from Oct 16 legacy formula
    res = c.execute(text('UPDATE experiences SET reward_5m = -5.0 WHERE reward_5m < -5.0;'))
    res2 = c.execute(text('UPDATE experiences SET reward_1h = -15.0 WHERE reward_1h < -15.0;'))
    res3 = c.execute(text('UPDATE experiences SET reward_4h = -25.0 WHERE reward_4h < -25.0;'))
    c.commit()
    print('Cleaned outlier rewards from legacy runs.')
