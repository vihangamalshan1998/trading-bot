from sqlalchemy import create_engine, text
from core.config.settings import settings
import json

e = create_engine(settings.database_url)
with e.connect() as c:
    row = c.execute(text('SELECT id, symbol, action, entry_price, market_state, reward, reward_5m, reward_1h, reward_4h FROM experiences WHERE id=66148')).fetchone()
    print('Row:', row[0], row[1], row[2], 'entry_price:', row[3], 'reward:', row[5], 'r5m:', row[6], 'r1h:', row[7], 'r4h:', row[8])
    if row[4]:
        ms = json.loads(row[4]) if isinstance(row[4], str) else row[4]
        print('mid_price in market_state:', ms.get('mid_price') if isinstance(ms, dict) else type(ms))

    # Also check what the prices of future rows were around that timestamp
    ts_row = c.execute(text('SELECT timestamp FROM experiences WHERE id=66148')).fetchone()[0]
    future = c.execute(text(f'SELECT id, timestamp, entry_price, market_state FROM experiences WHERE symbol="NEARUSDT" AND timestamp >= {ts_row} AND timestamp <= {ts_row + 15000} ORDER BY timestamp ASC LIMIT 5')).fetchall()
    print('\nFuture rows around ts:')
    for f in future:
        f_ms = json.loads(f[3]) if isinstance(f[3], str) else f[3]
        px = f_ms.get('mid_price') if isinstance(f_ms, dict) else 'none'
        print('  f_id:', f[0], 'delta_s:', f[1] - ts_row, 'entry_px:', f[2], 'mid_px:', px)
