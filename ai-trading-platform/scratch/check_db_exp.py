import sys
sys.path.insert(0, '.')
from core.ai.replay_buffer import ReplayBuffer
from core.database.models.ai import Experience

rb = ReplayBuffer(capacity_cache=0)
with rb.SessionLocal() as session:
    e = session.query(Experience).order_by(Experience.timestamp.desc()).first()
    if e:
        print("Latest Exp Symbol:", e.symbol, "Action:", e.action)
        f = e.market_state[0] if (isinstance(e.market_state, list) and len(e.market_state) > 0 and isinstance(e.market_state[0], list)) else e.market_state
        if isinstance(f, list):
            print("Dims 0-9 (Portfolio):", [round(x, 2) for x in f[:10]])
            print("Dims 10-19 (Position):", [round(x, 2) for x in f[10:20]])
            print("Dims 20-25 (Market features):", f[20:25])
            print("Total dims:", len(f))
    else:
        print("No experiences found.")
