import sys
sys.path.insert(0, '.')
from core.ai.replay_buffer import ReplayBuffer
from core.database.models.ai import Experience

rb = ReplayBuffer(capacity_cache=0)
with rb.SessionLocal() as session:
    exps = session.query(Experience).order_by(Experience.timestamp.desc()).limit(20).all()
    for e in exps:
        print(f"{e.symbol:10s} action={e.action:10s} conf={e.confidence} act_prob={e.action_probability}")
