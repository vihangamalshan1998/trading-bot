import sys, torch
sys.path.insert(0, '.')
from core.ai.replay_buffer import ReplayBuffer

rb = ReplayBuffer(capacity_cache=1000)
# Force load from MySQL
with rb.SessionLocal() as session:
    from core.database.models.ai import Experience
    exps = session.query(Experience).order_by(Experience.timestamp.desc()).limit(100).all()

print(f"Loaded {len(exps)} experiences from DB.")
states, actions, rewards, next_states, dones = rb.build_tensors(exps, seq_len=300)

print("\n--- ACTIONS TENSOR INSPECTION ---")
print(f"Actions shape: {actions.shape}")
print(f"Action dims: [dir, conf, size, offset]")
print(f"Mean across batch: {actions.mean(dim=0).numpy()}")
print(f"Min across batch:  {actions.min(dim=0).values.numpy()}")
print(f"Max across batch:  {actions.max(dim=0).values.numpy()}")
print("\nFirst 10 action rows:")
for i in range(min(10, len(actions))):
    print(f"Row {i:2d}: {actions[i].numpy()}")
