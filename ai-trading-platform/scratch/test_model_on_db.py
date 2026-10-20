import sys, torch
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry
from core.ai.replay_buffer import ReplayBuffer

m = SingleSymbolActorCritic(input_dim=200)
reg = ModelRegistry()
m = reg.load_model(m)
m.eval()

rb = ReplayBuffer(capacity_cache=100)
with rb.SessionLocal() as session:
    from core.database.models.ai import Experience
    exps = session.query(Experience).order_by(Experience.timestamp.desc()).limit(100).all()

states, actions, rewards, next_states, dones = rb.build_tensors(exps, seq_len=300)

with torch.no_grad():
    a_pred, c_pred = m(states)

print("=== MODEL OUTPUTS ON 100 DATABASE SAMPLES ===")
print("Ground truth actions (from DB):")
print(f"  Dir mean:  {actions[:, 0].mean().item():.3f} (min: {actions[:, 0].min().item():.3f}, max: {actions[:, 0].max().item():.3f})")
print(f"  Conf mean: {actions[:, 1].mean().item():.3f} (min: {actions[:, 1].min().item():.3f}, max: {actions[:, 1].max().item():.3f})")
print(f"  Size mean: {actions[:, 2].mean().item():.3f} (min: {actions[:, 2].min().item():.3f}, max: {actions[:, 2].max().item():.3f})")

print("\nModel predicted actions (a_pred):")
print(f"  Dir mean:  {a_pred[:, 0].mean().item():.3f} (min: {a_pred[:, 0].min().item():.3f}, max: {a_pred[:, 0].max().item():.3f})")
print(f"  Conf mean: {a_pred[:, 1].mean().item():.3f} (min: {a_pred[:, 1].min().item():.3f}, max: {a_pred[:, 1].max().item():.3f})")
print(f"  Size mean: {a_pred[:, 2].mean().item():.3f} (min: {a_pred[:, 2].min().item():.3f}, max: {a_pred[:, 2].max().item():.3f})")

print("\nFirst 5 predicted rows [dir, conf, size, offset]:")
for i in range(5):
    print(f"Sample {i}: Pred: {a_pred[i].numpy()} | True: {actions[i].numpy()}")
