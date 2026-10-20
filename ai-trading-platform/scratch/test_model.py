import torch, sys
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry

reg = ModelRegistry()
model = SingleSymbolActorCritic(input_dim=200)
model = reg.load_model(model)
model.eval()

t1 = torch.randn(1, 20, 200)
t2 = torch.randn(1, 20, 200) * 5.0
t_zeros = torch.zeros(1, 20, 200)

with torch.no_grad():
    res1, crit1 = model(t1)
    res2, crit2 = model(t2)
    res0, crit0 = model(t_zeros)

print("--- Random 1 ---")
print("Action Logits:", res1.numpy())
print("Critic:", crit1.numpy())

print("--- Random 2 (Scaled x5) ---")
print("Action Logits:", res2.numpy())
print("Critic:", crit2.numpy())

print("--- Zeros ---")
print("Action Logits:", res0.numpy())
print("Critic:", crit0.numpy())
