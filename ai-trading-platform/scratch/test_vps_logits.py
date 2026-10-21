import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch, redis, json
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry

registry = ModelRegistry()
model = SingleSymbolActorCritic(input_dim=200)
model = registry.load_model(model)
model.eval()

# Let's inspect weights of actor head!
print("Actor head final layer weight mean:", model.actor_fc.weight.mean().item())
print("Actor head final layer bias:", model.actor_fc.bias.tolist())

# Test with zero input (1, 20, 200)
x = torch.zeros(1, 20, 200)
with torch.no_grad():
    logits, critics = model(x)
print("Logits for zero input:", logits[0].tolist())
print("Critics for zero input:", critics[0].tolist())

# Test with random input
x_rand = torch.randn(1, 20, 200)
with torch.no_grad():
    logits_rand, critics_rand = model(x_rand)
print("Logits for rand input:", logits_rand[0].tolist())
print("Critics for rand input:", critics_rand[0].tolist())
