import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
from apps.research.model import SingleSymbolActorCritic

m = SingleSymbolActorCritic(input_dim=200)
m.load_state_dict(torch.load('models/production/model_v1.pt', map_location='cpu'))
m.eval()

x = torch.zeros(1, 1, 200)
with torch.no_grad():
    a, c = m(x)
    print("Baseline Zero State Action:", [round(v, 4) for v in a.squeeze().tolist()])
    print("Baseline Zero State Critic:", [round(v, 4) for v in c.squeeze().tolist()])
