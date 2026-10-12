import torch
import numpy as np
from apps.research.model import SingleSymbolActorCritic

model = SingleSymbolActorCritic(input_dim=200)
state_dict = torch.load("models/production/model_v1.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Test with 10 random samples and zeros
with torch.no_grad():
    zeros = torch.zeros(1, 10, 200)
    out_zeros, _ = model(zeros)
    z_logits = out_zeros[0].numpy()
    
    print("Zeros Input:")
    print("  action_val:", z_logits[0])
    print("  raw_conf_logit:", z_logits[1], "-> confidence:", (z_logits[1] + 1.0) / 2.0)
    print("  raw_size_logit:", z_logits[2], "-> target_size:", (z_logits[2] + 1.0) / 2.0)
    
    print("\nRandom 5 inputs:")
    for i in range(5):
        rand = torch.randn(1, 10, 200)
        out_rand, _ = model(rand)
        r_logits = out_rand[0].numpy()
        conf = (r_logits[1] + 1.0) / 2.0
        size = (r_logits[2] + 1.0) / 2.0
        print(f"  Sample {i}: action={r_logits[0]:.3f}, raw_conf={r_logits[1]:.3f} (conf={conf*100:.1f}%), size={size*100:.1f}%")
