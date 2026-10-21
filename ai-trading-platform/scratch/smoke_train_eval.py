"""
Read-only smoke test of the market-return trainer on real data.
Trains a COPY of the production champion for a few steps and runs the holdout tournament.
Never promotes, never writes to models/production, never pushes to the VPS.

usage: python scratch/smoke_train_eval.py [limit] [steps]
"""
import os
import sys
import time
sys.path.insert(0, os.path.abspath("."))
os.environ["LOCAL_MODE"] = "True"

import torch
from apps.trainer.ppo import PPOTrainer
from apps.research.model import SingleSymbolActorCritic

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 800
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
torch.manual_seed(0)
torch.set_num_threads(2)

model = SingleSymbolActorCritic(input_dim=200)
model.load_state_dict(torch.load("models/production/model_v1.pt", map_location="cpu"))

t0 = time.time()
trainer = PPOTrainer(model=model, limit=limit)
print(f"cache: {len(trainer.buffer.cache)} train / {len(trainer.buffer.holdout_cache)} holdout ({time.time() - t0:.0f}s)")

print("\n=== BEFORE training (challenger == champion) ===")
before = trainer.evaluate_candidate(sample_size=1000)
for k in ("reason", "champion_sim_pnl", "champion_trades", "champion_win_rate", "champion_critic_skill", "sample_size"):
    print(f"  {k}: {before.get(k)}")

for i in range(steps):
    trainer.train_step(batch_size=64, epochs=2)

print(f"\n=== AFTER {steps} steps ===")
after = trainer.evaluate_candidate(sample_size=1000)
for k in ("approved", "reason", "challenger_sim_pnl", "champion_sim_pnl", "challenger_trades", "champion_trades",
          "challenger_win_rate", "champion_win_rate", "challenger_critic_skill", "champion_critic_skill", "sample_size"):
    print(f"  {k}: {after.get(k)}")
