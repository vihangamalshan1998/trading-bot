import torch, sys, json
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic
from core.db.redis import redis_manager
from core.config.settings import settings
import asyncio

async def main():
    await redis_manager.connect()
    r = redis_manager.redis

    # Load current model (v1_push_89)
    model_curr = SingleSymbolActorCritic(input_dim=200)
    sd_curr = torch.load("models/production/model_v1.pt", map_location="cpu", weights_only=True)
    if "model_state_dict" in sd_curr:
        model_curr.load_state_dict(sd_curr["model_state_dict"])
    else:
        model_curr.load_state_dict(sd_curr)
    model_curr.eval()

    # Load backup model (v1_push_88)
    model_prev = SingleSymbolActorCritic(input_dim=200)
    sd_prev = torch.load("models/production/model_v1_backup.pt", map_location="cpu", weights_only=True)
    if "model_state_dict" in sd_prev:
        model_prev.load_state_dict(sd_prev["model_state_dict"])
    else:
        model_prev.load_state_dict(sd_prev)
    model_prev.eval()

    print("=== MODEL WEIGHTS COMPARISON ===")
    print("CURRENT (v1_push_89):")
    print("  actor_fc.weight norm:", model_curr.actor_fc.weight.norm().item())
    print("  actor_fc.bias:", model_curr.actor_fc.bias.detach().numpy())
    print("PREVIOUS (v1_push_88 backup):")
    print("  actor_fc.weight norm:", model_prev.actor_fc.weight.norm().item())
    print("  actor_fc.bias:", model_prev.actor_fc.bias.detach().numpy())

    print("\n=== LIVE SYMBOL INFERENCE COMPARISON ===")
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "DOGEUSDT"]:
        raw_m = await r.get(f"market:state:{sym}")
        if not raw_m:
            continue
        m = json.loads(raw_m)
        feats = m.get("features", [])
        if len(feats) != 90:
            continue
        obs = [5000.0, 5000.0, 5000.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0]*10 + feats + [0.0]*20 + [0.0]*70
        t = torch.tensor([[obs] * 300], dtype=torch.float32)
        with torch.no_grad():
            act_c, crit_c = model_curr(t)
            act_p, crit_p = model_prev(t)
        
        ac = act_c[0].numpy()
        ap = act_p[0].numpy()
        print(f"[{sym:8s}]")
        print(f"  CURRENT (v89):  act={ac[0]:+.3f}, conf={ac[1]:+.3f}, size={ac[2]:+.3f} | critic={crit_c[0].numpy()}")
        print(f"  BACKUP  (v88):  act={ap[0]:+.3f}, conf={ap[1]:+.3f}, size={ap[2]:+.3f} | critic={crit_p[0].numpy()}")

if __name__ == '__main__':
    asyncio.run(main())
