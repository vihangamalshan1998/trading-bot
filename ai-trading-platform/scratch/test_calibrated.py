import asyncio, sys, torch, json
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry
from core.db.redis import redis_manager
from core.config.settings import settings

async def main():
    await redis_manager.connect()
    r = redis_manager.redis
    model = SingleSymbolActorCritic(input_dim=200)
    registry = ModelRegistry()
    model = registry.load_model(model)
    model.eval()

    print("=== LIVE CALIBRATION TEST ACROSS ALL 18 SYMBOLS ===")
    
    for sym in settings.symbol_universe:
        raw_m = await r.get(f"market:state:{sym}")
        if not raw_m: continue
        m = json.loads(raw_m)
        feats = m.get("features", [])
        if len(feats) != 90: continue
        
        obs = [5000.0, 5000.0, 5000.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0]*10 + feats + [0.0]*20 + [0.0]*70
        t = torch.tensor([[obs] * 300], dtype=torch.float32)
        
        with torch.no_grad():
            last_out = model.lstm(model.shared_norm(t))[0][:, -1, :]
            actor_raw = model.actor_fc(last_out)[0].numpy()
            critic = model.critic_fc(last_out)[0].numpy()
            
            # Direction: full tanh
            act_dir = float(torch.tanh(torch.tensor(actor_raw[0])))
            
            # Confidence: calibrated so baseline quiet is ~30-40%, breakouts are 75-92%
            # Subtracting the idle offset (3.0) and scaling with temperature:
            calibrated_conf_logit = (actor_raw[1] - 3.0) / 4.0
            conf_val = float(torch.clamp(torch.tensor((torch.tanh(torch.tensor(calibrated_conf_logit)) + 1.0) / 2.0), 0.05, 0.99))
            
            # Size: calibrated from size logit
            calibrated_size_logit = (actor_raw[2] + 4.0) / 4.0
            size_val = float(torch.clamp(torch.tensor((torch.tanh(torch.tensor(calibrated_size_logit)) + 1.0) / 2.0), 0.25, 1.0))
            
            side = "LONG" if act_dir > 0.20 else ("SHORT" if act_dir < -0.20 else "HOLD")
            
        print(f"[{sym:12s}] Decision: {side:5s} | Conf: {conf_val*100:5.1f}% | Size: {size_val*100:5.1f}% | Critic 5m: {critic[0]:+.3f}% | raw_conf={actor_raw[1]:+5.2f}, raw_size={actor_raw[2]:+5.2f}")

if __name__ == '__main__':
    asyncio.run(main())
