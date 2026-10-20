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

    print("=== TESTING TEMPERATURE-SCALED ACTOR ON ALL 18 SYMBOLS ===")
    
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
            
            # Direction: full tanh
            act_dir = float(torch.tanh(torch.tensor(actor_raw[0])))
            
            # Rest: Temperature-scaled (T=6.0)
            act_conf = float(torch.tanh(torch.tensor(actor_raw[1] / 6.0)))
            act_size = float(torch.tanh(torch.tensor(actor_raw[2] / 6.0)))
            
            # Unscale to live dashboard values
            conf_pct = float(torch.clamp(torch.tensor((act_conf + 1.0) / 2.0), 0.05, 0.99)) * 100.0
            size_pct = float(torch.clamp(torch.tensor((act_size + 1.0) / 2.0), 0.25, 1.0)) * 100.0
            
            side = "LONG" if act_dir > 0.20 else ("SHORT" if act_dir < -0.20 else "HOLD")
            
        print(f"[{sym:12s}] Side: {side:5s} (dir={act_dir:+.2f}) | Conf: {conf_pct:5.1f}% | Size: {size_pct:5.1f}% | raw_conf={actor_raw[1]:+5.2f}, raw_size={actor_raw[2]:+5.2f}")

if __name__ == '__main__':
    asyncio.run(main())
