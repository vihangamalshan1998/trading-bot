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

    print("=== LIVE SYMBOL EVALUATION ON LOADED PRODUCTION MODEL ===")
    
    for sym in settings.symbol_universe:
        raw_m = await r.get(f"market:state:{sym}")
        if not raw_m:
            continue
        m = json.loads(raw_m)
        feats = m.get("features", [])
        if len(feats) != 90:
            continue
            
        # Build live vector matching _build_state_vector
        obs = [5000.0, 5000.0, 5000.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] # 10
        obs += [0.0] * 10 # 10 pos
        obs += feats # 90 market
        obs += [0.0] * 20 # 20 macro
        obs += [0.0] * 70 # 70 pad
        
        t = torch.tensor([[obs] * 300], dtype=torch.float32)
        with torch.no_grad():
            last_out = model.lstm(model.shared_norm(t))[0][:, -1, :]
            actor_raw = model.actor_fc(last_out)[0].numpy()
            actor_out = torch.tanh(model.actor_fc(last_out))[0].numpy()
            critic = model.critic_fc(last_out)[0].numpy()
            
        print(f"[{sym:10s}]")
        print(f"   actor_raw (before tanh): [dir={actor_raw[0]:+6.2f}, conf={actor_raw[1]:+6.2f}, size={actor_raw[2]:+6.2f}, offset={actor_raw[3]:+6.2f}]")
        print(f"   actor_out (after tanh):  [dir={actor_out[0]:+6.3f}, conf={actor_out[1]:+6.3f}, size={actor_out[2]:+6.3f}, offset={actor_out[3]:+6.3f}]")
        print(f"   critic: [5m={critic[0]:+6.3f}%, 1h={critic[1]:+6.3f}%, 4h={critic[2]:+6.3f}%]")

if __name__ == '__main__':
    asyncio.run(main())
