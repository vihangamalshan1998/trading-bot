import asyncio, sys, torch, json
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry
from core.db.redis import redis_manager

async def main():
    await redis_manager.connect()
    r = redis_manager.redis
    model = SingleSymbolActorCritic(input_dim=200)
    registry = ModelRegistry()
    model = registry.load_model(model)
    model.eval()

    print("=== TESTING EFFECT OF SCALING PORTFOLIO INPUTS ===")
    
    # Let's get real live market features from Redis for 4 coins
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT"]:
        raw_m = await r.get(f"market:state:{sym}")
        if not raw_m: continue
        m = json.loads(raw_m)
        feats = m.get("features", [0.0]*90)
        
        # Test Case A: UN-NORMALIZED ($5,000 cash in vector)
        obs_unnorm = [5000.0, 5000.0, 5000.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0]*10 + feats + [0.0]*90
        t_unnorm = torch.tensor([[obs_unnorm] * 30], dtype=torch.float32)
        with torch.no_grad():
            a_unnorm, c_unnorm = model(t_unnorm)
            
        # Test Case B: NORMALIZED (percentages / ratios)
        # wallet growth=0.0, drawdown=0.0, margin_util=0.0, etc.
        obs_norm = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0]*10 + feats + [0.0]*90
        t_norm = torch.tensor([[obs_norm] * 30], dtype=torch.float32)
        with torch.no_grad():
            a_norm, c_norm = model(t_norm)

        print(f"\n--- {sym} ---")
        print(f"UN-NORMALIZED ($5000 input):")
        print(f"  act raw: {a_unnorm[0].numpy()}")
        print(f"  conf unscaled: {(a_unnorm[0][1].item() + 1.0)/2.0:.4f}")
        print(f"  size unscaled: {(a_unnorm[0][2].item() + 1.0)/2.0:.4f}")
        print(f"  critic: {c_unnorm[0].numpy()}")

        print(f"NORMALIZED (0.0-1.0 input):")
        print(f"  act raw: {a_norm[0].numpy()}")
        print(f"  conf unscaled: {(a_norm[0][1].item() + 1.0)/2.0:.4f}")
        print(f"  size unscaled: {(a_norm[0][2].item() + 1.0)/2.0:.4f}")
        print(f"  critic: {c_norm[0].numpy()}")

if __name__ == '__main__':
    asyncio.run(main())
