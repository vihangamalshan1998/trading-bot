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

    print("=== SCIENTIFIC PROOF: EFFECT OF $5000 CASH BALANCE VS NORMALIZED ===")
    
    test_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "DOGEUSDT"]
    
    print("\n--- TEST A: WITH $5,000 UNNORMALIZED CASH BALANCE (CURRENT STATE) ---")
    for sym in test_coins:
        raw_m = await r.get(f"market:state:{sym}")
        if not raw_m: continue
        m = json.loads(raw_m)
        feats = m.get("features", [])
        if len(feats) != 90: continue
        
        # $5000 in slots 0-2
        obs_5000 = [5000.0, 5000.0, 5000.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0]*10 + feats + [0.0]*20 + [0.0]*70
        t_5000 = torch.tensor([[obs_5000] * 300], dtype=torch.float32)
        with torch.no_grad():
            act, _ = model(t_5000)
            conf = (act[0][1].item() + 1.0) / 2.0 * 100.0
            size = (act[0][2].item() + 1.0) / 2.0 * 100.0
            size_clipped = max(25.0, min(100.0, size))
        print(f"[{sym:10s}] Raw Logit: {model.actor_fc(model.lstm(model.shared_norm(t_5000))[0][:, -1, :])[0][1].item():+5.2f} -> Conf: {conf:5.1f}% | Size: {size_clipped:5.1f}%")

    print("\n--- TEST B: WITH NORMALIZED CASH BALANCE (1.0 RATIO) ---")
    for sym in test_coins:
        raw_m = await r.get(f"market:state:{sym}")
        if not raw_m: continue
        m = json.loads(raw_m)
        feats = m.get("features", [])
        if len(feats) != 90: continue
        
        # 1.0 ratio in slots 0-2 (normalized)
        obs_norm = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] + [0.0]*10 + feats + [0.0]*20 + [0.0]*70
        t_norm = torch.tensor([[obs_norm] * 300], dtype=torch.float32)
        with torch.no_grad():
            act, _ = model(t_norm)
            conf = (act[0][1].item() + 1.0) / 2.0 * 100.0
            size = (act[0][2].item() + 1.0) / 2.0 * 100.0
            size_clipped = max(25.0, min(100.0, size))
        print(f"[{sym:10s}] Raw Logit: {model.actor_fc(model.lstm(model.shared_norm(t_norm))[0][:, -1, :])[0][1].item():+5.2f} -> Conf: {conf:5.1f}% | Size: {size_clipped:5.1f}%")

if __name__ == '__main__':
    asyncio.run(main())
