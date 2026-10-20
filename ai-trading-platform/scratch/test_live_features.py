import asyncio, json, sys, torch
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry
from core.db.redis import redis_manager

async def main():
    await redis_manager.connect()
    r = redis_manager.redis
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "NEARUSDT", "APTUSDT"]
    
    print("\n=== TESTING INFERENCE WITH LOADED MODEL ===")
    model = SingleSymbolActorCritic(input_dim=200)
    registry = ModelRegistry()
    try:
        model = registry.load_model(model)
        print("Model loaded successfully via ModelRegistry!")
    except Exception as e:
        print(f"Error loading model: {e}")
    model.eval()

    print("\nModel actor_fc weights norm:", model.actor_fc.weight.norm().item())
    print("Model actor_fc bias:", model.actor_fc.bias.data)
    print("Model critic_fc bias:", model.critic_fc.bias.data)

    z = torch.zeros(1, 30, 200)
    with torch.no_grad():
        az_raw = model.actor_fc(model.lstm(model.shared_norm(z))[0][:, -1, :])
        az = torch.tanh(az_raw)
        cz = model.critic_fc(model.lstm(model.shared_norm(z))[0][:, -1, :])
    print(f"\n[ALL ZEROS INPUT]")
    print(f"  actor_raw: {az_raw[0].numpy()}")
    print(f"  actor_out: {az[0].numpy()}")
    print(f"  critic_out: {cz[0].numpy()}")
    print(f"  conf: {1.0 / (1.0 + torch.exp(-az[0][1])):.4f}")
    print(f"  target_size: {1.0 / (1.0 + torch.exp(-az[0][2])):.4f}")


    for sym in symbols:
        raw_m = await r.get(f"market:state:{sym}")
        if not raw_m:
            continue
        m = json.loads(raw_m)
        feats = m.get("features", [])
        
        portfolio_part = [5000.0, 5000.0, 5000.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        pos_part = [0.0] * 10
        mkt_part = feats if len(feats) == 90 else [0.0] * 90
        macro_part = [0.0] * 20
        pad_part = [0.0] * 70
        full_vec = portfolio_part + pos_part + mkt_part + macro_part + pad_part
        t = torch.tensor([[full_vec] * 30], dtype=torch.float32)
        
        with torch.no_grad():
            x = model.shared_norm(t)
            lstm_out, _ = model.lstm(x)
            last_out = lstm_out[:, -1, :]
            actor_raw = model.actor_fc(last_out)
            actor_out = torch.tanh(actor_raw)
            critic_out = model.critic_fc(last_out)
        
        print(f"\n[{sym}]")
        print(f"  actor_raw (before tanh): {actor_raw[0].numpy()}")
        print(f"  actor_out (after tanh):  {actor_out[0].numpy()}")
        print(f"  critic_out:              {critic_out[0].numpy()}")
        conf = 1.0 / (1.0 + torch.exp(-actor_out[0][1]))
        target_size = 1.0 / (1.0 + torch.exp(-actor_out[0][2]))
        print(f"  Calculated Conf: {conf:.4f}, Target Size: {target_size:.4f}")

if __name__ == '__main__':
    asyncio.run(main())
