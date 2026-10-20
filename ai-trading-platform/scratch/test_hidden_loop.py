import sys, torch, json, time
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic
from core.ai.registry import ModelRegistry
from core.db.redis import redis_manager

async def test_hidden_loop():
    await redis_manager.connect()
    r = redis_manager.redis
    model = SingleSymbolActorCritic(input_dim=200)
    registry = ModelRegistry()
    model = registry.load_model(model)
    model.eval()

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT"]
    hidden_states = {}
    state_histories = {s: [] for s in symbols}

    print("=== SIMULATING TICKS WITH HIDDEN STATE RETENTION ===")
    for tick in range(1, 21):
        batch_tensors = []
        batch_syms = []
        for sym in symbols:
            raw_m = await r.get(f"market:state:{sym}")
            m = json.loads(raw_m) if raw_m else {}
            feats = m.get("features", [0.0] * 90)
            if len(feats) != 90: feats = [0.0] * 90
            
            # Recreate _build_state_vector
            obs = [5000.0, 5000.0, 5000.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0] # 10
            obs += [0.0] * 10 # 10 pos
            obs += feats # 90 market
            obs += [0.0] * 20 # 20 macro
            obs += [0.0] * 70 # 70 pad
            
            state_histories[sym].append(obs)
            seq = list(state_histories[sym])
            while len(seq) < 300:
                seq.insert(0, seq[0])
            seq = seq[-300:]
            batch_tensors.append(torch.tensor(seq, dtype=torch.float32))
            batch_syms.append(sym)
        
        final_batch = torch.stack(batch_tensors)
        final_batch = torch.nan_to_num(final_batch, nan=0.0, posinf=1.0, neginf=-1.0)
        
        h_list, c_list = [], []
        has_hidden = False
        hidden_dim = 256
        for sym in batch_syms:
            h = hidden_states.get(sym)
            if h is not None:
                h_list.append(h[0])
                c_list.append(h[1])
                has_hidden = True
            else:
                h_list.append(torch.zeros(1, 1, hidden_dim))
                c_list.append(torch.zeros(1, 1, hidden_dim))
        
        batch_hidden = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1)) if has_hidden else None

        with torch.no_grad():
            batch_logits, batch_critic, new_hidden_batch = model(final_batch, hidden_state=batch_hidden, return_hidden=True)

        for i, sym in enumerate(batch_syms):
            hidden_states[sym] = (new_hidden_batch[0][:, i:i+1, :].detach(), new_hidden_batch[1][:, i:i+1, :].detach())
            if tick in [1, 2, 5, 10, 20]:
                crit = batch_critic[i].numpy()
                act = batch_logits[i].numpy()
                conf = 1.0 / (1.0 + torch.exp(-torch.tensor(act[1])))
                print(f"Tick {tick} [{sym}]: critic={crit}, a[0]={act[0]:.3f}, a[1]={act[1]:.3f}, conf={conf:.4f}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(test_hidden_loop())
