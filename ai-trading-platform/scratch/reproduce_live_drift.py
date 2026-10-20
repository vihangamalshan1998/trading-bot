import asyncio
import time
import torch
import numpy as np
import collections
from apps.trading_bot.main import ProductionTradingBot

async def reproduce():
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "1000PEPEUSDT", "NEARUSDT", "AVAXUSDT", "LINKUSDT",
        "LTCUSDT", "APTUSDT", "RENDERUSDT", "TAOUSDT", "INJUSDT", "WIFUSDT"
    ]
    bot = ProductionTradingBot(symbols=symbols)
    
    # Load market states from Redis
    from core.database.models.market import MarketState
    import json
    for sym in symbols:
        raw = await bot.redis.redis.get(f"market:state:{sym}")
        if raw:
            bot.market_states[sym] = MarketState(**json.loads(raw))
            
    print(f"Loaded market states for {len(bot.market_states)} symbols.")
    
    # Run 50 ticks of the EXACT batch inference logic from main.py lines 890-950
    for tick in range(1, 51):
        batch_tensors = []
        batch_syms = []
        precomputed_seqs = {}
        precomputed_actions = {}
        precomputed_critics = {}
        
        for sym in bot.symbols:
            if sym not in bot.market_states:
                continue
            sv, mf = bot._build_state_vector(sym)
            sv_list = sv.squeeze(0).tolist()
            seq = list(bot.state_history[sym]) + [sv_list]
            while len(seq) < bot.sequence_length:
                seq.insert(0, seq[0])
            seq = seq[-bot.sequence_length:]
            
            st = torch.tensor(seq, dtype=torch.float32)
            if st.shape == (bot.sequence_length, 200):
                batch_tensors.append(st)
                batch_syms.append(sym)
                precomputed_seqs[sym] = (sv, seq)
        
        if batch_tensors:
            final_batch = torch.stack(batch_tensors)
            final_batch = torch.nan_to_num(final_batch, nan=0.0, posinf=1.0, neginf=-1.0)
            
            h_list, c_list = [], []
            has_hidden = False
            hidden_dim = 256
            for sym in batch_syms:
                h = bot.hidden_states.get(sym)
                if h is not None:
                    h_list.append(h[0])
                    c_list.append(h[1])
                    has_hidden = True
                else:
                    h_list.append(torch.zeros(1, 1, hidden_dim))
                    c_list.append(torch.zeros(1, 1, hidden_dim))
            
            batch_hidden = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1)) if has_hidden else None
            
            with torch.no_grad():
                batch_logits, batch_critic, new_hidden_batch = bot.model(final_batch, hidden_state=batch_hidden, return_hidden=True)
                
            for i, sym in enumerate(batch_syms):
                precomputed_actions[sym] = batch_logits[i].numpy()
                precomputed_critics[sym] = batch_critic[i].numpy()
                bot.hidden_states[sym] = (new_hidden_batch[0][:, i:i+1, :].detach(), new_hidden_batch[1][:, i:i+1, :].detach())
                
                # Update state_history as main loop does
                state_vector, seq = precomputed_seqs[sym]
                bot.state_history[sym].append(state_vector.squeeze(0).tolist())
                
                raw_act = float(precomputed_actions[sym][0])
                raw_conf = float(np.clip((precomputed_actions[sym][1] + 1.0) / 2.0, 0.05, 0.99))
                bot.smoothed_action_history[sym].append(raw_act)
                bot.smoothed_confidence_history[sym].append(raw_conf)
                
        if tick in [1, 2, 5, 10, 20, 30, 50]:
            print(f"\n--- TICK {tick} ---")
            for sym in ['BTCUSDT', 'DOGEUSDT', '1000PEPEUSDT']:
                crit = precomputed_critics[sym][0]
                crit1h = precomputed_critics[sym][1]
                conf = np.mean(bot.smoothed_confidence_history[sym])
                act = np.mean(bot.smoothed_action_history[sym])
                side = "LONG" if act > 0.2 else ("SHORT" if act < -0.2 else "HOLD")
                print(f"  [{sym:12s}] Side={side:5s} | Conf={conf*100:5.1f}% | Critic 5m={crit:+.5f}% | 1h={crit1h:+.5f}%")
                
    await bot.binance.close()
    await bot.redis.redis.close()

if __name__ == "__main__":
    asyncio.run(reproduce())
