import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch, redis, json, numpy as np
from apps.trading_bot.main import ProductionTradingBot
from core.schemas.state_schema import MarketState
from core.config.settings import settings

bot = ProductionTradingBot(symbols=settings.symbol_universe)
r = redis.Redis(decode_responses=True)

for sym in ["DOGEUSDT", "1000PEPEUSDT", "WIFUSDT", "BTCUSDT", "NEARUSDT", "ADAUSDT", "XRPUSDT"]:
    raw = r.get(f"market:state:{sym}")
    if raw:
        data = json.loads(raw)
        bot.market_states[sym] = MarketState(**data)
        sv, mf = bot._build_state_vector(sym)
        seq = [sv.squeeze(0).tolist()] * 300
        st = torch.tensor([seq], dtype=torch.float32)
        with torch.no_grad():
            act, crit = bot.model(st)
        act_l = act[0].numpy()
        crit_l = crit[0].numpy()
        raw_action = float(act_l[0])
        raw_conf = float((act_l[1] + 1.0) / 2.0)
        c5m = float(crit_l[0])
        
        # Directional Fusion:
        critic_signal = float(np.clip(c5m / 1.5, -0.80, 0.80))
        actor_signal = float(np.clip(raw_action - 0.50, -0.50, 0.50))
        
        if c5m <= -0.15:
            action_val = float(np.clip(critic_signal + min(0.0, actor_signal), -1.0, -0.25))
        elif c5m >= 0.15:
            action_val = float(np.clip(critic_signal + max(0.0, actor_signal), 0.25, 1.0))
        else:
            action_val = float(np.clip(actor_signal + critic_signal, -1.0, 1.0))
            
        side = "SHORT" if action_val < -0.20 else ("LONG" if action_val > 0.20 else "HOLD")
        print(f"[{sym:12s}] raw_act={raw_action:+.2f}, Critic 5m={c5m:+.3f}% => action_val={action_val:+.4f} | SIDE: {side}")
