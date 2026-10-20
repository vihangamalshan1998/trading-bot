import sys
import torch
import numpy as np
from apps.trading_bot.main import TradingBot
from apps.research.model import SingleSymbolActorCritic

bot = TradingBot()
bot.registry.load_active_model()
print("TradingBot instantiated.")

for sym in ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'RENDERUSDT']:
    try:
        sv, mf = bot._build_state_vector(sym)
        arr = sv.squeeze(0).numpy()
        print(f"[{sym}] State Vector Shape: {sv.shape} | min: {np.min(arr):.4f} | max: {np.max(arr):.4f} | mean: {np.mean(arr):.4f} | std: {np.std(arr):.4f}")
        top5 = np.argsort(np.abs(arr))[-5:][::-1]
        print(f"    Top 5 slots: " + ", ".join([f"slot {i} = {arr[i]:.2f}" for i in top5]))
    except Exception as e:
        print(f"[{sym}] Error: {e}")
