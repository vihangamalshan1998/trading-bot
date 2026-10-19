import asyncio
import json
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.db.redis import redis_manager
from core.config.settings import settings
from core.schemas.state_schema import MarketState
from apps.research.model import SingleSymbolActorCritic
from core.ai.institutional_alpha import InstitutionalAlphaEngine

async def diagnose_all_coins():
    await redis_manager.connect()
    r = redis_manager.redis
    if not r:
        print("ERROR: Redis connection failed!")
        return

    symbols = settings.symbol_universe
    print(f"Diagnosing all {len(symbols)} coins at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}...")

    # Load Model
    model = SingleSymbolActorCritic(input_dim=200)
    checkpoint_path = "models/production/model_v1.pt"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("Loaded production model checkpoint: models/production/model_v1.pt")
    else:
        print("WARNING: models/production/model_v1.pt does not exist!")
    model.eval()

    alpha_engine = InstitutionalAlphaEngine(symbols=symbols)

    market_states = {}
    whale_states = {}

    for sym in symbols:
        raw_m = await r.get(f"market:state:{sym}")
        if raw_m:
            try:
                m_data = json.loads(raw_m)
                market_states[sym] = MarketState(**m_data)
                alpha_engine.update_tick(sym, market_states[sym])
            except Exception as e:
                print(f"Error parsing market state for {sym}: {e}")

        raw_w = await r.get(f"whale:state:{sym}")
        if raw_w:
            try:
                whale_states[sym] = json.loads(raw_w)
            except Exception:
                pass

    print(f"Retrieved {len(market_states)}/{len(symbols)} live market states from Redis.\n")

    # Current Session Hurdle
    current_hour_utc = time.gmtime().tm_hour
    weekday = time.gmtime().tm_wday
    if weekday in (5, 6):
        session_name = "WEEKEND"
        hurdle = settings.conf_hurdle_weekend
    elif 13 <= current_hour_utc < 21:
        session_name = "US_NY_PEAK"
        hurdle = settings.conf_hurdle_ny_peak
    elif 8 <= current_hour_utc < 13:
        session_name = "LONDON"
        hurdle = settings.conf_hurdle_london
    else:
        session_name = "ASIA_OFFPEAK"
        hurdle = settings.conf_hurdle_asia

    print(f"Active Market Session: {session_name} (Current UTC Hour: {current_hour_utc:02d}:00)")
    print(f"Confidence Hurdle Required for Entry: {hurdle:.2f}")
    print(f"Expected Return Hurdle Required: {settings.min_open_expected_return_pct:.2f}%\n")

    print(f"{'SYMBOL':<12} | {'ACTION_VAL':<10} | {'RAW_CONF':<8} | {'CRITIC_5M':<10} | {'PASSES_AI?':<10} | {'BLOCKING REASON'}")
    print("-" * 80)

    for sym in symbols:
        if sym not in market_states:
            print(f"{sym:<12} | {'NO DATA':<10} | {'--':<8} | {'--':<10} | {'NO':<10} | No market state in Redis")
            continue

        if sym in getattr(settings, 'excluded_trading_symbols', []):
            print(f"{sym:<12} | {'EXCLUDED':<10} | {'--':<8} | {'--':<10} | {'NO':<10} | In excluded_trading_symbols (SUI)")
            continue

        m = market_states[sym]
        
        # Build dummy sequence with current features
        # Note: In production, the bot uses a 300-step sequence. We evaluate current tick.
        # SingleSymbolActorCritic takes (1, seq_len, 200). We build a dummy observation.
        # Let's inspect features length
        features = m.features
        if len(features) != 90:
            print(f"{sym:<12} | {'BAD FEAT':<10} | {'--':<8} | {'--':<10} | {'NO':<10} | Features length = {len(features)} != 90")
            continue

        # Build 200-dim observation
        # 10 dims portfolio + 10 dims position + 90 market + 90 macro/statarb/orderflow = 200 dims
        obs = [0.0] * 200
        # Put 90 market features
        obs[20:110] = features

        tensor_in = torch.tensor([[obs]], dtype=torch.float32)
        with torch.no_grad():
            act_logits_t, exp_ret_t = model(tensor_in)
            act_logits = act_logits_t[0].numpy()
            exp_ret = exp_ret_t[0].numpy()

        raw_act = float(act_logits[0])
        raw_conf = float(1.0 / (1.0 + np.exp(-act_logits[1])))
        critic_5m = float(exp_ret[0])

        # Check conditions
        passes_action = abs(raw_act) > 0.20
        passes_conf = raw_conf >= hurdle
        passes_critic = abs(critic_5m) >= settings.min_open_expected_return_pct

        side_str = "LONG" if raw_act > 0.20 else ("SHORT" if raw_act < -0.20 else "HOLD")

        reasons = []
        if not passes_action:
            reasons.append(f"AI Action in HOLD deadband ({raw_act:+.2f} between -0.20 and +0.20)")
        if not passes_conf:
            reasons.append(f"Confidence {raw_conf:.2f} < {hurdle:.2f} ({session_name} Hurdle)")
        if not passes_critic:
            reasons.append(f"Critic 5m {critic_5m:+.2f}% < {settings.min_open_expected_return_pct:.2f}%")

        # Check alpha pillars if AI wanted to trade
        if passes_action and passes_conf and passes_critic:
            # Pillar 1
            p1_ok, p1_r = alpha_engine.evaluate_stop_hunt(sym, m, side_str)
            if not p1_ok:
                reasons.append(f"Pillar 1: {p1_r}")
            # Pillar 2
            p2_ok, _, p2_r = alpha_engine.evaluate_oi_divergence(sym, m, side_str)
            if not p2_ok:
                reasons.append(f"Pillar 2: {p2_r}")
            # Pillar 3
            p3_ok, p3_r = alpha_engine.evaluate_btc_lead_lag(sym, market_states, side_str)
            if not p3_ok:
                reasons.append(f"Pillar 3: {p3_r}")
            # Pillar 6
            p6_ok, r_name, p6_r = alpha_engine.classify_and_filter_regime(sym, m, "NEUTRAL", side_str)
            if not p6_ok:
                reasons.append(f"Pillar 6: {p6_r}")
            # Pillar 7
            p7_ok, _, p7_r = alpha_engine.evaluate_market_breadth(market_states, side_str)
            if not p7_ok:
                reasons.append(f"Pillar 7: {p7_r}")
            # Pillar 8
            p8_ok, _, p8_r = alpha_engine.evaluate_market_wide_cvd(sym, m, whale_states, side_str)
            if not p8_ok:
                reasons.append(f"Pillar 8: {p8_r}")

        passes_all = len(reasons) == 0
        status_str = "YES" if passes_all else "NO"
        reason_str = "CLEAR FOR ENTRY!" if passes_all else "; ".join(reasons)

        print(f"{sym:<12} | {raw_act:>+9.2f}  | {raw_conf:>7.2f}  | {critic_5m:>+9.2f}% | {status_str:<10} | {reason_str}")

if __name__ == "__main__":
    asyncio.run(diagnose_all_coins())
