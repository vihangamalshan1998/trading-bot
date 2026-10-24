# Run from ai-trading-platform: PYTHONPATH=. venv/Scripts/python apps/research/studies/exit_study.py [random|brain [rows]]
"""Read-only exit-rule study on mainnet history: current rules vs 4 variants.
Set 1: entries the live brain would take (replayed history rows, live entry rule + its trade policy).
Set 2: random entries (no edge), every 2 h, both directions, all coins, 90 days."""
import json, sys, time, random
from datetime import date, datetime, timedelta, timezone
import numpy as np, torch
from core.risk import exit_policy as ep
from apps.replay import archive
from core.config.settings import settings

COST = ep.ROUND_TRIP_COST
VARIANTS = {
    "A current rules": {},
    "B trail 0.35% after +0.5%": {"trail_act": 0.005, "trail_gap": 0.0035},
    "C stop->breakeven after +0.5%": {"be_act": 0.005},
    "D 30% partial at +0.6%": {"early": (0.006, 0.30)},
    "E D + breakeven after +0.6%": {"early": (0.006, 0.30), "be_act": 0.006},
}

def sim(direction, entry, bars, lv, v):
    rem, real, tp1_done, early_done, peak, held = 1.0, 0.0, False, False, 0.0, 0
    mc = 0.0
    for h, l, c in bars:
        held += 60
        if direction > 0:
            best, worst, mc = (h - entry) / entry, (l - entry) / entry, (c - entry) / entry
        else:
            best, worst, mc = (entry - l) / entry, (entry - h) / entry, (entry - c) / entry
        floor = max(lv.breakeven, peak - lv.trail_gap) if tp1_done else -lv.stop
        if "trail_act" in v and peak >= v["trail_act"]:
            floor = max(floor, peak - v["trail_gap"])
        if "be_act" in v and peak >= v["be_act"]:
            floor = max(floor, lv.breakeven)
        if worst <= floor:
            return real + rem * floor
        if best >= lv.hard_tp:
            return real + rem * lv.hard_tp
        if "early" in v and not early_done and best >= v["early"][0]:
            real += v["early"][1] * v["early"][0]; rem -= v["early"][1]; early_done = True
        if not tp1_done and best >= lv.tp1:
            real += rem * 0.5 * lv.tp1; rem *= 0.5; tp1_done = True
        peak = max(peak, best)
        if held >= ep.MAX_HOLD_SECONDS:
            break
    return real + rem * mc

_kl = {}
def klines(sym, day):
    key = (sym, day)
    if key not in _kl:
        _kl[key] = archive.load_klines("klines", sym, day)
    return _kl[key]

def path_for(sym, ts):
    d = datetime.fromtimestamp(ts, timezone.utc).date()
    parts = [klines(sym, d + timedelta(days=i)) for i in (-1, 0, 1)]
    if any(p is None for p in parts[:2]):
        return None
    k = np.concatenate([p for p in parts if p is not None])
    i = int(np.searchsorted(k[:, 0], ts * 1000.0, side="left"))
    if i < 1000 or i + 240 > len(k):
        return None
    ohlc = k[:, 1:5]
    lv = ep.exit_levels(ep.volatility_1h_from_ohlc(ohlc[i - 1000:i]))
    return ohlc[i, 0], ohlc[i:i + 240, 1:4], lv

def report(title, trades):
    print(f"\n### {title}: {len(trades)} trades")
    print(f"{'rule':32s} {'avg/trade':>10s} {'win':>6s} {'ROI@20x':>8s} {'stopped':>8s}")
    for name, v in VARIANTS.items():
        r = np.array([sim(d, e, b, lv, v) for d, e, b, lv in trades]) - COST
        stopped = np.mean([abs(x + COST + lv.stop) < 1e-9 for x, (_, _, _, lv) in zip(r, trades)])
        print(f"{name:32s} {r.mean()*100:+9.3f}% {np.mean(r > 0)*100:5.1f}% {r.mean()*2000:+7.2f}% {stopped*100:7.1f}%")
    sys.stdout.flush()

mode = sys.argv[1] if len(sys.argv) > 1 else "both"

if mode in ("both", "random"):
    random.seed(1)
    days = [date(2026, 10, 21) - timedelta(days=i) for i in range(88)][::-1]
    trades = []
    for sym in settings.symbol_universe:
        for d in days:
            base = datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()
            for hh in range(0, 24, 2):
                p = path_for(sym, base + hh * 3600 + random.randint(0, 3599))
                if p:
                    trades.append((random.choice((1, -1)), *p))
    report("RANDOM entries (no edge)", trades)
    _kl.clear()

if mode in ("both", "brain"):
    from core.ai.replay_buffer import ReplayBuffer
    import core.ai.replay_buffer as rbm
    from core.schemas.state_layout import FULL_INPUTS_SINCE
    from apps.research.model import SingleSymbolActorCritic
    from apps.trainer.ppo import normalize_state_tensor, PPOTrainer, live_entry_hurdle_pct
    from core.ai import critic_units as cu
    tp = json.load(open("models/production/version.json"))["metrics"]["trade_policy"]
    print("brain", json.load(open("models/production/version.json"))["version"], "min_conf", tp["min_confidence"], "blocked", tp["blocked_symbols"])
    m = SingleSymbolActorCritic(input_dim=200); m.load_state_dict(torch.load("models/production/model_v1.pt", map_location="cpu")); m.eval()
    b = ReplayBuffer(); b.min_timestamp = FULL_INPUTS_SINCE; b.replay_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    rbm.REPLAY_DAYS_PER_LOAD = 200
    b.impute_values = m.feat_mean.detach().clone().numpy()
    with b.SessionLocal() as s:
        rows = b._load_replay_rows(s)
    print("history rows", len(rows)); sys.stdout.flush()
    bias = torch.tensor(tp.get("critic_bias") or [0, 0, 0], dtype=torch.float32)
    trades = []
    for i in range(0, len(rows), 200):
        chunk = rows[i:i + 200]
        st, _, _, _, _ = b.build_tensors(chunk, seq_len=300)
        st = normalize_state_tensor(st)
        with torch.no_grad():
            acts, raw = m(st)
        vals = cu.to_pct_torch(m, raw, st[:, -1, :]) - bias
        hur = torch.tensor([live_entry_hurdle_pct(e.timestamp) for e in chunk])
        pos = PPOTrainer.live_entry_positions(vals, hur)[:, 0]
        conf = (acts[:, 1] + 1) / 2
        for e, p, c in zip(chunk, pos.tolist(), conf.tolist()):
            if p == 0 or c < (tp["min_confidence"] or 0) or e.symbol in tp["blocked_symbols"]:
                continue
            pth = path_for(e.symbol, e.timestamp)
            if pth:
                trades.append((int(p), *pth))
    longs = sum(1 for t in trades if t[0] > 0)
    print(f"brain trades {len(trades)} (long {longs})")
    report("BRAIN-selected entries (live rule, history)", trades)
