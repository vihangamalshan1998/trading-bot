# Run from ai-trading-platform: PYTHONPATH=. venv/Scripts/python apps/research/studies/stop_study.py [random|brain [rows]]
"""Read-only STOP-WIDTH study (same dollar risk: size x stop constant) on mainnet history: current rules vs 4 variants.
Set 1: entries the live brain would take (replayed history rows, live entry rule + its trade policy).
Set 2: random entries (no edge), every 2 h, both directions, all coins, 90 days."""
import json, sys, time, random
from datetime import date, datetime, timedelta, timezone
import numpy as np, torch
from core.risk import exit_policy as ep
from apps.replay import archive
from core.config.settings import settings

COST = ep.ROUND_TRIP_COST
BASE_STOP = ep.MAX_STOP
# name: (stop multiple of 1h volatility, stop cap)
VARIANTS = {
    "A current (1.0x vol, cap 0.75%)": (1.0, 0.0075),
    "W1 cap 1.0%": (1.0, 0.010),
    "W2 cap 1.2%": (1.0, 0.012),
    "W3 stop = 1h vol, cap 2%": (1.0, 0.020),
    "W4 stop = 1.25x vol, cap 2%": (1.25, 0.020),
}

def levels(vol, mult, cap):
    vol = vol if vol and vol > 0 else ep.DEFAULT_VOLATILITY
    stop = min(cap, max(ep.MIN_STOP, vol * mult))
    return ep.ExitLevels(stop=stop, tp1=max(stop * ep.TP1_RR, stop + 2.0 * COST), hard_tp=stop * ep.HARD_TP_RR,
                         breakeven=COST, trail_gap=stop)

def sim(direction, entry, bars, lv, v=None):
    v = v or {}
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
    return ohlc[i, 0], ohlc[i:i + 240, 1:4], ep.volatility_1h_from_ohlc(ohlc[i - 1000:i])

def report(title, trades):
    print(f"\n### {title}: {len(trades)} trades  (size scaled so a stop always loses the same $ as a 0.75% stop)")
    print(f"{'rule':32s} {'median stop':>11s} {'stopped':>8s} {'win':>6s} {'net move':>9s} {'per $-risk':>10s} {'ROI@20x':>8s}")
    for name, (mult, cap) in VARIANTS.items():
        res = []
        for d, e, b, vol in trades:
            lv = levels(vol, mult, cap)
            g = sim(d, e, b, lv)
            res.append((g - COST, lv.stop, abs(g + lv.stop) < 1e-12))
        net = np.array([r[0] for r in res]); st = np.array([r[1] for r in res])
        scaled = net * (BASE_STOP / st)
        print(f"{name:32s} {np.median(st)*100:10.2f}% {np.mean([r[2] for r in res])*100:7.1f}% {np.mean(net > 0)*100:5.1f}% "
              f"{net.mean()*100:+8.3f}% {scaled.mean()*100:+9.3f}% {scaled.mean()*2000:+7.2f}%")
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
