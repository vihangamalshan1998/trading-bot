# Run from ai-trading-platform: PYTHONPATH=. venv/Scripts/python apps/research/studies/ladder_study.py [random|brain [rows]]
"""Read-only study: TP ladders (TP1 / TP2 / TP3) vs the live exits, live sizing (size x min(1, 0.75% / stop)) on mainnet history: current rules vs 4 variants.
Set 1: entries the live brain would take (replayed history rows, live entry rule + its trade policy).
Set 2: random entries (no edge), every 2 h, both directions, all coins, 90 days."""
import json, sys, time, random
from datetime import date, datetime, timedelta, timezone
import numpy as np, torch
from core.risk import exit_policy as ep
from apps.replay import archive
from core.config.settings import settings

COST = ep.ROUND_TRIP_COST
BASE_STOP = 0.0075
# All variants use the live stop (1h swing, 0.35-2%). tps: (multiple of the stop, share of the position)
VARIANTS = {
    "A live rules (TP1 50% @1.5x, trail, safe line +1%)": dict(tps=[("tp1", 0.5)], be_trigger=0.010, trail_rest=True),
    "L1 1/3 @1x, 1/3 @2x, 1/3 @3x, SL->entry after TP1": dict(tps=[(1.0, 1 / 3), (2.0, 1 / 3), (3.0, 1 / 3)]),
    "L2 1/3 @1.5x, 1/3 @2.5x, rest trails, SL->entry": dict(tps=[(1.5, 1 / 3), (2.5, 1 / 3)], trail_rest=True),
    "L3 = L1 + safe line +1%": dict(tps=[(1.0, 1 / 3), (2.0, 1 / 3), (3.0, 1 / 3)], be_trigger=0.010),
    "L4 50% @1x, rest trails, SL->entry": dict(tps=[(1.0, 0.5)], trail_rest=True),
}

def levels(vol, mult, cap):
    vol = vol if vol and vol > 0 else ep.DEFAULT_VOLATILITY
    stop = min(cap, max(ep.MIN_STOP, vol * mult))
    return ep.ExitLevels(stop=stop, tp1=max(stop * ep.TP1_RR, stop + 2.0 * COST), hard_tp=stop * ep.HARD_TP_RR,
                         breakeven=COST, trail_gap=stop)

def sim(direction, entry, bars, lv, tps, be_trigger=None, trail_rest=False):
    """Ladder exits. After the first TP the stop moves to breakeven; once every fixed TP is hit and
    trail_rest is set, the rest trails one stop behind the peak (never below breakeven)."""
    rem, real, hit, peak, held, mc = 1.0, 0.0, 0, 0.0, 0, 0.0
    floor = -lv.stop
    for h, l, c in bars:
        held += 60
        if direction > 0:
            best, worst, mc = (h - entry) / entry, (l - entry) / entry, (c - entry) / entry
        else:
            best, worst, mc = (entry - l) / entry, (entry - h) / entry, (entry - c) / entry
        cur = floor
        if be_trigger is not None and peak >= be_trigger:
            cur = max(cur, lv.breakeven)
        if trail_rest and hit >= len(tps):
            cur = max(cur, peak - lv.trail_gap, lv.breakeven)
        if worst <= cur:
            return real + rem * cur
        if best >= lv.hard_tp:
            return real + rem * lv.hard_tp
        while hit < len(tps) and best >= (lv.tp1 if tps[hit][0] == "tp1" else tps[hit][0] * lv.stop):
            real += tps[hit][1] * (lv.tp1 if tps[hit][0] == "tp1" else tps[hit][0] * lv.stop)
            rem -= tps[hit][1]
            hit += 1
            floor = max(floor, lv.breakeven)
            if rem <= 1e-9:
                return real
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
    print(f"\n### {title}: {len(trades)} trades (live stop + live sizing; net of 0.10% cost)")
    print(f"{'rule':52s} {'stopped':>8s} {'win':>6s} {'per trade':>10s} {'ROI@20x':>8s}")
    for name, kw in VARIANTS.items():
        res = []
        for d, e, b, vol in trades:
            lv = ep.exit_levels(vol)
            g = sim(d, e, b, lv, **kw)
            if name.startswith("A "):  # the ladder sim must match the live simulator exactly
                assert abs(g - ep.simulate_trade(d, e, b, lv)) < 1e-12, (g, ep.simulate_trade(d, e, b, lv))
            res.append(((g - COST) * min(1.0, BASE_STOP / lv.stop), abs(g + lv.stop) < 1e-12))
        net = np.array([r[0] for r in res])
        print(f"{name:52s} {np.mean([r[1] for r in res])*100:7.1f}% {np.mean(net > 0)*100:5.1f}% {net.mean()*100:+9.3f}% {net.mean()*2000:+7.2f}%")
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
