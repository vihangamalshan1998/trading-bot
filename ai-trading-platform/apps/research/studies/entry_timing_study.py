# Run from ai-trading-platform: PYTHONPATH=. venv/Scripts/python apps/research/studies/entry_timing_study.py [rows]
"""Read-only study: enter at once vs wait for the turn, on the entries the live brain would take (replayed history,
live entry rule + its trade policy). Live exits (simulate_trade) and live sizing (size x min(1, 0.75% / stop)).
A waiting rule that gives up takes no trade (0). "per signal" = total / signals, the number that decides."""
import json, sys
from datetime import datetime, timedelta, timezone
import numpy as np, torch
from core.risk import exit_policy as ep
from apps.replay import archive

COST = ep.ROUND_TRIP_COST
BASE_STOP = 0.0075
HOLD_BARS = 240

def wait_candle(o, c, d, n, give_up):
    """First n-minute candle (blocks from the signal) closing in the trade's direction -> entry index after it."""
    for s in range(0, give_up, n):
        if (c[s + n - 1] - o[s]) * d > 0:
            return s + n
    return None

def wait_turn(h, l, c, d, bounce, give_up):
    """Long: first close >= bounce above the lowest low since the signal (short: mirror) -> entry index after it."""
    ext = None
    for s in range(give_up):
        ext = (l[s] if ext is None else min(ext, l[s])) if d > 0 else (h[s] if ext is None else max(ext, h[s]))
        if (d > 0 and c[s] >= ext * (1 + bounce)) or (d < 0 and c[s] <= ext * (1 - bounce)):
            return s + 1
    return None

RULES = {
    "NOW  enter at once (live)": lambda o, h, l, c, d, st: 0,
    "G1   1m candle our way (30m)": lambda o, h, l, c, d, st: wait_candle(o, c, d, 1, 30),
    "G5   5m candle our way (60m)": lambda o, h, l, c, d, st: wait_candle(o, c, d, 5, 60),
    "T03  bounce 0.3% off extreme (60m)": lambda o, h, l, c, d, st: wait_turn(h, l, c, d, 0.003, 60),
    "TH   bounce half stop (60m)": lambda o, h, l, c, d, st: wait_turn(h, l, c, d, st / 2, 60),
}

_kl = {}
def klines(sym, day):
    if (sym, day) not in _kl:
        _kl[(sym, day)] = archive.load_klines("klines", sym, day)
    return _kl[(sym, day)]

def window(sym, ts):
    """1m OHLC from 1000 bars before the signal to 60 + 240 bars after it, and the signal index."""
    d = datetime.fromtimestamp(ts, timezone.utc).date()
    parts = [klines(sym, d + timedelta(days=i)) for i in (-1, 0, 1)]
    if any(p is None for p in parts[:2]):
        return None
    k = np.concatenate([p for p in parts if p is not None])
    i = int(np.searchsorted(k[:, 0], ts * 1000.0, side="left"))
    if i < 1000 or i + 60 + HOLD_BARS + 1 > len(k):
        return None
    return k[i - 1000:i + 60 + HOLD_BARS + 1, 1:5], 1000

def grade(d, ohlc, i0, rule):
    st = ep.exit_levels(ep.volatility_1h_from_ohlc(ohlc[i0 - 1000:i0])).stop
    o, h, l, c = (ohlc[i0:, j] for j in range(4))
    w = rule(o, h, l, c, d, st)
    if w is None:
        return None
    j = i0 + w
    lv = ep.exit_levels(ep.volatility_1h_from_ohlc(ohlc[j - 1000:j]))
    g = ep.simulate_trade(d, ohlc[j, 0], ohlc[j:j + HOLD_BARS, 1:4], lv)
    return (g - COST) * min(1.0, BASE_STOP / lv.stop)

def report(title, sigs):
    print(f"\n### {title}")
    for side_name, side in (("LONG", 1), ("SHORT", -1)):
        ss = [s for s in sigs if s[0] == side]
        print(f"  {side_name}: {len(ss)} signals")
        print(f"  {'rule':36s} {'taken':>6s} {'win':>6s} {'per trade':>10s} {'per signal':>11s} {'total':>8s}")
        for name, rule in RULES.items():
            res = [grade(d, ohlc, i0, rule) for d, ohlc, i0 in ss]
            got = np.array([r for r in res if r is not None])
            tot = got.sum() if len(got) else 0.0
            print(f"  {name:36s} {len(got):6d} {(np.mean(got > 0) * 100 if len(got) else 0):5.1f}% "
                  f"{(got.mean() * 100 if len(got) else 0):+9.3f}% {tot / max(1, len(ss)) * 100:+10.3f}% {tot * 100:+7.2f}%")
    sys.stdout.flush()

from core.ai.replay_buffer import ReplayBuffer
import core.ai.replay_buffer as rbm
from core.schemas.state_layout import FULL_INPUTS_SINCE
from apps.research.model import SingleSymbolActorCritic
from apps.trainer.ppo import normalize_state_tensor, PPOTrainer, live_entry_hurdle_pct
from core.ai import critic_units as cu
vj = json.load(open("models/production/version.json"))
tp = vj["metrics"]["trade_policy"]
print("brain", vj["version"], "min_conf", tp["min_confidence"], "blocked", tp["blocked_symbols"])
m = SingleSymbolActorCritic(input_dim=200); m.load_state_dict(torch.load("models/production/model_v1.pt", map_location="cpu")); m.eval()
b = ReplayBuffer(); b.min_timestamp = FULL_INPUTS_SINCE; b.replay_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
rbm.REPLAY_DAYS_PER_LOAD = 200
b.impute_values = m.feat_mean.detach().clone().numpy()
bias = torch.tensor(tp.get("critic_bias") or [0, 0, 0], dtype=torch.float32)
for exam in (False, True):
    with b.SessionLocal() as s:
        rows = b._load_replay_rows(s, exam=exam)
    sigs = []
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
            w = window(e.symbol, e.timestamp)
            if w:
                sigs.append((int(p), *w))
    report(f"{'EXAM days (never seen)' if exam else 'TRAINING days (seen)'}: {len(rows)} rows, {len(sigs)} signals", sigs)
    _kl.clear()
