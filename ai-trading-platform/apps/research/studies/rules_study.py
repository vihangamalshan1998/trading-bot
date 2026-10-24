# Run from ai-trading-platform: PYTHONPATH=. venv/Scripts/python apps/research/studies/rules_study.py [training_days] [noexam] [combo|tune]
"""Read-only study: extra entry/hold rules vs the live rules, on WHOLE history days (every graded moment of every coin,
about one per coin per 10 min) so trades follow each other in time like live. The live brain decides every moment
(live entry rule + its trade policy); a portfolio replay applies the live limits: one trade per coin, at most
5 open on one side, 60 min lock on a coin's direction after a loss, 10 entries per hour. Live exits and live sizing
(size x min(1, 0.75% / stop)), net of 0.10% cost. Each rule is switched on alone against that baseline.
Sets: the 15 exam days (never trained on) and N random training days (seen)."""
import json, sys, random
from datetime import datetime, timedelta, timezone
import numpy as np, torch
from sqlalchemy.sql import text
from core.risk import exit_policy as ep
from apps.replay import archive

COST = ep.ROUND_TRIP_COST
BASE_STOP = 0.0075
MAX_BARS = 8 * 60

_kl = {}
def klines(sym, day):
    if (sym, day) not in _kl:
        _kl[(sym, day)] = archive.load_klines("klines", sym, day)
    return _kl[(sym, day)]

def window(sym, ts):
    """1m OHLC from 1000 bars before ts to MAX_BARS after it, and the index of ts."""
    d = datetime.fromtimestamp(ts, timezone.utc).date()
    parts = [klines(sym, d + timedelta(days=i)) for i in (-1, 0, 1)]
    if any(p is None for p in parts[:2]):
        return None
    k = np.concatenate([p for p in parts if p is not None])
    i = int(np.searchsorted(k[:, 0], ts * 1000.0, side="left"))
    if i < 1000 or i + MAX_BARS + 1 > len(k):
        return None
    return k[i - 1000:i + MAX_BARS + 1, 1:5]

def btc_ret(ts, mins):
    w = window("BTCUSDT", ts)
    return None if w is None else w[1000, 0] / w[1000 - mins, 0] - 1.0

class Sig:
    __slots__ = ("sym", "ts", "d", "conf", "funding", "ohlc", "lv", "btc1", "btc4", "btc30", "own1", "own30", "_g")
    def grade(self, hold_h):
        """(sized net move, exit time) with the live exits and a max hold of hold_h hours."""
        if hold_h not in self._g:
            ep.MAX_HOLD_SECONDS = int(hold_h * 3600)
            bars = self.ohlc[1000:1000 + MAX_BARS, 1:4]
            full = ep.simulate_trade(self.d, self.ohlc[1000, 0], bars, self.lv)
            lo, hi = 1, int(hold_h * 60)  # smallest bar count that gives the same result = the exit bar
            while lo < hi:
                mid = (lo + hi) // 2
                if ep.simulate_trade(self.d, self.ohlc[1000, 0], bars[:mid], self.lv) == full:
                    hi = mid
                else:
                    lo = mid + 1
            ep.MAX_HOLD_SECONDS = 4 * 3600
            self._g[hold_h] = ((full - COST) * min(1.0, BASE_STOP / self.lv.stop), self.ts + lo * 60)
        return self._g[hold_h]

def replay(sigs, keep=lambda s: True, side_cap=5, lock_min=60, hold=lambda s, by: 4, size=lambda s: 1.0, per_hour=10):
    """Portfolio replay in time order. Returns the list of (signal, sized net result)."""
    open_until, side_open, lock, entries, out = {}, [], {}, [], []
    for s in sigs:
        if not keep(s):
            continue
        side_open = [(t, d) for t, d in side_open if t > s.ts]
        entries = [t for t in entries if t > s.ts - 3600]
        if open_until.get(s.sym, 0) > s.ts or lock.get((s.sym, s.d), 0) > s.ts or len(entries) >= per_hour:
            continue
        if sum(1 for _, d in side_open if d == s.d) >= side_cap:
            continue
        g, t_exit = s.grade(hold(s, by_coin))
        open_until[s.sym] = t_exit
        side_open.append((t_exit, s.d))
        entries.append(s.ts)
        if g < 0 and lock_min:
            lock[(s.sym, s.d)] = t_exit + lock_min * 60
        out.append((s, g * size(s)))
    return out

def roll_hold(s, by):
    """4 h, or 8 h when the brain gives the same signal on this coin 3.5-4 h after entry (roll-over)."""
    return 8 if any(o.d == s.d and s.ts + 3.5 * 3600 <= o.ts <= s.ts + 4 * 3600 for o in by.get(s.sym, ())) else 4

def report(title, sigs):
    global by_coin
    by_coin = {}
    for s in sigs:
        by_coin.setdefault(s.sym, []).append(s)
    mean_conf = np.mean([s.conf for s in sigs])
    f = np.array([s.funding for s in sigs])
    f_hi, f_lo = np.quantile(f, 0.75), np.quantile(f, 0.25)
    rules = {
        "BASE live rules": {},
        "1a BTC trend 1h (no long if BTC 1h down)": dict(keep=lambda s: s.btc1 is not None and s.btc1 * s.d >= 0),
        "1b BTC trend 4h": dict(keep=lambda s: s.btc4 is not None and s.btc4 * s.d >= 0),
        "1c BTC 1h strongly against (>0.5%) only": dict(keep=lambda s: s.btc1 is None or s.btc1 * s.d > -0.005),
        "2a confidence >= 0.50": dict(keep=lambda s: s.conf >= 0.50),
        "2b confidence >= 0.55": dict(keep=lambda s: s.conf >= 0.55),
        "2c confidence >= 0.60": dict(keep=lambda s: s.conf >= 0.60),
        "2d longs need confidence >= 0.55": dict(keep=lambda s: s.d < 0 or s.conf >= 0.55),
        "3a hold max 2 h": dict(hold=lambda s, by: 2),
        "3b hold max 6 h": dict(hold=lambda s, by: 6),
        "3c hold max 8 h": dict(hold=lambda s, by: 8),
        "3d roll-over at 4 h if brain still agrees": dict(hold=roll_hold),
        "4  size by confidence (same average)": dict(size=lambda s: s.conf / mean_conf),
        "6a max 2 open on one side": dict(side_cap=2),
        "6b max 3 open on one side": dict(side_cap=3),
        "7  funding filter (no crowded side)": dict(keep=lambda s: not (s.d > 0 and s.funding >= f_hi) and not (s.d < 0 and s.funding <= f_lo)),
        "8a no lock after a loss": dict(lock_min=0),
        "8b 30 min lock": dict(lock_min=30),
        "8c 120 min lock": dict(lock_min=120),
    }
    btc = rules["1c BTC 1h strongly against (>0.5%) only"]["keep"]
    fund = rules["7  funding filter (no crowded side)"]["keep"]
    combos = {
        "C1 BTC + funding (1c + 7)": dict(keep=lambda s: btc(s) and fund(s)),
        "C2 BTC + 30 min lock (1c + 8b)": dict(keep=btc, lock_min=30),
        "C3 funding + 30 min lock (7 + 8b)": dict(keep=fund, lock_min=30),
        "C4 all three (1c + 7 + 8b)": dict(keep=lambda s: btc(s) and fund(s), lock_min=30),
    }
    if "combo" in sys.argv:
        rules = {k: rules[k] for k in ("BASE live rules", "1c BTC 1h strongly against (>0.5%) only",
                                       "7  funding filter (no crowded side)", "8b 30 min lock")}
    rules.update(combos)
    if "tune" in sys.argv:  # C2 is live since 24 Oct: tune its BTC limit / window, then add the coin's own move
        def btc_lim(lim, win=60):
            return lambda s: (s.btc1 if win == 60 else s.btc30) is None or (s.btc1 if win == 60 else s.btc30) * s.d > -lim
        def own(lim, win=60, against=True):
            """Skip when the coin itself moved >= lim against the trade (falling knife) or with it (chasing)."""
            return lambda s: ((s.own1 if win == 60 else s.own30) * s.d * (1 if against else -1)) > -lim
        c2 = btc_lim(0.005)
        rules = {"BASE C2 (live now: BTC 1h 0.5% + 30 min lock)": dict(keep=c2, lock_min=30)}
        for lim in (0.003, 0.004, 0.006, 0.007, 0.010):
            rules[f"T1 BTC 1h limit {lim * 100:.1f}%"] = dict(keep=btc_lim(lim), lock_min=30)
        for lim in (0.003, 0.005):
            rules[f"T2 BTC 30m window, limit {lim * 100:.1f}%"] = dict(keep=btc_lim(lim, 30), lock_min=30)
        for lim in (0.005, 0.010, 0.015, 0.020):
            o = own(lim)
            rules[f"T3 C2 + coin 1h fell/rose {lim * 100:.1f}% against"] = dict(keep=lambda s, o=o: c2(s) and o(s), lock_min=30)
        for lim in (0.005, 0.010):
            o = own(lim, 30)
            rules[f"T4 C2 + coin 30m {lim * 100:.1f}% against"] = dict(keep=lambda s, o=o: c2(s) and o(s), lock_min=30)
        for lim in (0.010, 0.020):
            o = own(lim, against=False)
            rules[f"T5 C2 + no chasing: coin 1h {lim * 100:.1f}% with"] = dict(keep=lambda s, o=o: c2(s) and o(s), lock_min=30)
    print(f"\n### {title}: {len(sigs)} brain signals ({sum(s.d > 0 for s in sigs)} long), "
          f"funding filter: no long at >= {f_hi:.6f}, no short at <= {f_lo:.6f}")
    print(f"{'rule':44s} {'trades':>6s} {'long':>5s} {'win':>6s} {'per trade':>10s} {'total':>8s} {'long tot':>9s} {'short tot':>9s}")
    base = None
    for name, kw in rules.items():
        res = replay(sigs, **kw)
        g = np.array([r for _, r in res]) if res else np.zeros(0)
        lt = sum(r for s, r in res if s.d > 0) * 100
        st = sum(r for s, r in res if s.d < 0) * 100
        tot = g.sum() * 100
        base = tot if base is None else base
        print(f"{name:44s} {len(g):6d} {sum(s.d > 0 for s, _ in res):5d} {(np.mean(g > 0) * 100 if len(g) else 0):5.1f}% "
              f"{(g.mean() * 100 if len(g) else 0):+9.3f}% {tot:+7.2f}% {lt:+8.2f}% {st:+8.2f}%  ({tot - base:+.2f})")
        if name.startswith("BASE"):
            base_res = res
    print("  BASE by UTC hour (rule 5):  " + "  ".join(
        f"{h:02d}-{h + 4:02d}h {sum(r for s, r in base_res if (s.ts // 3600) % 24 // 4 == h // 4) * 100:+.2f}%" for h in range(0, 24, 4)))
    sys.stdout.flush()

from core.ai.replay_buffer import ReplayBuffer, REPLAY_TABLE, GRADED_FLAG, is_exam_day
from core.database.models.ai import Experience
from core.schemas.state_layout import FULL_INPUTS_SINCE, SLOT_BY_NAME
from apps.replay.states import SequenceReader
from apps.research.model import SingleSymbolActorCritic
from apps.trainer.ppo import normalize_state_tensor, PPOTrainer, live_entry_hurdle_pct
from core.ai import critic_units as cu
vj = json.load(open("models/production/version.json"))
tp = vj["metrics"]["trade_policy"]
print("brain", vj["version"], "min_conf", tp["min_confidence"], "blocked", tp["blocked_symbols"])
m = SingleSymbolActorCritic(input_dim=200); m.load_state_dict(torch.load("models/production/model_v1.pt", map_location="cpu")); m.eval()
b = ReplayBuffer()
b.impute_values = m.feat_mean.detach().clone().numpy()
bias = torch.tensor(tp.get("critic_bias") or [0, 0, 0], dtype=torch.float32)
F_SLOT = SLOT_BY_NAME["funding_rate"]
random.seed(7)

def signals_for(days):
    with b.SessionLocal() as s:
        params = {f"d_{i}": d for i, d in enumerate(days)}
        index = s.execute(text(
            f"SELECT id, symbol, timestamp, day, frame_idx, reward_5m, reward_1h, reward_4h FROM {REPLAY_TABLE} WHERE data_quality_flags = '{GRADED_FLAG}' "
            f"AND timestamp < {int(FULL_INPUTS_SINCE)} AND day IN ({','.join(':' + k for k in params)})"), params).fetchall()
    index = sorted(index, key=lambda r: (r.symbol, r.day, r.timestamp))
    reader, sigs = SequenceReader(), []
    for i in range(0, len(index), 200):
        chunk = []
        for r in index[i:i + 200]:
            frames = reader.sequence(r.symbol, r.day, int(r.frame_idx))
            if frames is None:
                continue
            e = Experience(id=r.id, symbol=r.symbol, timestamp=r.timestamp, action="HOLD", reward_5m=r.reward_5m,
                           reward_1h=r.reward_1h, reward_4h=r.reward_4h, data_quality_flags=GRADED_FLAG)
            e._src = "replay"
            e._frames = np.array(frames, copy=True)
            e._trade_outcome = None
            chunk.append(e)
        if not chunk:
            continue
        st, _, _, _, _ = b.build_tensors(chunk, seq_len=300)
        funding = st[:, -1, F_SLOT].tolist()
        st = normalize_state_tensor(st)
        with torch.no_grad():
            acts, raw = m(st)
        vals = cu.to_pct_torch(m, raw, st[:, -1, :]) - bias
        hur = torch.tensor([live_entry_hurdle_pct(e.timestamp) for e in chunk])
        pos = PPOTrainer.live_entry_positions(vals, hur)[:, 0]
        conf = (acts[:, 1] + 1) / 2
        for e, p, c, fr in zip(chunk, pos.tolist(), conf.tolist(), funding):
            if p == 0 or c < (tp["min_confidence"] or 0) or e.symbol in tp["blocked_symbols"]:
                continue
            w = window(e.symbol, e.timestamp)
            if w is None:
                continue
            s = Sig()
            s.sym, s.ts, s.d, s.conf, s.funding, s.ohlc, s._g = e.symbol, int(e.timestamp), int(p), float(c), float(fr), w, {}
            s.lv = ep.exit_levels(ep.volatility_1h_from_ohlc(w[:1000]))
            s.btc1, s.btc4, s.btc30 = btc_ret(s.ts, 60), btc_ret(s.ts, 240), btc_ret(s.ts, 30)
            s.own1, s.own30 = w[1000, 0] / w[940, 0] - 1.0, w[1000, 0] / w[970, 0] - 1.0
            sigs.append(s)
        if i % 4000 == 0:
            print(f"  {i}/{len(index)} moments, {len(sigs)} signals"); sys.stdout.flush()
    return sorted(sigs, key=lambda s: s.ts)

with b.SessionLocal() as s:
    all_days = [r[0] for r in s.execute(text(
        f"SELECT DISTINCT day FROM {REPLAY_TABLE} WHERE data_quality_flags = '{GRADED_FLAG}' AND timestamp < {int(FULL_INPUTS_SINCE)}")).fetchall()]
exam_days = sorted(d for d in all_days if is_exam_day(d))
pool = [d for d in all_days if not is_exam_day(d)]
train_days = sorted(random.sample(pool, min(len(pool), int(sys.argv[1]) if len(sys.argv) > 1 else 15)))
sets = (("EXAM days (never seen)", exam_days), ("TRAINING days (seen)", train_days))
for title, days in sets[1:] if "noexam" in sys.argv else sets:
    print(f"\nloading {title}: {len(days)} days"); sys.stdout.flush()
    report(f"{title}, {len(days)} whole days", signals_for(days))
    _kl.clear()
