"""History replay (apps/replay) and how the trainer uses replayed rows."""
import types
from datetime import date

import numpy as np

from apps.replay.layout import REPLAY_MISSING_SLOTS, REPLAY_APPROX_SLOTS
from apps.replay.symbol_replay import SymbolDayReplay, _MinuteSeries, frame_grid, FRAME_SECONDS
from core.ai.replay_buffer import ReplayBuffer
from core.database.models.ai import GRADED_FLAG
from core.schemas.state_layout import SLOTS, ACTIVE, BRAIN_EXCLUDED_SLOTS


def _row(src, row_id, value=0.5):
    r = types.SimpleNamespace(id=row_id, reward_5m=0.1, reward_1h=0.2, reward_4h=0.3, data_quality_flags=GRADED_FLAG,
                              action="HOLD", symbol="BTCUSDT", timestamp=1792000000, _trade_outcome=(0.4, -0.4),
                              _frames=np.full((300, 200), value, dtype=np.float32))
    if src:
        r._src = src
    return r


def _buffer():
    b = ReplayBuffer.__new__(ReplayBuffer)
    b.impute_values = None
    b.replay_cache = []
    return b


def test_layout_lists_only_active_brain_slots():
    status = {s.index: s.status for s in SLOTS}
    for i in REPLAY_MISSING_SLOTS + REPLAY_APPROX_SLOTS:
        assert status[i] == ACTIVE and i not in BRAIN_EXCLUDED_SLOTS
    assert not set(REPLAY_MISSING_SLOTS) & set(REPLAY_APPROX_SLOTS)


def test_replayed_rows_get_the_model_mean_in_missing_slots():
    b = _buffer()
    b.impute_values = np.arange(200, dtype=np.float32) / 1000.0
    s, _, r, _, _ = b.build_tensors([_row("replay", 1), _row(None, 2)])
    miss = list(REPLAY_MISSING_SLOTS)
    assert np.allclose(s[0, :, miss].numpy(), b.impute_values[miss])
    assert np.allclose(s[1, :, miss].numpy(), 0.5)            # live rows keep their real values
    keep = [i for i in range(200) if i not in REPLAY_MISSING_SLOTS]
    assert np.allclose(s[0, :, keep].numpy(), 0.5)            # replayed inputs that exist are untouched
    assert r.shape == (2, 5)


def test_replayed_rows_carry_nan_while_feature_stats_are_fitted():
    b = _buffer()
    s, _, _, _, _ = b.build_tensors([_row("replay", 1)])
    assert np.isnan(s[0, :, list(REPLAY_MISSING_SLOTS)].numpy()).all()
    assert not np.isnan(s[0, :, 25].numpy()).any()


def test_dedup_keeps_rows_from_different_tables_with_the_same_id():
    assert ReplayBuffer._row_key(_row("replay", 7)) != ReplayBuffer._row_key(_row(None, 7))


def _replay_with(premium_by_minute, funding_times_ms):
    r = SymbolDayReplay("BTCUSDT", date(2026, 10, 20))
    rows = np.array([[m * 60000.0, p, p, p, p, 0, 0, 0] for m, p in premium_by_minute])
    r.premium = _MinuteSeries(rows)
    r.funding_times = np.array(funding_times_ms, dtype=np.int64)
    return r


def test_funding_estimate_uses_only_past_premiums_with_linear_weights():
    start = 1792454400  # 2026-10-20 00:00 UTC (a funding time)
    m0 = start // 60
    r = _replay_with([(m0, 0.001), (m0 + 1, 0.003), (m0 + 2, 0.5)], [start * 1000, (start + 8 * 3600) * 1000])
    t = start + 2 * 60 + 30  # inside minute 2: minute 2's premium (0.5) is not complete yet
    avg = (1 * 0.001 + 2 * 0.003) / 3
    interest = 0.0003 / 3  # 8h interval
    expected = avg + np.clip(interest - avg, -0.0005, 0.0005)
    assert abs(r._estimated_funding(t) - expected) < 1e-12
    assert r._funding_period(t) == (start, start + 8 * 3600)


def test_open_interest_never_looks_ahead():
    r = SymbolDayReplay("BTCUSDT", date(2026, 10, 20))
    r.metrics = {"t": np.array([1000, 1300, 1600]), "oi": np.array([10.0, 20.0, 40.0]), "ls": np.array([1.0, 1.1, 1.2])}
    assert r._open_interest(1299) == 10.0           # before the 1300 print only the first value is known
    assert r._open_interest(1450) == 15.0           # halfway from 10 to 20, a 5-minute lag behind
    assert r._open_interest(1600) == 20.0 and r._open_interest(2000) == 40.0
    assert r._ls_ratio(1299) == 1.0 and r._ls_ratio(1300) == 1.1


def test_frame_grid_matches_the_live_cycle():
    g = frame_grid(date(2026, 10, 20))
    assert abs(g[1] - g[0] - FRAME_SECONDS) < 1e-6 and len(g) == int(86400 / FRAME_SECONDS) + 1


def test_round_sig_matches_the_bot_storage_rounding():
    from apps.replay.states import round_sig
    from core.schemas.state_builder import round_frame
    rng = np.random.default_rng(5)
    vals = rng.normal(0, 1, 2000) * rng.choice([1e-7, 1e-3, 1.0, 37.0, 4.9], 2000)
    vals[:5] = [0.0, 5.0, -5.0, 1e-12, 123456.789]
    ours = round_sig(vals).astype(np.float32)
    ref = np.asarray(round_frame(vals), dtype=np.float32)
    assert np.allclose(ours, ref, rtol=2e-6, atol=0)


def test_sequences_are_cut_across_midnight_and_refuse_gaps(tmp_path, monkeypatch):
    from apps.replay import states
    monkeypatch.setattr(states, "STATES_DIR", str(tmp_path))
    d1, d2 = date(2026, 10, 19), date(2026, 10, 20)
    n = 400
    t1 = 1792368000 + np.arange(n) * 5.3
    t2 = 1792454400 + np.arange(n) * 5.3
    s1 = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 200))
    s2 = s1 + 1000
    states.save_states("BTCUSDT", d1, t1, s1, np.ones(n, bool))
    # day 2 starts right after day 1 in this toy (consecutive frames)
    t2 = t1[-1] + 5.3 * (1 + np.arange(n))
    states.save_states("BTCUSDT", d2, t2, s2, np.ones(n, bool))
    rd = states.SequenceReader()
    seq = rd.sequence("BTCUSDT", d2, 49)                        # 50 frames of day 2 + 250 of day 1
    assert seq.shape == (300, 200) and seq[0, 0] == 150 and seq[-1, 0] == 1049
    assert rd.sequence("BTCUSDT", d2, 399)[0, 0] == 1100
    bad = np.ones(n, bool); bad[10] = False
    states.save_states("ETHUSDT", d2, t2, s2, bad)
    assert states.SequenceReader().sequence("ETHUSDT", d2, 305) is None   # invalid frame inside the window
    assert states.SequenceReader().sequence("ETHUSDT", d2, 399) is not None
    assert states.SequenceReader().sequence("SOLUSDT", d2, 399) is None   # no file


def test_holdout_is_the_newest_24h_capped_at_60_percent():
    from types import SimpleNamespace
    from core.ai.replay_buffer import ReplayBuffer, HOLDOUT_EMBARGO_SEC
    t0 = 1_792_648_800
    # 3 days at ~102 records / hour: the holdout is exactly the newest 24h
    rows = [SimpleNamespace(timestamp=t0 + i * 35) for i in range(7400)]
    train, holdout = ReplayBuffer.split_holdout(rows)
    span = holdout[-1].timestamp - holdout[0].timestamp
    assert 23.9 * 3600 <= span <= 24 * 3600
    assert train and train[-1].timestamp < holdout[0].timestamp - HOLDOUT_EMBARGO_SEC
    # 34 hours (the refit on 23 Oct): 24h would be 70% of the rows, so the 60% cap applies (~20h)
    rows = rows[:3500]
    train, holdout = ReplayBuffer.split_holdout(rows)
    assert len(holdout) == int(3500 * 0.6) and train
    # Little data: the 60% cap keeps training rows
    few = [SimpleNamespace(timestamp=t0 + i * 600) for i in range(200)]
    train, holdout = ReplayBuffer.split_holdout(few)
    assert len(holdout) == 120 and len(train) > 0
    assert ReplayBuffer.split_holdout(few[:50]) == (few[:50], [])


def test_tiny_golden_pool_is_not_repeated_in_every_batch():
    import random
    from types import SimpleNamespace
    from core.ai.replay_buffer import ReplayBuffer, GOLDEN_MIN_POOL
    b = ReplayBuffer.__new__(ReplayBuffer)
    b.cache = [SimpleNamespace(id=i, reward=0.0, timestamp=i) for i in range(500)]
    b.golden_cache = [SimpleNamespace(id=10_000 + i, reward=0.0, timestamp=i, _src="gold") for i in range(5)]
    b.sumtree = SimpleNamespace(n_entries=0)
    random.seed(0)
    batch = b.sample(64)
    assert not any(getattr(e, "_src", None) == "gold" for e in batch)   # 5 < GOLDEN_MIN_POOL: not used
    b.golden_cache = [SimpleNamespace(id=10_000 + i, reward=0.0, timestamp=i, _src="gold") for i in range(GOLDEN_MIN_POOL)]
    assert sum(getattr(e, "_src", None) == "gold" for e in b.sample(64)) == 6


def test_newest_rows_get_no_extra_weight_when_history_is_loaded():
    import random
    from types import SimpleNamespace
    from core.ai.replay_buffer import ReplayBuffer
    b = ReplayBuffer.__new__(ReplayBuffer)
    hist = [SimpleNamespace(id=i, reward=0.0, timestamp=i, _src="replay") for i in range(800)]
    live = [SimpleNamespace(id=10_000 + i, reward=0.0, timestamp=10_000 + i) for i in range(200)]
    b.cache, b.replay_cache, b.golden_cache = hist + live, hist, []
    b.sumtree = SimpleNamespace(n_entries=0)
    random.seed(1)
    share = sum(sum(getattr(e, "_src", None) != "replay" for e in b.sample(64)) for _ in range(200)) / (200 * 64)
    assert 0.12 < share < 0.30   # ~20% = live rows' share of the pool (was ~45% with the 30% "recent" slice)
