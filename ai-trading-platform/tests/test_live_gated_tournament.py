from types import SimpleNamespace

import torch

from apps.research.model import SingleSymbolActorCritic
from apps.trainer.ppo import LIVE_ENTRY_HURDLE_PCT, MIN_GATED_TRADES, ROUND_TRIP_COST_PCT, PPOTrainer

H = LIVE_ENTRY_HURDLE_PCT


def _trainer():
    t = PPOTrainer.__new__(PPOTrainer)
    t.model = SingleSymbolActorCritic(input_dim=200)
    return t


def test_live_entry_rule_mirrors_the_bot():
    vals = torch.tensor([
        [0.0, H * 1.2, H * 1.2],      # clear LONG forecast
        [0.0, -H * 1.2, -H * 1.2],    # clear SHORT forecast
        [0.0, H * 0.5, H * 0.5],      # below the hurdle: no trade
        [0.0, -0.25, 3.0],            # blended forecast up, but 1h opposes the long: no trade
        [0.0, 0.25, -3.0],            # blended forecast down, but 1h opposes the short: no trade
    ])
    pos = PPOTrainer.live_entry_positions(vals)[:, 0].tolist()
    assert pos == [1.0, -1.0, 0.0, 0.0, 0.0]


def _res(pos, pnl, conf):
    pos = torch.tensor(pos, dtype=torch.float32).unsqueeze(1)
    pnl = torch.tensor(pnl, dtype=torch.float32).unsqueeze(1)
    conf = torch.tensor(conf, dtype=torch.float32)
    acts = torch.zeros(len(pos), 4)
    acts[:, 1] = conf * 2.0 - 1.0          # the trainer maps conf = (a1 + 1) / 2
    return {"live_pos": pos, "live_pnl": pnl, "acts": acts, "conf": conf}


def test_threshold_is_chosen_on_part_a_and_graded_on_part_b():
    # Part A: high-confidence trades win, so A picks a high threshold.
    # Part B: the same high-confidence trades LOSE -> the honest Part B result must show the loss.
    n_a = 80
    conf = ([0.9] * 40 + [0.6] * 40) + ([0.9] * 40 + [0.6] * 40)
    pnl = ([0.5] * 40 + [-0.5] * 40) + ([-0.4] * 40 + [0.5] * 40)
    res = _res([1] * 160, pnl, conf)
    out = _trainer()._live_split_score(res, ["BTCUSDT"] * 160, n_a)

    assert out["policy"]["min_confidence"] >= 0.89          # chosen on A
    assert out["policy"]["selection_pnl_per_trade"] > 0       # looked great on A (in-sample)
    assert out["n_trades"] == 40                              # B graded with A's threshold
    assert out["pnl_per_trade"] < 0                           # ...and the honest B result is a loss
    assert out["policy"]["filtered_pnl_per_trade"] < 0        # the number shipped to the live bot is the honest one


class _Buffer:
    holdout_start_ts = 0.0

    def __init__(self, n):
        self.rows = [SimpleNamespace(timestamp=n - i, symbol="BTCUSDT") for i in range(n)]
        self.last_symbols = []

    def sample_holdout(self, k):
        return list(self.rows)

    def build_tensors(self, batch, seq_len=300):
        self.last_symbols = [b.symbol for b in batch]
        n = len(batch)
        return torch.zeros(n, 2, 200), torch.zeros(n, 4), torch.zeros(n, 5), None, None


def _scored(live_pos, live_pnl, conf):
    r = _res(live_pos, live_pnl, conf)
    n = len(live_pos)
    raw_pos = torch.tensor([[1.0], [-1.0]] * (n // 2))
    r.update({"vals": torch.zeros(n, 3), "pnl": torch.zeros(n, 1), "pos": raw_pos, "sim_pnl": 0.0, "total_pnl": 0.0,
              "n_trades": n, "trade_rate": 1.0, "win_rate": 0.5, "long_share": 0.5, "critic_mse": 1.0,
              "critic_skill": [0.0, 0.0, 0.0]})
    return r


def _run(tmp_path, monkeypatch, chall, champ):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "production").mkdir(parents=True)
    torch.save(SingleSymbolActorCritic(input_dim=200).state_dict(), tmp_path / "models" / "production" / "model_v1.pt")
    t = _trainer()
    t.buffer = _Buffer(len(chall["live_pos"]))
    calls = iter([chall, champ])
    monkeypatch.setattr(t, "_score_policy", lambda model, states, rewards: next(calls))
    return t.evaluate_candidate(sample_size=1000)


def test_tournament_approves_a_brain_that_wins_as_live(tmp_path, monkeypatch):
    n = 200
    # challenger: 40 trades per half, alternating LONG / SHORT, each +0.3% after costs
    pos = ([1, -1] * 20 + [0] * 60) * 2
    chall = _scored(pos, [0.3 if p else 0.0 for p in pos], [0.8] * n)
    champ = _scored(pos, [-0.2 if p else 0.0 for p in pos], [0.8] * n)
    res = _run(tmp_path, monkeypatch, chall, champ)
    assert res["approved"] is True, res.get("reason")
    assert res["evaluator_version"] == "v3_live_gated_split"
    assert res["challenger_trades"] == 40 and res["challenger_pnl_per_trade"] > 0


def test_tournament_rejects_too_few_live_trades(tmp_path, monkeypatch):
    n = 200
    few = MIN_GATED_TRADES // 2
    pos = ([1] * few + [0] * (100 - few)) * 2
    chall = _scored(pos, [0.3 if p else 0.0 for p in pos], [0.8] * n)
    champ = _scored([0] * n, [0.0] * n, [0.8] * n)
    res = _run(tmp_path, monkeypatch, chall, champ)
    assert res["approved"] is False and res["reason"] == "TOO_FEW_LIVE_TRADES"


def test_tournament_rejects_a_brain_that_only_won_in_part_a(tmp_path, monkeypatch):
    # Wins on the older half (where the filters are chosen), loses on the newer half (the honest grade)
    n = 200
    half = [1, -1] * 20 + [0] * 60
    pos = half * 2
    pnl = [0.3 if p else 0.0 for p in half] + [-ROUND_TRIP_COST_PCT if p else 0.0 for p in half]
    chall = _scored(pos, pnl, [0.8] * n)
    champ = _scored([0] * n, [0.0] * n, [0.8] * n)
    res = _run(tmp_path, monkeypatch, chall, champ)
    assert res["approved"] is False and res["reason"] == "NO_EDGE_AFTER_COSTS"
