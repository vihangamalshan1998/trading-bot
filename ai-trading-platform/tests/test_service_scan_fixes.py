from types import SimpleNamespace

import pandas as pd
import numpy as np

from apps.experience.reward_calculator import KlinePriceSource
from apps.market_collector.news_macro import _day_momentum


def test_macro_momentum_ignores_nan_closes():
    assert _day_momentum(pd.DataFrame({"Close": [100.0, 101.0, np.nan]})) == 0.01
    assert _day_momentum(pd.DataFrame({"Close": [np.nan]})) == 0.0
    assert _day_momentum(None) == 0.0


def test_kline_cache_is_pruned():
    src = KlinePriceSource()
    src.cache["BTCUSDT"] = {1_000: (1, 1, 1, 1), 5_000: (2, 2, 2, 2)}
    src.fetched_ranges["BTCUSDT"] = [(0, 2_000), (4_000, 6_000)]
    src.prune(3_000)
    assert list(src.cache["BTCUSDT"]) == [5_000]
    assert src.fetched_ranges["BTCUSDT"] == [(4_000, 6_000)]


def test_telegram_commands_only_from_owner_chat():
    from apps.telegram_bot.service import TelegramCommandCenter as TelegramBotService
    svc = TelegramBotService.__new__(TelegramBotService)
    svc.chat_id = "123456789"
    assert svc._authorized("123456789") is True
    assert svc._authorized("987654321") is False
    svc.chat_id = ""
    assert svc._authorized("") is False


def test_vps_push_renames_version_before_model(monkeypatch):
    import subprocess
    import apps.trainer.ppo as ppo
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ppo.push_champion_to_vps("v1_push_99") is True
    assert [c[0] for c in calls] == ["ssh", "scp", "ssh"]
    rename = calls[-1][-1]
    assert rename.index("version.json version.json") < rename.index("model_v1.pt model_v1.pt")


def test_vps_push_failure_is_reported(monkeypatch):
    import subprocess
    import apps.trainer.ppo as ppo
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=255, stderr="Connection refused"))
    assert ppo.push_champion_to_vps("v1_push_99") is False
