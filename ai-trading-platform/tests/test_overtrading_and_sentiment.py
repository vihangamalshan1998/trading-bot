import sys
import time
import types
import collections

sys.modules.setdefault("feedparser", types.ModuleType("feedparser"))

from apps.market_collector.news_macro import aggregate_headline_scores, parse_headline_scores
from apps.trading_bot.main import ProductionTradingBot
from core.config.settings import settings
from core.risk import exit_policy


def test_same_headlines_give_same_sentiment():
    cache = {
        "ETF inflows hit record": {"s": 0.6, "v": 0.3, "src": "gemini"},
        "Exchange hacked": {"s": -0.8, "v": 0.9, "src": "gemini"},
        "Weekly recap": {"s": 0.0, "v": 0.2, "src": "lexicon"},
    }
    titles = list(cache)
    first = aggregate_headline_scores(cache, titles)
    second = aggregate_headline_scores(cache, list(reversed(titles)))
    assert first[:2] == second[:2]
    assert abs(first[0] - (-0.2 / 3)) < 1e-3
    assert aggregate_headline_scores(cache, ["unknown"])[:2] == (0.0, 0.5)


def test_parse_headline_scores_clamps_and_ignores_bad_rows():
    raw = '```json\n[{"i": 0, "s": 3.0, "v": -1}, {"i": 7, "s": 0.5}, {"i": 1, "s": -0.25, "v": 0.4}]\n```'
    scores = parse_headline_scores(raw, ["a", "b"])
    assert scores == {"a": (1.0, 0.0), "b": (-0.25, 0.4)}


def test_hourly_entry_cap_counts_only_last_hour():
    bot = types.SimpleNamespace(recent_entry_times=collections.deque())
    now = time.time()
    bot.recent_entry_times.extend([now - 4000, now - 3700, now - 100, now - 10])
    assert ProductionTradingBot._entries_last_hour(bot) == 2
    assert len(bot.recent_entry_times) == 2


def test_entry_hurdle_is_at_least_twice_round_trip_cost():
    hurdle = max(settings.min_open_expected_return_pct,
                 settings.min_edge_cost_multiple * exit_policy.ROUND_TRIP_COST * 100.0)
    assert hurdle >= 2.0 * exit_policy.ROUND_TRIP_COST * 100.0
    assert settings.entry_cooldown_seconds >= 300
    assert settings.max_entries_per_hour >= 1
