import asyncio
import collections
import time
from types import SimpleNamespace

import apps.trading_bot.main as bot_main


class FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


def _bot(redis):
    b = bot_main.ProductionTradingBot.__new__(bot_main.ProductionTradingBot)
    b.redis = SimpleNamespace(redis=redis)
    b.recent_entry_times = collections.deque()
    b.last_close_time = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
    b._saved_entry_guard = None
    b._entry_guard_loaded = False
    return b


def test_hourly_cap_and_cooldowns_survive_restart():
    redis = FakeRedis()
    now = time.time()

    async def run():
        before = _bot(redis)
        before.recent_entry_times.extend([now - 4000, now - 600, now - 60])   # one is older than an hour
        before.last_close_time["BTCUSDT"] = now - 120
        await before._save_entry_guard()

        after = _bot(redis)                                                   # the restarted bot
        await after._load_entry_guard()
        return after

    after = asyncio.run(run())
    assert after._entries_last_hour() == 2
    assert after.last_close_time["BTCUSDT"] == now - 120
    assert after.last_close_time["ETHUSDT"] == 0.0


def test_missing_or_broken_state_starts_clean():
    redis = FakeRedis()
    redis.store[bot_main.ENTRY_GUARD_KEY] = "not json"

    async def run():
        b = _bot(redis)
        await b._load_entry_guard()
        empty = _bot(FakeRedis())
        await empty._load_entry_guard()
        return b, empty

    b, empty = asyncio.run(run())
    assert b._entries_last_hour() == 0 and empty._entries_last_hour() == 0


def test_late_redis_merges_instead_of_overwriting_and_never_double_counts():
    redis = FakeRedis()
    now = time.time()

    async def run():
        before = _bot(redis)
        before.recent_entry_times.extend([now - 300, now - 200])
        await before._save_entry_guard()

        after = _bot(None)                      # restarted bot, Redis not connected yet
        await after._load_entry_guard()         # skipped
        after.recent_entry_times.append(now - 10)
        after.redis = SimpleNamespace(redis=redis)
        for _ in range(3):                      # inference ticks
            await after._save_entry_guard()
        return after

    after = asyncio.run(run())
    assert after._entries_last_hour() == 3
