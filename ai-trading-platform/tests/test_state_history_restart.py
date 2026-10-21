import asyncio
import collections
import time
from types import SimpleNamespace

import apps.trading_bot.main as bot_main


class FakeBinaryRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


def _bot(rb, seq_len=4):
    b = bot_main.ProductionTradingBot.__new__(bot_main.ProductionTradingBot)
    b.redis = SimpleNamespace(binary_redis=rb, redis=None)
    b.sequence_length = seq_len
    b.state_history = collections.defaultdict(lambda: collections.deque(maxlen=seq_len))
    b._last_history_save = 0.0
    b._warmup_logged = set()
    return b


def _frame(v):
    return [float(v)] * 200


def test_history_survives_a_quick_restart():
    rb = FakeBinaryRedis()

    async def run():
        before = _bot(rb)
        for i in range(4):
            before.state_history["XRPUSDT"].append(_frame(i + 0.5))
        before.state_history["BTCUSDT"].append(_frame(9))
        await before._save_state_history()

        after = _bot(rb)
        await after._load_state_history()
        return after

    after = asyncio.run(run())
    assert [f[0] for f in after.state_history["XRPUSDT"]] == [0.5, 1.5, 2.5, 3.5]
    assert after._history_warm("XRPUSDT") is True
    assert after._history_warm("BTCUSDT") is False      # only 1 of 4 frames: still warming up


def test_old_history_is_not_restored_and_ai_waits():
    rb = FakeBinaryRedis()

    async def run():
        before = _bot(rb)
        for i in range(4):
            before.state_history["XRPUSDT"].append(_frame(i))
        await before._save_state_history()
        import msgpack
        data = msgpack.unpackb(rb.store[bot_main.STATE_HISTORY_KEY], raw=False)
        data["ts"] = time.time() - bot_main.STATE_HISTORY_MAX_AGE_SECONDS - 60
        rb.store[bot_main.STATE_HISTORY_KEY] = msgpack.packb(data, use_bin_type=True)

        after = _bot(rb)
        await after._load_state_history()
        return after

    after = asyncio.run(run())
    assert len(after.state_history["XRPUSDT"]) == 0
    assert after._history_warm("XRPUSDT") is False


def test_no_saved_history_means_warm_up():
    async def run():
        b = _bot(FakeBinaryRedis())
        await b._load_state_history()
        return b

    b = asyncio.run(run())
    assert b._history_warm("ETHUSDT") is False
