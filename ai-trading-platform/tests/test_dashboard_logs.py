import asyncio
import time

from apps.dashboard_api import main as api


def test_services_listing_keeps_cached_log_paths(monkeypatch, tmp_path):
    log = tmp_path / "liquidation-hunter-out.log"
    log.write_text("line 1\nline 2\n")
    procs = [{"name": "liquidation-hunter", "status": "online", "restarts": 0, "uptime_since": 0,
              "cpu": 0, "memory_mb": 0, "out_log": str(log), "err_log": None}]
    monkeypatch.setitem(api._pm2_cache, "ts", time.time() + 3600)
    monkeypatch.setitem(api._pm2_cache, "procs", procs)

    async def no_redis(key, default=None):
        return default
    monkeypatch.setattr(api, "_redis_json", no_redis)

    listed = asyncio.run(api.get_services())
    assert all("out_log" not in s for s in listed["services"])  # paths never reach the browser
    # ...but the cached list still has them, so the log viewer keeps working after a services refresh
    out = asyncio.run(api.get_service_logs("liquidation-hunter", "out", 50))
    assert out["ok"] is True and out["logs"] == ["line 1", "line 2"]
