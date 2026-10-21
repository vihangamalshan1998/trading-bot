from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import math
import os
import re
import subprocess
import threading
import time
import uvicorn
import asyncio
from fastapi.responses import FileResponse
from core.db.redis import redis_manager

# Cache the last 50 metrics in memory (only filled when a trainer publishes training:metrics)
training_metrics = []
system_stats = {
    "news_count": 0,
    "latest_sentiment": 0.0,
    "latest_regime": 0.0,
    "model_update_count": 0,
    "last_model_update_time": None,
    "latest_headlines": []
}

MODEL_DIR = "models/production"
TRAINER_SERVICE = "ai-trainer"          # pseudo-service: the laptop trainer reports through Redis
TRAINER_STATUS_KEY = "trainer:status"
TRAINER_LOG_KEY = "trainer:log_tail"
GLOBAL_CONTEXT_KEY = "market:global_context"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
LOG_TAIL_BYTES = 768 * 1024

app = FastAPI(title="AI Trading Dashboard API")

# Allow React app to fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _finite(val, default=0.0):
    if val is None or not isinstance(val, (int, float)):
        return default
    return default if (math.isnan(val) or math.isinf(val)) else val


async def _redis_json(key: str, default=None):
    if not redis_manager.redis:
        return default
    try:
        raw = await redis_manager.redis.get(key)
        return json.loads(raw) if raw else default
    except Exception:
        return default


async def hydrate_macro_from_cache():
    """Hydrates in-memory system_stats from persistent Redis cache if available"""
    macro_data = await _redis_json("dashboard:macro_state")
    if macro_data:
        system_stats["latest_sentiment"] = float(macro_data.get("sentiment_score", 0.0))
        system_stats["latest_regime"] = float(macro_data.get("regime", 0.0))
        system_stats["latest_headlines"] = macro_data.get("headlines", [])
        system_stats["news_count"] = int(macro_data.get("news_count", len(system_stats["latest_headlines"])))


@app.on_event("startup")
async def startup():
    await redis_manager.connect()
    await hydrate_macro_from_cache()
    asyncio.create_task(listen_redis_events())


async def listen_redis_events():
    """Background task to listen to Redis for training metrics and system events"""
    await redis_manager.connect()

    if not redis_manager.redis:
        return

    pubsub = redis_manager.redis.pubsub()
    await pubsub.subscribe("training:metrics", "macro:state:global", "training:model_update")
    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                channel = message["channel"].decode("utf-8") if isinstance(message["channel"], bytes) else message["channel"]
                data = json.loads(message["data"])

                if channel == "training:metrics":
                    training_metrics.append(data)
                    if len(training_metrics) > 50:
                        training_metrics.pop(0)
                elif channel == "macro:state:global":
                    headlines = data.get("headlines", [])
                    system_stats["latest_sentiment"] = float(data.get("sentiment_score", 0.0))
                    system_stats["latest_regime"] = float(data.get("regime", 0.0))
                    system_stats["latest_headlines"] = headlines
                    system_stats["news_count"] = int(data.get("news_count") or len(headlines) or (system_stats["news_count"] + 1))
                elif channel == "training:model_update":
                    system_stats["model_update_count"] += 1
                    system_stats["last_model_update_time"] = data.get("timestamp")
            except Exception:
                pass


@app.get("/api/training")
async def get_training_metrics():
    return {"metrics": training_metrics}


from core.database.session import SessionLocal
from core.database.models.ai import Experience
from sqlalchemy import func

_experience_count = {"value": 0, "ts": 0.0}


def _cached_experience_count() -> int:
    """COUNT(*) on the experiences table is slow; refresh at most every 2 minutes (was every second)."""
    if time.time() - _experience_count["ts"] > 120:
        try:
            with SessionLocal() as session:
                _experience_count["value"] = int(session.query(func.count(Experience.id)).scalar() or 0)
        except Exception as e:
            print(f"Failed to fetch total experiences: {e}")
        _experience_count["ts"] = time.time()
    return _experience_count["value"]


@app.get("/api/system_stats")
async def get_system_stats_api():
    if not system_stats.get("latest_headlines") or system_stats.get("news_count", 0) == 0:
        await hydrate_macro_from_cache()

    macro = await _redis_json("dashboard:macro_state", {}) or {}
    macro.pop("headlines", None)
    kill = await redis_manager.redis.get("system:kill_switch:active") if redis_manager.redis else None
    return {
        **system_stats,
        "total_experiences": await asyncio.to_thread(_cached_experience_count),
        "macro": macro,
        "global_context": await _redis_json(GLOBAL_CONTEXT_KEY, {}) or {},
        "kill_switch_active": str(kill).lower() == "true",
    }


@app.get("/api/state")
async def get_system_state():
    """
    Returns the latest system state by polling Redis for the live data.
    """
    if not redis_manager.redis:
        return {"equity": 0.0, "positions": [], "market_states": {}}

    # Get portfolio
    portfolio_raw = await redis_manager.redis.get("dashboard:portfolio")
    portfolio = json.loads(portfolio_raw) if portfolio_raw else {"equity": 0.0, "positions": []}

    # Get market states and AI states
    market_states = {}
    keys = await redis_manager.redis.keys("market:state:*")

    import msgpack

    for key in keys:
        symbol = key.decode("utf-8").split(":")[-1] if isinstance(key, bytes) else key.split(":")[-1]
        state_raw = await redis_manager.redis.get(key)

        # Fetch corresponding AI state
        ai_state_raw = await redis_manager.redis.get(f"ai:state:{symbol}")

        if state_raw:
            try:
                parsed_state = msgpack.unpackb(state_raw)
            except Exception:
                parsed_state = json.loads(state_raw)

            # Sanitize market features (e.g. if MACD or Bollinger Bands output NaN due to 0 volume)
            if "features" in parsed_state and isinstance(parsed_state["features"], list):
                parsed_state["features"] = [_finite(x) for x in parsed_state["features"]]

            ai_data = {}
            if ai_state_raw:
                try:
                    ai_data = msgpack.unpackb(ai_state_raw)
                except Exception:
                    ai_data = json.loads(ai_state_raw)

            parsed_state["ai_confidence"] = _finite(ai_data.get("confidence", 0.0))
            parsed_state["ai_confidence_calibrated"] = _finite(ai_data.get("confidence_calibrated", ai_data.get("confidence", 0.0)))
            parsed_state["ai_target_size"] = _finite(ai_data.get("target_size", 0.0))
            parsed_state["ai_predicted_side"] = ai_data.get("predicted_side", "WAITING")
            parsed_state["ai_critic_return"] = _finite(ai_data.get("critic_return", 0.0))
            parsed_state["ai_critic_1h"] = _finite(ai_data.get("critic_1h", 0.0))
            parsed_state["ai_critic_4h"] = _finite(ai_data.get("critic_4h", 0.0))
            parsed_state["ai_updated"] = _finite(ai_data.get("timestamp", 0.0))

            market_states[symbol] = parsed_state

    return {
        "equity": portfolio.get("equity", 0.0),
        "positions": portfolio.get("positions", []),
        "market_states": market_states
    }


@app.get("/api/history")
async def get_trade_history():
    """Returns the latest 100 AI trades."""
    if not redis_manager.redis:
        return {"history": []}

    raw_history = await redis_manager.redis.lrange("dashboard:trade_history", 0, -1)
    history = []
    for item in raw_history:
        try:
            parsed = json.loads(item)
            # The history table does not need the 200-float state vector of every trade
            parsed.pop("state_vector", None)
            for key in ["confidence", "price", "quantity", "entry_price", "realized_pnl", "roi_pct", "gross_pnl", "fees_paid",
                        "notional", "mid_price", "confidence_calibrated", "predicted_move_pct", "est_cost_pct", "hold_seconds"]:
                if isinstance(parsed.get(key), (int, float)):
                    parsed[key] = _finite(parsed[key])
            history.append(parsed)
        except Exception as e:
            print(f"Failed to parse trade history item: {e}")
    return {"history": history}


# ---------------------------------------------------------------------------
# Brain: champion, tournaments and the (remote) trainer's heartbeat
# ---------------------------------------------------------------------------
def _read_json_file(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


@app.get("/api/brain")
async def get_brain():
    version = _read_json_file(os.path.join(MODEL_DIR, "version.json"), {}) or {}
    metrics = version.get("metrics") or {}
    nested = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else {}
    evaluator = metrics.get("evaluator_version") or nested.get("evaluator_version") or ""
    policy = metrics.get("trade_policy") or nested.get("trade_policy") or {}

    history = _read_json_file(os.path.join(MODEL_DIR, "tournament_history.json"), []) or []
    v3 = [e for e in history if str(e.get("evaluator_version") or "").startswith("v3")]
    reasons = {}
    for e in v3:
        reasons[e.get("reason") or "UNKNOWN"] = reasons.get(e.get("reason") or "UNKNOWN", 0) + 1

    model_path = os.path.join(MODEL_DIR, "model_v1.pt")
    trainer = await _redis_json(TRAINER_STATUS_KEY)
    return {
        "champion": {
            "version": version.get("version", "unknown"),
            "promoted_at": version.get("promoted_at") or version.get("timestamp"),
            "model_updated_at": os.path.getmtime(model_path) if os.path.exists(model_path) else None,
            "evaluator": evaluator or "legacy",
            "is_v3": evaluator.startswith("v3"),
            "holdout_pnl": metrics.get("challenger_sim_pnl", nested.get("challenger_sim_pnl")),
            "trade_policy": policy if isinstance(policy, dict) else {},
        },
        "tournaments": [{k: e.get(k) for k in (
            "version", "date", "timestamp", "approved", "reason", "challenger_sim_pnl", "champion_sim_pnl",
            "challenger_trades", "champion_trades", "challenger_win_rate", "challenger_critic_skill",
            "champion_critic_skill", "blocked_symbols", "min_confidence", "sample_size")} for e in v3[-300:]],
        "tournament_reasons": reasons,
        "legacy_tournaments": len(history) - len(v3),
        "trainer": trainer,
    }


# ---------------------------------------------------------------------------
# Services: every pm2 process (+ the laptop trainer) with status and logs
# ---------------------------------------------------------------------------
_pm2_cache = {"ts": 0.0, "procs": [], "ok_ts": 0.0}
_pm2_lock = threading.Lock()


def _pm2_processes() -> list:
    # One `pm2 jlist` at a time; concurrent requests wait and reuse its result.
    with _pm2_lock:
        if time.time() - _pm2_cache["ts"] < 5:
            return _pm2_cache["procs"]
        return _pm2_refresh()


def _pm2_refresh() -> list:
    procs = []
    try:
        res = subprocess.run("pm2 jlist", shell=True, capture_output=True, text=True, timeout=15)
        if res.returncode == 0 and res.stdout.strip():
            for p in json.loads(res.stdout[res.stdout.find("["):]):
                env = p.get("pm2_env") or {}
                monit = p.get("monit") or {}
                procs.append({
                    "name": p.get("name"),
                    "status": env.get("status", "unknown"),
                    "restarts": env.get("restart_time", 0),
                    "uptime_since": (env.get("pm_uptime") or 0) / 1000.0,
                    "cpu": monit.get("cpu", 0),
                    "memory_mb": round((monit.get("memory") or 0) / 1048576, 1),
                    "out_log": env.get("pm_out_log_path"),
                    "err_log": env.get("pm_err_log_path"),
                })
    except Exception:
        procs = []
    if not procs and _pm2_cache["procs"] and time.time() - _pm2_cache.get("ok_ts", 0) < 300:
        # pm2 is slow or busy (e.g. during a restart): keep the last good list instead of
        # briefly "losing" every service, which made the log viewer blank out and come back.
        _pm2_cache["ts"] = time.time()
        return _pm2_cache["procs"]
    _pm2_cache.update(ts=time.time(), procs=procs, ok_ts=time.time())
    return procs


def _local_log_services() -> list:
    """Fallback without pm2 (local development): every logs/*.log file is a service."""
    out = []
    if os.path.isdir("logs"):
        for f in sorted(os.listdir("logs")):
            if f.endswith(".log"):
                path = os.path.join("logs", f)
                out.append({"name": f[:-4].replace("_", "-"), "status": "file", "restarts": 0, "uptime_since": os.path.getmtime(path),
                            "cpu": 0, "memory_mb": 0, "out_log": path, "err_log": None})
    return out


def _resolve_log(service: str, stream: str):
    """Absolute log path of a known service (never an arbitrary path)."""
    if not SAFE_NAME.match(service or ""):
        return None
    canon = service.replace("_", "-")
    for p in _pm2_processes() or _local_log_services():
        if (p.get("name") or "").replace("_", "-") == canon:
            return p.get("err_log" if stream == "err" else "out_log")
    local = os.path.join("logs", f"{service.replace('-', '_')}.log")
    return local if stream != "err" and os.path.exists(local) else None


def _tail(path: str, n: int) -> list:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - LOG_TAIL_BYTES))
        data = f.read().decode("utf-8", errors="replace")
    lines = data.splitlines()
    if size > LOG_TAIL_BYTES and lines:
        lines = lines[1:]  # first line may be cut
    return lines[-n:]


@app.get("/api/services")
async def get_services():
    procs = await asyncio.to_thread(_pm2_processes)
    # Copies: the log paths are stripped below, and the cached pm2 list must keep them for /api/logs
    services = [dict(p) for p in (procs or await asyncio.to_thread(_local_log_services))]
    trainer = await _redis_json(TRAINER_STATUS_KEY)
    if trainer:
        age = time.time() - float(trainer.get("timestamp", 0) or 0)
        services.append({"name": TRAINER_SERVICE, "status": "online" if age < 180 else "silent",
                         "restarts": 0, "uptime_since": trainer.get("started_at"), "cpu": None, "memory_mb": None,
                         "remote": True, "host": trainer.get("host"), "last_seen_s": round(age)})
    elif not any(s["name"] in (TRAINER_SERVICE, "ai-trainer") for s in services):
        services.append({"name": TRAINER_SERVICE, "status": "not-reporting", "restarts": 0, "uptime_since": None,
                         "cpu": None, "memory_mb": None, "remote": True})
    for s in services:
        s.pop("out_log", None)
        s.pop("err_log", None)
    return {"services": services, "pm2": bool(procs)}


@app.get("/api/logs/{service}")
async def get_service_logs(service: str, stream: str = "out", lines: int = 500):
    """Last `lines` lines of a service's log (pm2 out / err log, or the trainer's Redis tail)."""
    lines = max(10, min(int(lines), 3000))
    if service.replace("_", "-") == TRAINER_SERVICE:
        tail = await _redis_json(TRAINER_LOG_KEY)
        if tail:
            return {"logs": tail[-lines:], "source": "redis (laptop trainer)", "ok": True}
        local = os.path.join("logs", "ai_trainer.log")
        if os.path.exists(local):
            return {"logs": await asyncio.to_thread(_tail, local, lines), "source": local, "ok": True}
        return {"logs": ["The laptop trainer is not reporting. Start MASTER_TRAINER (its SSH tunnel carries the log here)."], "source": None, "ok": False}
    path = await asyncio.to_thread(_resolve_log, service, stream)
    if not path or not os.path.exists(path):
        return {"logs": [f"No {stream} log found for '{service}'."], "source": None, "ok": False}
    try:
        return {"logs": await asyncio.to_thread(_tail, path, lines), "source": path, "ok": True}
    except Exception as e:
        return {"logs": [f"Error reading logs: {e}"], "source": path, "ok": False}


@app.get("/api/logs/download/{service}")
async def download_service_logs(service: str, stream: str = "out"):
    """Downloads the full log file for the given service."""
    path = await asyncio.to_thread(_resolve_log, service, stream)
    if not path or not os.path.exists(path):
        return {"error": "Log file not found"}
    return FileResponse(path=path, filename=f"{service}-{stream}.log", media_type="text/plain")


# ---------------------------------------------------------------------------
# System controls
# ---------------------------------------------------------------------------
@app.post("/api/system/pm2/stop")
async def stop_pm2():
    """Emergency stop all trading by setting the Redis kill switch flag."""
    try:
        if redis_manager.redis:
            await redis_manager.redis.set("system:kill_switch:active", "true")
            # Also publish an event for immediate reaction
            await redis_manager.redis.publish("system:events", json.dumps({"type": "KILL_SWITCH"}))
            return {"status": "success", "message": "EMERGENCY KILL SWITCH ACTIVATED! Bots are panicking closing positions."}
        else:
            return {"status": "error", "message": "Failed to connect to Redis."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to activate kill switch: {str(e)}"}


@app.post("/api/system/pm2/resume")
async def resume_pm2():
    """Disarms the emergency kill switch and resumes trading."""
    try:
        if redis_manager.redis:
            await redis_manager.redis.set("system:kill_switch:active", "false")
            await redis_manager.redis.publish("system:events", json.dumps({"type": "RESUME_TRADING"}))
            return {"status": "success", "message": "Emergency Kill Switch disarmed. Trading resumed."}
        else:
            return {"status": "error", "message": "Failed to connect to Redis."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to disarm kill switch: {str(e)}"}


@app.post("/api/system/pm2/restart")
async def restart_pm2():
    """Restart all bot pm2 processes (not the dashboard). Keeps the kill switch as it is."""
    try:
        active_names = [p["name"] for p in _pm2_processes() if p.get("name") and "dashboard" not in p["name"]]
        _pm2_cache["ts"] = 0.0
        if active_names:
            res = subprocess.run(f"pm2 restart {' '.join(active_names)}", shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                return {"status": "success", "message": f"Successfully restarted: {', '.join(active_names)}"}
            return {"status": "error", "message": f"PM2 Error: {res.stderr}"}
        return {"status": "error", "message": "No pm2 processes found on this machine."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to restart bots: {str(e)}"}


@app.post("/api/system/pm2/restart/{service}")
async def restart_one(service: str):
    """Restart a single pm2 service."""
    if not SAFE_NAME.match(service or "") or "dashboard" in service:
        return {"status": "error", "message": "Invalid service."}
    if service not in [p["name"] for p in _pm2_processes()]:
        return {"status": "error", "message": f"Unknown pm2 service '{service}'."}
    res = subprocess.run(["pm2", "restart", service], capture_output=True, text=True, shell=os.name == "nt")
    _pm2_cache["ts"] = 0.0
    if res.returncode == 0:
        return {"status": "success", "message": f"Restarted {service}."}
    return {"status": "error", "message": f"PM2 Error: {res.stderr}"}


@app.post("/api/system/logs/clean")
async def clean_system_logs():
    """Prunes and cleans oversized logs across logs/ and PM2 logs directory."""
    try:
        from core.logging.logger import clean_old_logs
        clean_old_logs("logs", max_size_mb=10, max_age_days=1)
        if os.path.exists("/root/.pm2/logs"):
            clean_old_logs("/root/.pm2/logs", max_size_mb=10, max_age_days=1)
            subprocess.run("pm2 flush", shell=True, capture_output=True, text=True)
        return {"status": "success", "message": "All logs pruned and PM2 logs flushed successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to clean logs: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("apps.dashboard_api.main:app", host="0.0.0.0", port=8000)
