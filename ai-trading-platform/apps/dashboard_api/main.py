from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import asyncio
import os
from fastapi.responses import FileResponse
from core.db.redis import redis_manager

# Cache the last 50 metrics in memory
training_metrics = []
system_stats = {
    "news_count": 0,
    "latest_sentiment": 0.0,
    "latest_regime": 0.0,
    "model_update_count": 0,
    "last_model_update_time": None,
    "latest_headlines": []
}

app = FastAPI(title="AI Trading Dashboard API")

# Allow React app to fetch
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def hydrate_macro_from_cache():
    """Hydrates in-memory system_stats from persistent Redis cache if available"""
    if redis_manager.redis:
        try:
            cached = await redis_manager.redis.get("dashboard:macro_state")
            if cached:
                macro_data = json.loads(cached)
                system_stats["latest_sentiment"] = float(macro_data.get("sentiment_score", 0.0))
                system_stats["latest_regime"] = float(macro_data.get("regime", 0.0))
                system_stats["latest_headlines"] = macro_data.get("headlines", [])
                system_stats["news_count"] = int(macro_data.get("news_count", len(system_stats["latest_headlines"])))
        except Exception:
            pass

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

@app.get("/api/system_stats")
async def get_system_stats_api():
    if not system_stats.get("latest_headlines") or system_stats.get("news_count", 0) == 0:
        await hydrate_macro_from_cache()

    try:
        with SessionLocal() as session:
            count = session.query(Experience).count()
            system_stats["total_experiences"] = count
    except Exception as e:
        print(f"Failed to fetch total experiences: {e}")
        if "total_experiences" not in system_stats:
            system_stats["total_experiences"] = 0
            
    return system_stats

@app.get("/api/state")
async def get_system_state():
    """
    Returns the latest system state by polling Redis for the live data.
    """
    if not redis_manager.redis:
        return {"equity": 0.0, "positions": [], "market_states": {}}
        
    # Get portfolio
    portfolio_raw = await redis_manager.redis.get("dashboard:portfolio")
    portfolio = json.loads(portfolio_raw) if portfolio_raw else {"equity": 10000.0, "positions": []}
    
    # Get market states and AI states
    market_states = {}
    keys = await redis_manager.redis.keys("market:state:*")
    
    import msgpack
    import math
    
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
            
            # Sanitize floats to prevent FastAPI JSON serialization crashes
            def sanitize_float(val):
                if val is None or not isinstance(val, (int, float)): return 0.0
                return 0.0 if math.isnan(val) or math.isinf(val) else val
            
            # Sanitize market features (e.g. if MACD or Bollinger Bands output NaN due to 0 volume)
            if "features" in parsed_state and isinstance(parsed_state["features"], list):
                parsed_state["features"] = [sanitize_float(x) for x in parsed_state["features"]]
                
            if ai_state_raw:
                try:
                    ai_data = msgpack.unpackb(ai_state_raw)
                except Exception:
                    ai_data = json.loads(ai_state_raw)
                
                parsed_state["ai_confidence"] = sanitize_float(ai_data.get("confidence", 0.0))
                parsed_state["ai_target_size"] = sanitize_float(ai_data.get("target_size", 0.0))
                parsed_state["ai_predicted_side"] = ai_data.get("predicted_side", "WAITING")
                parsed_state["ai_critic_return"] = sanitize_float(ai_data.get("critic_return", 0.0))
                parsed_state["ai_critic_1h"] = sanitize_float(ai_data.get("critic_1h", 0.0))
                parsed_state["ai_critic_4h"] = sanitize_float(ai_data.get("critic_4h", 0.0))
                
                # We also need to sanitize the state_vector array if it exists
                raw_vector = ai_data.get("state_vector", [])
                parsed_state["state_vector"] = [sanitize_float(x) for x in raw_vector]
            else:
                parsed_state["ai_confidence"] = 0.0
                parsed_state["ai_target_size"] = 0.0
                parsed_state["ai_predicted_side"] = "WAITING"
                parsed_state["ai_critic_return"] = 0.0
                parsed_state["ai_critic_1h"] = 0.0
                parsed_state["ai_critic_4h"] = 0.0
                parsed_state["state_vector"] = []
                
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
    import math
    for item in raw_history:
        try:
            parsed = json.loads(item)
            if "state_vector" in parsed:
                # FATAL CPU FIX: The history table does not need to download 100x200=20,000 floats every second.
                # Deleting this saves massive CPU math loops and megabytes of bandwidth per second.
                del parsed["state_vector"]
            
            for key in ["confidence", "price", "quantity", "entry_price", "realized_pnl", "roi_pct"]:
                if key in parsed and parsed[key] is not None:
                    # Type check: math.isnan crashes on Strings. This was causing a silent exception loop
                    # and spamming the pm2 log writer, eating up 50% CPU!
                    if isinstance(parsed[key], (int, float)):
                        if math.isnan(parsed[key]) or math.isinf(parsed[key]):
                            parsed[key] = 0.0
                        
            history.append(parsed)
        except Exception as e:
            print(f"Failed to parse trade history item: {e}")
    return {"history": history}

@app.get("/api/logs/{bot_name}")
async def get_bot_logs(bot_name: str):
    """Returns the last 50 lines of the requested bot's log file."""
    allowed_bots = ["market_collector", "ai_trainer", "trading_bot", "whale_tracker", "statarb_collector", "reward_calculator", "teacher_assistant"]
    if bot_name not in allowed_bots:
        return {"logs": ["Invalid bot name requested."]}
        
    # Standard local path
    log_path = f"logs/{bot_name}.log"
    
    # If standard path doesn't exist, check PM2 log path (for VPS)
    if not os.path.exists(log_path):
        pm2_name = bot_name.replace('_', '-')
        pm2_path = f"/root/.pm2/logs/{pm2_name}-out.log"
        if os.path.exists(pm2_path):
            log_path = pm2_path
        else:
            return {"logs": [f"Log file not found at {log_path} or PM2 {pm2_path} (Bot may not have started yet)"]}
        
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return {"logs": lines[-1000:]}
    except Exception as e:
        return {"logs": [f"Error reading logs: {e}"]}

@app.get("/api/logs/download/{bot_name}")
async def download_bot_logs(bot_name: str):
    """Downloads the full log file for the given bot."""
    allowed_bots = ["market_collector", "ai_trainer", "trading_bot", "whale_tracker", "statarb_collector", "reward_calculator", "teacher_assistant"]
    if bot_name not in allowed_bots:
        return {"error": "Invalid bot name requested."}
        
    log_path = f"logs/{bot_name}.log"
    
    if not os.path.exists(log_path):
        pm2_name = bot_name.replace('_', '-')
        pm2_path = f"/root/.pm2/logs/{pm2_name}-out.log"
        if os.path.exists(pm2_path):
            log_path = pm2_path
        else:
            return {"error": "Log file not found"}
        
    return FileResponse(path=log_path, filename=f"{bot_name}.log", media_type="text/plain")

import subprocess

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
    """Restart all active bot pm2 processes dynamically without crashing on absent processes"""
    try:
        import subprocess
        if redis_manager.redis:
            await redis_manager.redis.set("system:kill_switch:active", "false")
            
        # Discover running PM2 processes dynamically via jlist
        jlist_res = subprocess.run("pm2 jlist", shell=True, capture_output=True, text=True)
        active_names = []
        if jlist_res.returncode == 0:
            try:
                pm2_procs = json.loads(jlist_res.stdout)
                active_names = [p.get("name") for p in pm2_procs if p.get("name") and "dashboard" not in p.get("name", "")]
            except Exception:
                pass
                
        # If jlist succeeded and found active bots, restart only those
        if active_names:
            bots_to_restart = " ".join(active_names)
            res = subprocess.run(f"pm2 restart {bots_to_restart}", shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                return {"status": "success", "message": f"Successfully restarted active bots: {', '.join(active_names)}"}
            else:
                return {"status": "error", "message": f"PM2 Error: {res.stderr}"}
        else:
            # Fallback: Restart known standard bots individually so absent ones don't break the rest
            known_bots = ["trading-bot", "whale-tracker", "market-collector", "statarb-collector", "macro-collector"]
            restarted = []
            for b in known_bots:
                r = subprocess.run(f"pm2 restart {b}", shell=True, capture_output=True, text=True)
                if r.returncode == 0:
                    restarted.append(b)
            return {"status": "success", "message": f"Restarted: {', '.join(restarted)}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to restart bots: {str(e)}"}

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
