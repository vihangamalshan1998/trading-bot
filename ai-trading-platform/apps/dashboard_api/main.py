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

@app.on_event("startup")
async def startup():
    await redis_manager.connect()
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
                    system_stats["news_count"] += 1
                    system_stats["latest_sentiment"] = data.get("sentiment_score", 0.0)
                    system_stats["latest_regime"] = data.get("regime", 0.0)
                    system_stats["latest_headlines"] = data.get("headlines", [])
                elif channel == "training:model_update":
                    system_stats["model_update_count"] += 1
                    system_stats["last_model_update_time"] = data.get("timestamp")
            except Exception:
                pass

@app.get("/api/training")
async def get_training_metrics():
    return {"metrics": training_metrics}

@app.get("/api/system_stats")
async def get_system_stats_api():
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
    for key in keys:
        symbol = key.decode("utf-8").split(":")[-1] if isinstance(key, bytes) else key.split(":")[-1]
        state_raw = await redis_manager.redis.get(key)
        
        # Fetch corresponding AI state
        ai_state_raw = await redis_manager.redis.get(f"ai:state:{symbol}")
        
        if state_raw:
            parsed_state = json.loads(state_raw)
            if ai_state_raw:
                ai_data = json.loads(ai_state_raw)
                parsed_state["ai_confidence"] = ai_data.get("confidence", 0.0)
                parsed_state["ai_target_size"] = ai_data.get("target_size", 0.0)
                parsed_state["ai_predicted_side"] = ai_data.get("predicted_side", "WAITING")
                parsed_state["state_vector"] = ai_data.get("state_vector", [])
            else:
                parsed_state["ai_confidence"] = 0.0
                parsed_state["ai_target_size"] = 0.0
                parsed_state["ai_predicted_side"] = "WAITING"
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
    for item in raw_history:
        try:
            history.append(json.loads(item))
        except Exception:
            pass # ignore malformed records
    return {"history": history}

@app.get("/api/logs/{bot_name}")
async def get_bot_logs(bot_name: str):
    """Returns the last 50 lines of the requested bot's log file."""
    allowed_bots = ["market_collector", "ai_trainer", "trading_bot", "whale_tracker", "statarb_collector"]
    if bot_name not in allowed_bots:
        return {"logs": ["Invalid bot name requested."]}
        
    log_path = f"logs/{bot_name}.log"
    if not os.path.exists(log_path):
        return {"logs": [f"Log file not found: {log_path} (Bot may not have started yet)"]}
        
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return {"logs": lines[-1000:]}
    except Exception as e:
        return {"logs": [f"Error reading logs: {e}"]}

@app.get("/api/logs/download/{bot_name}")
async def download_bot_logs(bot_name: str):
    """Downloads the full log file for the given bot."""
    allowed_bots = ["market_collector", "ai_trainer", "trading_bot", "whale_tracker", "statarb_collector"]
    if bot_name not in allowed_bots:
        return {"error": "Invalid bot name requested."}
        
    log_path = f"logs/{bot_name}.log"
    if not os.path.exists(log_path):
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

@app.post("/api/system/pm2/restart")
async def restart_pm2():
    """Restart all pm2 processes except dashboard"""
    try:
        bots_to_restart = "market-collector macro-collector ai-trainer trading-bot"
        result = subprocess.run(f"pm2 restart {bots_to_restart}", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "success", "message": "All bots restarted successfully."}
        else:
            return {"status": "error", "message": f"PM2 Error: {result.stderr}"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to restart bots: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("apps.dashboard_api.main:app", host="0.0.0.0", port=8000, reload=True)
