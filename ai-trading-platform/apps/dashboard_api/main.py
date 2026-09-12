from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import asyncio
from core.db.redis import redis_manager

# Cache the last 50 metrics in memory
training_metrics = []

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
    asyncio.create_task(listen_training_metrics())

async def listen_training_metrics():
    pubsub = redis_manager.redis.pubsub()
    await pubsub.subscribe("training:metrics")
    async for message in pubsub.listen():
        if message["type"] == "message":
            data = json.loads(message["data"])
            training_metrics.append(data)
            if len(training_metrics) > 50:
                training_metrics.pop(0)

@app.get("/api/training")
async def get_training_metrics():
    return {"metrics": training_metrics}

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
    
    # Get market states
    market_states = {}
    keys = await redis_manager.redis.keys("dashboard:market_states:*")
    for key in keys:
        symbol = key.decode("utf-8").split(":")[-1] if isinstance(key, bytes) else key.split(":")[-1]
        state_raw = await redis_manager.redis.get(key)
        if state_raw:
            market_states[symbol] = json.loads(state_raw)
            
    return {
        "equity": portfolio.get("equity", 0.0),
        "positions": portfolio.get("positions", []),
        "market_states": market_states
    }

if __name__ == "__main__":
    uvicorn.run("apps.dashboard_api.main:app", host="0.0.0.0", port=8000, reload=True)
