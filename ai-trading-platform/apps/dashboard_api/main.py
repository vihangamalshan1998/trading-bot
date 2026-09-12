from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
from core.db.redis import redis_manager

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

@app.get("/api/state")
async def get_system_state():
    """
    Returns the latest system state by polling Redis.
    (In a real app, we'd use WebSockets for push).
    """
    # Mocking for Phase 12 completion
    return {
        "equity": 10500.25,
        "positions": [
            {"symbol": "BTCUSDT", "side": "LONG", "quantity": 0.5, "pnl": 125.50},
            {"symbol": "ETHUSDT", "side": "SHORT", "quantity": 10.0, "pnl": -45.20}
        ],
        "market_states": {
            "BTCUSDT": {"price": 65120.50, "trend": "UP"},
            "ETHUSDT": {"price": 3490.10, "trend": "DOWN"}
        }
    }

if __name__ == "__main__":
    uvicorn.run("apps.dashboard_api.main:app", host="0.0.0.0", port=8000, reload=True)
