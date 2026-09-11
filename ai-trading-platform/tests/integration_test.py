import asyncio
import json
import numpy as np
from core.db.redis import redis_manager
from core.logging.logger import logger
from core.risk.manager import RiskManager

async def test_risk_manager_override():
    logger.info("=== Running Risk Manager Override Test ===")
    risk_manager = RiskManager(max_daily_drawdown=0.05, bearish_macro_threshold=-0.8)
    
    # Mock states
    portfolio_state = {
        "wallet_balance": 9000.0,
        "start_of_day_balance": 10000.0, # 10% drawdown
        "positions": {"BTCUSDT": 1.0}
    }
    
    macro_state = {
        "sentiment_score": -0.9, # Extreme bearish
        "volatility_expectation": 0.8
    }
    
    ai_action = 1 # OPEN_LONG (AI is hallucinating that we should buy more)
    
    logger.info(f"AI intends to OPEN_LONG (action={ai_action})")
    
    safe_action = risk_manager.validate_action(
        symbol="BTCUSDT",
        ai_action=ai_action,
        portfolio_state=portfolio_state,
        macro_state=macro_state
    )
    
    # We expect 3 (CLOSE_LONG) because drawdown is -10% and we have a long position
    assert safe_action == 3, f"Expected action 3 (CLOSE_LONG), got {safe_action}"
    logger.info("Test Passed: Risk Manager correctly overrode OPEN_LONG to CLOSE_LONG due to drawdown.")

async def test_redis_pubsub_flow():
    logger.info("=== Running Redis Pub/Sub Payload Test ===")
    await redis_manager.connect()
    conn = redis_manager.redis
    
    pubsub = conn.pubsub()
    await pubsub.subscribe("macro:state:global")
    
    # Publish a payload
    payload = {"sentiment_score": 0.5, "volatility_expectation": 0.2}
    await conn.publish("macro:state:global", json.dumps(payload))
    
    # Receive it
    msg = None
    for _ in range(5):
        msg = await pubsub.get_message(ignore_subscribe_messages=True)
        if msg:
            break
        await asyncio.sleep(0.1)
        
    assert msg is not None, "Did not receive Redis message."
    received = json.loads(msg["data"])
    assert received["sentiment_score"] == 0.5, "Payload data mismatch."
    logger.info("Test Passed: Redis Pub/Sub flow is operational.")

async def run_all_tests():
    await test_risk_manager_override()
    await test_redis_pubsub_flow()
    logger.info("All Phase 11 Integration Tests Passed.")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
