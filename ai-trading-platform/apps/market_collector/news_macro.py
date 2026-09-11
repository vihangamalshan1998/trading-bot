import asyncio
import json
import random
import time
from core.db.redis import redis_manager
from core.logging.logger import logger
from apps.market_collector.sentiment_analyzer import SentimentAnalyzer

class NewsCollectorService:
    """
    Phase 10: Mocks the collection of global macroeconomic data and news sentiment.
    Uses the SentimentAnalyzer (LLM proxy) to process raw headlines.
    """
    def __init__(self, publish_interval: float = 5.0):
        self.publish_interval = publish_interval
        self.redis = redis_manager
        self.channel = "macro:state:global"
        self._running = False
        self.analyzer = SentimentAnalyzer()
        
        self.mock_headlines = [
            "Federal Reserve announces surprise emergency rate cut!",
            "Crypto markets crash as SEC bans Ethereum.",
            "Bitcoin adoption surges as institutional investors buy record amounts.",
            "Inflation drops unexpectedly, markets jump.",
            "Major crypto exchange hack causes widespread fear.",
            "Nothing much happening in the markets today.",
            "Tech stocks drop, dragging Bitcoin down.",
            "Breaking: SEC approves new crypto ETF!"
        ]
        
    async def start(self):
        self._running = True
        logger.info(f"NewsCollectorService started. Publishing to {self.channel}")
        
        # Connect to redis
        await self.redis.connect()
        redis_conn = self.redis.redis
        
        while self._running:
            try:
                # Simulate grabbing a new headline
                headline = random.choice(self.mock_headlines)
                
                # Analyze sentiment via LLM proxy
                analysis = self.analyzer.analyze(headline)
                
                payload = {
                    "timestamp": time.time(),
                    "headline": headline,
                    "sentiment_score": analysis["sentiment_score"],
                    "volatility_expectation": analysis["volatility_expectation"]
                }
                
                await redis_conn.publish(self.channel, json.dumps(payload))
                logger.debug(f"Published Macro State: {payload}")
                
                await asyncio.sleep(self.publish_interval)
                
            except Exception as e:
                logger.error(f"Error in NewsCollectorService: {e}")
                await asyncio.sleep(5.0)
                
    def stop(self):
        self._running = False
        logger.info("NewsCollectorService stopping...")

if __name__ == "__main__":
    # Standalone test runner
    async def main():
        service = NewsCollectorService(publish_interval=2.0)
        task = asyncio.create_task(service.start())
        await asyncio.sleep(10)
        service.stop()
        await task
        
    asyncio.run(main())
