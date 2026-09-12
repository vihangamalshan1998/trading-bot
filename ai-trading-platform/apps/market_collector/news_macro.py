import asyncio
import json
import time
import feedparser
from core.db.redis import redis_manager
from core.logging.logger import logger
from core.schemas.state_schema import MacroState

class NewsMacroCollector:
    """
    Phase 7: Real External Information Pipeline.
    Fetches real RSS feeds (e.g., ForexLive, CoinDesk) and broadcasts canonical MacroState.
    """
    def __init__(self):
        self.redis = redis_manager
        self.rss_urls = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss"
        ]
        self.running = False

    async def fetch_and_analyze(self) -> MacroState:
        """
        Fetches RSS feeds and runs simple heuristic sentiment analysis.
        (In a full ML pipeline, this would call a FinBERT microservice).
        """
        all_entries = []
        for url in self.rss_urls:
            try:
                # feedparser is synchronous, but fast enough for this infrequent polling
                feed = feedparser.parse(url)
                all_entries.extend(feed.entries[:5]) # Top 5 recent
            except Exception as e:
                logger.error(f"Error fetching RSS {url}: {e}")
                
        sentiment_score = 0.0
        bullish_keywords = ["surge", "bull", "adopt", "buy", "high", "growth", "approve"]
        bearish_keywords = ["crash", "bear", "ban", "sell", "low", "hack", "reject"]
        
        for entry in all_entries:
            text = (entry.title + " " + entry.get('summary', '')).lower()
            bull_count = sum(1 for word in bullish_keywords if word in text)
            bear_count = sum(1 for word in bearish_keywords if word in text)
            
            if bull_count > bear_count: sentiment_score += 0.2
            elif bear_count > bull_count: sentiment_score -= 0.2
            
        # Normalize to [-1, 1]
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Determine macro regime based on sentiment
        regime = 1.0 if sentiment_score > 0.3 else (-1.0 if sentiment_score < -0.3 else 0.0)
        
        return MacroState(
            timestamp=time.time(),
            sentiment_score=sentiment_score,
            volatility_expectation=0.5 + abs(sentiment_score) * 0.5, # High extreme sentiment = higher vol
            regime=regime
        )

    async def run(self):
        logger.info("Starting News/Macro Collector...")
        self.running = True
        await self.redis.connect()
        
        while self.running:
            try:
                macro_state = await self.fetch_and_analyze()
                
                # Broadcast Canonical MacroState
                await self.redis.redis.publish(
                    "macro:state:global", 
                    macro_state.json()
                )
                logger.info(f"Published MacroState: Sentiment {macro_state.sentiment_score:.2f}, Regime {macro_state.regime}")
                
            except Exception as e:
                logger.error(f"Error in NewsMacroCollector loop: {e}")
                
            await asyncio.sleep(300) # Poll every 5 minutes
            
if __name__ == "__main__":
    collector = NewsMacroCollector()
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        collector.running = False
