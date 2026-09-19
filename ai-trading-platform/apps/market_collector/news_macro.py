import asyncio
import json
import time
import feedparser
from core.db.redis import redis_manager
from core.logging.logger import logger
from core.schemas.state_schema import MacroState
from core.config.settings import settings

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
        Fetches RSS feeds and runs Gemini NLP sentiment analysis.
        """
        all_entries = []
        for url in self.rss_urls:
            try:
                # feedparser is synchronous, but fast enough for this infrequent polling
                feed = feedparser.parse(url)
                all_entries.extend(feed.entries[:5]) # Top 5 recent
            except Exception as e:
                logger.error(f"Error fetching RSS {url}: {e}")
                
        # Combine headlines
        headlines = [entry.title for entry in all_entries]
        combined_text = "\n".join(headlines)
        
        logger.info(f"📰 Fetched {len(headlines)} Headlines for Gemini: {headlines[:3]}...")
        
        if not combined_text or not settings.gemini_api_key:
            return MacroState(timestamp=time.time(), sentiment_score=0.0, volatility_expectation=0.5, regime=0.0)

        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=settings.gemini_api_key)
            prompt = (
                "You are a quantitative financial analyst. Read the following cryptocurrency headlines. "
                "Determine the overall market sentiment and expected volatility. "
                "Return a raw JSON object (and nothing else, no markdown) with two keys: "
                "'sentiment_score' (float from -1.0 for extreme bearish fear to 1.0 for extreme bullish greed) and "
                "'volatility_expectation' (float from 0.0 for calm to 1.0 for extreme panic/shock).\n\n"
                f"Headlines:\n{combined_text}"
            )
            
            # Using generate_content synchronously (wrapped in try/except)
            response = client.models.generate_content(
                model=settings.gemini_model_version,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            
            result = json.loads(response.text.strip('`').replace('json\n', ''))
            
            sentiment_score = float(result.get('sentiment_score', 0.0))
            volatility_expectation = float(result.get('volatility_expectation', 0.5))
            
            # Bound the values
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            volatility_expectation = max(0.0, min(1.0, volatility_expectation))
            
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            sentiment_score = 0.0
            volatility_expectation = 0.5
            
        # Determine macro regime based on sentiment
        regime = 1.0 if sentiment_score > 0.3 else (-1.0 if sentiment_score < -0.3 else 0.0)
        
        return MacroState(
            timestamp=time.time(),
            sentiment_score=sentiment_score,
            volatility_expectation=volatility_expectation,
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
