import asyncio
import json
import aiohttp
import feedparser
import time
from core.logging.logger import logger
from core.config.settings import settings
from core.db.redis import redis_manager

class MacroCollectorService:
    def __init__(self):
        self.running = False
        self._task = None
        # Cointelegraph & Global Macro RSS
        self.rss_urls = [
            "https://cointelegraph.com/rss",          # Crypto Specific News
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", # WSJ Global Markets
            "https://finance.yahoo.com/news/rss"      # Yahoo Finance Global News
        ]
        
    async def fetch_news(self) -> str:
        """Fetches the latest headlines from RSS feeds and concatenates them."""
        headlines = []
        # Since feedparser is synchronous and uses urllib, we run it in a thread to avoid blocking.
        # But wait, we can just use feedparser.parse directly on a URL, but it blocks. 
        # Better: use aiohttp to fetch XML, then feedparser to parse XML string.
        async with aiohttp.ClientSession() as session:
            for url in self.rss_urls:
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            xml_data = await response.text()
                            feed = feedparser.parse(xml_data)
                            for entry in feed.entries[:10]: # Top 10 per feed
                                title = entry.get('title', '')
                                if title:
                                    headlines.append(title)
                except Exception as e:
                    logger.error(f"Failed to fetch RSS feed {url}: {e}")
                    
        return "\n".join(headlines)

    async def analyze_sentiment(self, text: str) -> dict:
        """Calls Gemini API to calculate macro sentiment score."""
        if not settings.gemini_api_key or not text:
            return {"sentiment_score": 0.0, "regime": 0}
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model_version}:generateContent?key={settings.gemini_api_key}"
        
        prompt = f"""
You are a professional cryptocurrency quantitative macro analyst. 
Analyze the following latest crypto and global macroeconomic news headlines.
Determine the overall market sentiment score between -1.0 (extremely bearish) and 1.0 (extremely bullish).
Also determine the current market regime (-1 for bearish, 1 for bullish, 0 for neutral).
Output ONLY a raw JSON object with NO markdown formatting, NO backticks, and NO extra text.
Format: {{"sentiment_score": 0.5, "regime": 1}}

Headlines:
{text}
"""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            # Extract text
                            content_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            # Clean markdown if Gemini still added it
                            clean_text = content_text.strip().replace("```json", "").replace("```", "").strip()
                            result = json.loads(clean_text)
                            return {
                                "sentiment_score": float(result.get("sentiment_score", 0.0)),
                                "regime": int(result.get("regime", 0))
                            }
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API returned {response.status}: {error_text}")
            except Exception as e:
                logger.error(f"Failed to analyze sentiment with Gemini: {e}")
                
        return {"sentiment_score": 0.0, "regime": 0}

    async def _loop(self):
        while self.running:
            logger.info("MacroCollector: Fetching latest news...")
            news_text = await self.fetch_news()
            
            if news_text:
                logger.info(f"MacroCollector: Found {len(news_text.splitlines())} headlines. Analyzing with Gemini...")
                analysis = await self.analyze_sentiment(news_text)
                
                # Add headlines to payload for dashboard
                analysis["headlines"] = news_text.split("\n")
                
                logger.info(f"MacroCollector: Analysis complete: {analysis}")
                
                # Publish to Redis
                if redis_manager.redis:
                    await redis_manager.redis.publish("macro:state:global", json.dumps(analysis))
                    logger.info("MacroCollector: Published to Redis.")
            else:
                logger.warning("MacroCollector: No news fetched.")
                
            # Sleep for 5 minutes
            for _ in range(300):
                if not self.running:
                    break
                await asyncio.sleep(1)

    async def start(self):
        logger.info("Starting Gemini Macro Collector Service...")
        self.running = True
        await redis_manager.connect()
        
        self._task = asyncio.create_task(self._loop())
        
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Macro Collector Service shutting down...")
        finally:
            self._task.cancel()
            await redis_manager.disconnect()

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
