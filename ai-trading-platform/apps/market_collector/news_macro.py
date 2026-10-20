import asyncio
import json
import time
import re
import urllib.request
import feedparser
from core.db.redis import redis_manager
from core.logging.logger import logger
from core.schemas.state_schema import MacroState
from core.config.settings import settings

BULLISH_KEYWORDS = [
    "surge", "surges", "surging", "rally", "rallies", "bull", "bullish", "ath",
    "all-time high", "record", "profit", "profits", "gain", "gains", "jump", "jumps",
    "soar", "soars", "soaring", "breakout", "approved", "approval", "adoption",
    "accumulation", "bounce", "bounces", "inflow", "inflows", "buy", "buys", "buying"
]

BEARISH_KEYWORDS = [
    "crash", "crashes", "crashing", "dump", "dumps", "dumping", "plunge", "plunges",
    "drop", "drops", "dropping", "hack", "hacked", "exploit", "exploited", "lawsuit",
    "sec", "ban", "bans", "banned", "scam", "fraud", "crisis", "bleed", "bleeding",
    "fear", "panic", "liquidated", "liquidation", "fall", "falls", "falling",
    "down", "dip", "dips", "loss", "losses", "stolen", "penalty", "outflows"
]

class NewsMacroCollector:
    """
    Phase 7: Real External Information Pipeline.
    Fetches real RSS feeds (CoinDesk, CoinTelegraph, Decrypt) and broadcasts canonical MacroState.
    Features 3-tier sentiment engine: Gemini SDK -> Gemini REST API -> Financial Lexicon NLP.
    Persists canonical state in Redis key 'dashboard:macro_state' for immediate dashboard hydration.
    """
    def __init__(self):
        self.redis = redis_manager
        self.rss_urls = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://decrypt.co/feed"
        ]
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        self.running = False

    def _lexicon_nlp_sentiment(self, text: str) -> tuple[float, float]:
        """
        Rule-based financial lexicon NLP sentiment fallback.
        Returns (sentiment_score [-1.0, 1.0], volatility_expectation [0.0, 1.0]).
        """
        lower_text = text.lower()
        bull_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', lower_text)) for word in BULLISH_KEYWORDS)
        bear_hits = sum(len(re.findall(r'\b' + re.escape(word) + r'\b', lower_text)) for word in BEARISH_KEYWORDS)
        
        total_hits = bull_hits + bear_hits
        if total_hits == 0:
            return 0.0, 0.35
            
        sentiment_score = (bull_hits - bear_hits) / max(1, total_hits)
        sentiment_score = max(-1.0, min(1.0, round(sentiment_score, 2)))
        volatility_expectation = max(0.2, min(1.0, round(total_hits / 10.0, 2)))
        return sentiment_score, volatility_expectation

    def _gemini_rest_sentiment(self, prompt: str) -> tuple[float, float]:
        """Direct REST HTTP call to Gemini API, zero third-party SDK dependencies."""
        if not settings.gemini_api_key:
            raise ValueError("No GEMINI_API_KEY configured")
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model_version}:generateContent?key={settings.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                clean_json = raw_text.strip().replace("```json", "").replace("```", "").strip()
                result = json.loads(clean_json)
                s_score = float(result.get("sentiment_score", 0.0))
                v_exp = float(result.get("volatility_expectation", 0.5))
                return max(-1.0, min(1.0, s_score)), max(0.0, min(1.0, v_exp))
        raise ValueError("Invalid Gemini REST response structure")

    async def fetch_and_analyze(self) -> MacroState:
        """
        Fetches RSS feeds and runs multi-tier sentiment analysis.
        """
        headline_items = []
        for url in self.rss_urls:
            try:
                feed = feedparser.parse(url, agent=self.user_agent)
                for entry in feed.entries[:5]: # Top 5 recent per outlet
                    title = getattr(entry, "title", "").strip()
                    link = getattr(entry, "link", "")
                    if title:
                        headline_items.append({"title": title, "url": link})
            except Exception as e:
                logger.error(f"Error fetching RSS {url}: {e}")
                
        # Deduplicate headlines while preserving order
        seen_titles = set()
        unique_headlines = []
        for item in headline_items:
            if item["title"] not in seen_titles:
                seen_titles.add(item["title"])
                unique_headlines.append(item)
                
        combined_text = "\n".join([item["title"] for item in unique_headlines])
        logger.info(f"📰 Fetched {len(unique_headlines)} Unique Headlines for Sentiment Analysis")
        
        sentiment_score = 0.0
        volatility_expectation = 0.5
        sentiment_source = "Lexicon"

        prompt = (
            "You are a quantitative financial analyst. Read the following cryptocurrency headlines. "
            "Determine the overall market sentiment and expected volatility. "
            "Return a raw JSON object (and nothing else, no markdown) with two keys: "
            "'sentiment_score' (float from -1.0 for extreme bearish fear to 1.0 for extreme bullish greed) and "
            "'volatility_expectation' (float from 0.0 for calm to 1.0 for extreme panic/shock).\n\n"
            f"Headlines:\n{combined_text}"
        )

        # Tier 1: Try Gemini SDK
        if settings.gemini_api_key and combined_text:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=settings.gemini_api_key)
                response = client.models.generate_content(
                    model=settings.gemini_model_version,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                result = json.loads(response.text.strip('`').replace('json\n', ''))
                sentiment_score = float(result.get('sentiment_score', 0.0))
                volatility_expectation = float(result.get('volatility_expectation', 0.5))
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
                volatility_expectation = max(0.0, min(1.0, volatility_expectation))
                sentiment_source = "Gemini SDK"
            except Exception as sdk_err:
                # Tier 2: Try Gemini Direct REST API
                try:
                    sentiment_score, volatility_expectation = self._gemini_rest_sentiment(prompt)
                    sentiment_source = "Gemini REST"
                except Exception as rest_err:
                    logger.warning(f"Gemini API unavailable (SDK: {sdk_err}, REST: {rest_err}). Falling back to NLP Lexicon.")
                    sentiment_score, volatility_expectation = self._lexicon_nlp_sentiment(combined_text)
                    sentiment_source = "Lexicon Fallback"
        elif combined_text:
            # Tier 3: Lexicon NLP
            sentiment_score, volatility_expectation = self._lexicon_nlp_sentiment(combined_text)
            sentiment_source = "Lexicon Rule-Based"

        # Determine macro regime based on sentiment
        regime = 1.0 if sentiment_score > 0.15 else (-1.0 if sentiment_score < -0.15 else 0.0)
        logger.info(f"📊 Market Sentiment: {sentiment_score:.2f} ({sentiment_source}), Regime: {regime}")
        
        # --- Traditional Macro Data (S&P 500, DXY, VIX, Gold, Treasury Yield, NDX) ---
        sp500_mom = 0.0
        dxy_mom = 0.0
        vix_mom = 0.0
        gold_mom = 0.0
        tnx_mom = 0.0
        ndx_mom = 0.0
        try:
            import yfinance as yf
            tickers = yf.Tickers("^GSPC DX-Y.NYB ^VIX GC=F ^TNX ^NDX")
            sp_hist = tickers.tickers["^GSPC"].history(period="5d")
            dx_hist = tickers.tickers["DX-Y.NYB"].history(period="5d")
            vx_hist = tickers.tickers["^VIX"].history(period="5d")
            gl_hist = tickers.tickers["GC=F"].history(period="5d")
            tn_hist = tickers.tickers["^TNX"].history(period="5d")
            nx_hist = tickers.tickers["^NDX"].history(period="5d")
            
            if len(sp_hist) >= 2:
                sp500_mom = (sp_hist['Close'].iloc[-1] - sp_hist['Close'].iloc[-2]) / sp_hist['Close'].iloc[-2]
            if len(dx_hist) >= 2:
                dxy_mom = (dx_hist['Close'].iloc[-1] - dx_hist['Close'].iloc[-2]) / dx_hist['Close'].iloc[-2]
            if len(vx_hist) >= 2:
                vix_mom = (vx_hist['Close'].iloc[-1] - vx_hist['Close'].iloc[-2]) / vx_hist['Close'].iloc[-2]
            if len(gl_hist) >= 2:
                gold_mom = (gl_hist['Close'].iloc[-1] - gl_hist['Close'].iloc[-2]) / gl_hist['Close'].iloc[-2]
            if len(tn_hist) >= 2:
                tnx_mom = (tn_hist['Close'].iloc[-1] - tn_hist['Close'].iloc[-2]) / tn_hist['Close'].iloc[-2]
            if len(nx_hist) >= 2:
                ndx_mom = (nx_hist['Close'].iloc[-1] - nx_hist['Close'].iloc[-2]) / nx_hist['Close'].iloc[-2]
        except Exception as e:
            logger.warning(f"yfinance fetch failed: {e}")
            
        # --- Fetch Crypto Fear & Greed Index ---
        fg_val = 50.0
        try:
            req = urllib.request.Request("https://api.alternative.me/fng/", headers={'User-Agent': self.user_agent})
            with urllib.request.urlopen(req, timeout=5) as response:
                fg_data = json.loads(response.read().decode())
                fg_val = float(fg_data['data'][0]['value'])
        except Exception as e:
            logger.warning(f"Fear&Greed fetch failed: {e}")
            
        fg_normalized = fg_val / 100.0
        
        # --- Fetch DefiLlama TVL Momentum ---
        # BUG FIX: previously the RAW Ethereum TVL in dollars (~$49,500,000,000) was published as
        # "momentum". That single giant value dominated LayerNorm(200) in the model and made every
        # coin produce identical predictions. Now we publish the % change vs the previous fetch.
        tvl_mom = 0.0
        try:
            req_tvl = urllib.request.Request("https://api.llama.fi/v2/chains", headers={'User-Agent': self.user_agent})
            with urllib.request.urlopen(req_tvl, timeout=5) as response:
                tvl_data = json.loads(response.read().decode())
                for chain in tvl_data:
                    if chain.get('name') == 'Ethereum':
                        tvl_now = float(chain.get('tvl', 0.0))
                        prev_tvl = getattr(self, '_prev_eth_tvl', 0.0)
                        if prev_tvl > 0 and tvl_now > 0:
                            tvl_mom = max(-1.0, min(1.0, (tvl_now - prev_tvl) / prev_tvl))
                        if tvl_now > 0:
                            self._prev_eth_tvl = tvl_now
                        break
        except Exception:
            pass
            
        return MacroState(
            timestamp=time.time(),
            sentiment_score=sentiment_score,
            volatility_expectation=volatility_expectation,
            regime=regime,
            headlines=unique_headlines,
            sp500_momentum=float(sp500_mom),
            dxy_momentum=float(dxy_mom),
            vix_momentum=float(vix_mom),
            gold_momentum=float(gold_mom),
            treasury_yield_momentum=float(tnx_mom),
            ndx_momentum=float(ndx_mom),
            defi_tvl_momentum=float(tvl_mom),
            fear_greed_index=float(fg_normalized)
        )

    async def run(self):
        logger.info("Starting News/Macro Collector...")
        self.running = True
        await self.redis.connect()
        
        while self.running:
            try:
                macro_state = await self.fetch_and_analyze()
                
                # Prepare payload with explicit news_count for consumers
                base_dict = macro_state.model_dump() if hasattr(macro_state, "model_dump") else json.loads(macro_state.json())
                base_dict["news_count"] = len(macro_state.headlines)
                payload_json = json.dumps(base_dict)
                
                # 1. Broadcast Canonical MacroState for trading-bot and real-time listeners
                if self.redis.redis:
                    await self.redis.redis.publish("macro:state:global", payload_json)
                    # 2. Persist in Redis key for dashboard_api and cold-starting services
                    await self.redis.redis.set("dashboard:macro_state", payload_json)
                    logger.info(f"Published & Cached MacroState: Sentiment {macro_state.sentiment_score:.2f}, Regime {macro_state.regime}, News: {len(macro_state.headlines)}")
                else:
                    logger.warning("Redis not connected, skipping publish/cache.")
                
            except Exception as e:
                logger.error(f"Error in NewsMacroCollector loop: {e}", exc_info=True)
                
            await asyncio.sleep(300) # Poll every 5 minutes
            
if __name__ == "__main__":
    collector = NewsMacroCollector()
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        collector.running = False

