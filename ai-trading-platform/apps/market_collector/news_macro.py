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

HEADLINE_CACHE_KEY = "news:headline_scores"
HEADLINE_CACHE_TTL = 48 * 3600


def parse_headline_scores(raw_text: str, titles: list) -> dict:
    """Parses Gemini's per-headline JSON array -> {title: (sentiment, volatility)}."""
    clean = raw_text.strip().replace("```json", "").replace("```", "").strip()
    rows = json.loads(clean[clean.index("["): clean.rindex("]") + 1])
    scores = {}
    for row in rows:
        i = int(row.get("i", -1))
        if 0 <= i < len(titles):
            s = max(-1.0, min(1.0, float(row.get("s", 0.0))))
            v = max(0.0, min(1.0, float(row.get("v", 0.5))))
            scores[titles[i]] = (s, v)
    return scores


def aggregate_headline_scores(cache: dict, titles: list, n_new: int = 0) -> tuple:
    """Average of the cached per-headline scores for the CURRENT headlines -> (sentiment, volatility, source)."""
    rows = [cache[t] for t in titles if t in cache]
    if not rows:
        return 0.0, 0.5, "none"
    sentiment = sum(r["s"] for r in rows) / len(rows)
    volatility = sum(r["v"] for r in rows) / len(rows)
    n_gemini = sum(1 for r in rows if r.get("src") == "gemini")
    source = f"{n_gemini}/{len(rows)} Gemini, {n_new} newly scored"
    return round(max(-1.0, min(1.0, sentiment)), 4), round(max(0.0, min(1.0, volatility)), 4), source


class NewsMacroCollector:
    """
    Phase 7: Real External Information Pipeline.
    Fetches real RSS feeds (CoinDesk, CoinTelegraph, Decrypt) and broadcasts canonical MacroState.
    Each headline is scored ONCE (Gemini SDK -> Gemini REST -> lexicon fallback) and cached; sentiment = average.
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
        self.headline_scores = {}  # title -> {"s": sentiment, "v": volatility, "src": "gemini"|"lexicon", "ts": epoch}
        self._cache_loaded = False

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

    def _gemini_generate(self, prompt: str) -> str:
        """Returns Gemini's raw text answer. Tier 1: SDK, Tier 2: direct REST (no SDK dependency)."""
        if not settings.gemini_api_key:
            raise ValueError("No GEMINI_API_KEY configured")
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=settings.gemini_model_version,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            return response.text
        except Exception as sdk_err:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model_version}:generateContent?key={settings.gemini_api_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.0}}
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError(f"Invalid Gemini response (SDK error: {sdk_err})")
            return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    def _gemini_score_headlines(self, titles: list) -> dict:
        """
        Scores each NEW headline individually in a single Gemini call.
        Returns {title: (sentiment [-1, 1], volatility [0, 1])}.
        """
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
        prompt = (
            "You are a quantitative crypto analyst. Score EACH headline below on its own.\n"
            "Return ONLY a raw JSON array (no markdown), one object per headline, in the same order: "
            '[{"i": <number>, "s": <sentiment float -1.0 very bearish .. 1.0 very bullish for crypto prices>, '
            '"v": <expected volatility float 0.0 calm .. 1.0 shock>}]. '
            "Use 0.0 sentiment for headlines that are neutral or not market-moving.\n\n"
            f"Headlines:\n{numbered}"
        )
        return parse_headline_scores(self._gemini_generate(prompt), titles)

    async def _load_score_cache(self):
        if self._cache_loaded or not self.redis.redis:
            return
        self._cache_loaded = True
        try:
            raw = await self.redis.redis.get(HEADLINE_CACHE_KEY)
            if raw:
                self.headline_scores = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                logger.info(f"Loaded {len(self.headline_scores)} cached headline scores.")
        except Exception as e:
            logger.warning(f"Could not load headline score cache: {e}")

    async def _save_score_cache(self):
        cutoff = time.time() - HEADLINE_CACHE_TTL
        self.headline_scores = {t: v for t, v in self.headline_scores.items() if v.get("ts", 0) >= cutoff}
        if self.redis.redis:
            try:
                await self.redis.redis.set(HEADLINE_CACHE_KEY, json.dumps(self.headline_scores))
            except Exception as e:
                logger.warning(f"Could not save headline score cache: {e}")

    async def score_headlines(self, titles: list) -> tuple:
        """
        Stable sentiment: every headline is scored ONCE and the score is remembered (Redis, 48h).
        The published sentiment is the average over the current headlines, so it only moves when
        the news actually changes. (Previously all headlines were re-sent to Gemini every 5 minutes
        and the same 15 headlines flipped between -0.15 and +0.15 -> pure noise for the model.)
        Headlines scored by the lexicon fallback are re-tried with Gemini on the next cycle.
        Returns (sentiment_score, volatility_expectation, source_summary).
        """
        await self._load_score_cache()
        now = time.time()
        to_score = [t for t in titles if t not in self.headline_scores
                    or (self.headline_scores[t].get("src") == "lexicon" and settings.gemini_api_key)]
        if to_score:
            scored = {}
            if settings.gemini_api_key:
                try:
                    scored = await asyncio.to_thread(self._gemini_score_headlines, to_score)
                    for t, (s, v) in scored.items():
                        self.headline_scores[t] = {"s": s, "v": v, "src": "gemini", "ts": now}
                except Exception as e:
                    logger.warning(f"Gemini headline scoring unavailable ({e}). Using lexicon for new headlines.")
            for t in to_score:
                if t not in scored and t not in self.headline_scores:
                    s, v = self._lexicon_nlp_sentiment(t)
                    self.headline_scores[t] = {"s": s, "v": v, "src": "lexicon", "ts": now}
            await self._save_score_cache()
        return aggregate_headline_scores(self.headline_scores, titles, len(to_score))

    async def fetch_and_analyze(self) -> MacroState:
        """
        Fetches RSS feeds and computes stable per-headline sentiment.
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

        logger.info(f"📰 Fetched {len(unique_headlines)} Unique Headlines for Sentiment Analysis")

        sentiment_score, volatility_expectation, sentiment_source = await self.score_headlines(
            [item["title"] for item in unique_headlines])

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

