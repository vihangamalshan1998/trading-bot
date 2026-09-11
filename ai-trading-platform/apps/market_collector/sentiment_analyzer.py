import random

class SentimentAnalyzer:
    """
    Phase 10: Extracts sentiment and volatility expectation from news text.
    In a true production environment, this would call an LLM API (e.g., OpenAI or local Llama).
    Here, it simulates the extraction process.
    """
    def __init__(self):
        # We can simulate parsing keywords
        self.bullish_keywords = ['surge', 'jump', 'record', 'adoption', 'approval', 'growth', 'buy']
        self.bearish_keywords = ['crash', 'ban', 'lawsuit', 'hack', 'drop', 'fear', 'sell', 'inflation']
        self.volatile_keywords = ['unexpected', 'shock', 'decision', 'breaking', 'emergency', 'war']

    def analyze(self, headline: str) -> dict:
        """
        Takes a raw headline and returns a dict with sentiment_score and volatility_expectation.
        """
        headline_lower = headline.lower()
        
        bull_score = sum(1 for w in self.bullish_keywords if w in headline_lower)
        bear_score = sum(1 for w in self.bearish_keywords if w in headline_lower)
        vol_score = sum(1 for w in self.volatile_keywords if w in headline_lower)
        
        # Calculate sentiment [-1.0 to 1.0]
        total_sentiment_words = bull_score + bear_score
        if total_sentiment_words == 0:
            sentiment = 0.0
        else:
            sentiment = (bull_score - bear_score) / total_sentiment_words
            
        # Add a tiny bit of random noise for LLM hallucination/uncertainty simulation
        sentiment += random.uniform(-0.1, 0.1)
        sentiment = max(-1.0, min(1.0, sentiment))
        
        # Calculate Volatility [0.0 to 1.0]
        base_vol = 0.2
        volatility = min(1.0, base_vol + (vol_score * 0.3) + (abs(sentiment) * 0.2))
        
        return {
            "sentiment_score": round(sentiment, 4),
            "volatility_expectation": round(volatility, 4)
        }
