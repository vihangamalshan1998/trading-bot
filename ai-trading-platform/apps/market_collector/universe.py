"""
Consolidated Universe Screener wrapper for backward compatibility.
Points directly to the canonical UniverseScreener implementation in universe_screener.py.
"""
from apps.market_collector.universe_screener import UniverseScreener

__all__ = ["UniverseScreener"]
