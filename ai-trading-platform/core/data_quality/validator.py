import math
import time
from typing import Dict, Any, Tuple
from core.logging.logger import logger
from core.config.settings import settings

class DataQualityValidator:
    """
    Phase 4: Data Quality and Time Alignment layer.
    Ensures that bad, stale, or leaked data never enters the AI state.
    """
    
    @staticmethod
    def validate_tick(data: Dict[str, Any], event_timestamp_ms: int, current_time_ms: int = None) -> Tuple[bool, str]:
        """
        Validates raw tick data (Orderbook or AggTrade).
        Returns (is_valid, reason).
        """
        if current_time_ms is None:
            current_time_ms = int(time.time() * 1000)
            
        # 1. Time Alignment & Leakage check
        if event_timestamp_ms > current_time_ms:
            return False, "FUTURE_LEAKAGE"
            
        # Stale data check (e.g., > max_market_data_age_seconds)
        age_seconds = (current_time_ms - event_timestamp_ms) / 1000.0
        if age_seconds > settings.max_market_data_age_seconds:
            return False, "STALE_DATA"
            
        # 2. Check for NaN / Inf
        for key, value in data.items():
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    return False, f"INVALID_NUMBER_{key}"
                    
        # 3. Microstructure logic validation
        if "best_bid" in data and "best_ask" in data:
            if data["best_bid"] <= 0 or data["best_ask"] <= 0:
                return False, "NEGATIVE_PRICE"
            if data["best_bid"] > data["best_ask"]:
                return False, "CROSSED_BOOK"
                
        if "volume" in data and data["volume"] < 0:
            return False, "NEGATIVE_VOLUME"
            
        return True, "OK"
