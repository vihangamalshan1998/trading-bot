import os
import time
import json
import urllib.request
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class HistoricalFuturesLoader:
    """
    Phase 8: Real Binance USDT-M Perpetual Futures Historical Data Pipeline.
    Replaces synthetic random walks with real historical Binance candles.
    Enforces strict chronological train / validation / test splits to prevent look-ahead bias.
    """
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.base_url = "https://fapi.binance.com/fapi/v1/klines"

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "5m",
        start_time_ms: int = None,
        end_time_ms: int = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetches chronological klines from Binance public USDT-M endpoint with rate-limit respect.
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms

        query_str = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.base_url}?{query_str}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if not data:
            return pd.DataFrame()

        # Schema: [Open time, Open, High, Low, Close, Volume, Close time, Quote volume, Trades, Taker buy base, Taker buy quote, Ignore]
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
        for c in numeric_cols:
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.sort_values("open_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def compute_forward_returns(self, df: pd.DataFrame, fee_pct: float = 0.0007) -> pd.DataFrame:
        """
        Computes leak-free forward returns for 5m (1 step), 1h (12 steps), and 4h (48 steps).
        """
        df = df.copy()
        # 1-step ahead, 12-step ahead, 48-step ahead (assuming 5m candles)
        df["fwd_ret_5m"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100.0
        df["fwd_ret_1h"] = (df["close"].shift(-12) - df["close"]) / df["close"] * 100.0
        df["fwd_ret_4h"] = (df["close"].shift(-48) - df["close"]) / df["close"] * 100.0
        
        # Net of transaction costs (e.g. 0.07% round-trip)
        df["net_fwd_long_5m"] = df["fwd_ret_5m"] - (fee_pct * 100.0)
        df["net_fwd_short_5m"] = -df["fwd_ret_5m"] - (fee_pct * 100.0)
        return df

    def split_chronological(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partitions dataset chronologically into Train, Validation, and untouched Out-of-Sample Test sets.
        Strictly forbids random shuffling to preserve temporal market causality.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, val_df, test_df
