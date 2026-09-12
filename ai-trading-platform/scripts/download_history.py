import aiohttp
import asyncio
import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

# Binance Spot API for historical K-lines is easier/public without auth
BINANCE_API = "https://api.binance.com/api/v3/klines"

async def download_historical_klines(symbol: str, interval: str, start_time: int, end_time: int) -> List[list]:
    """Downloads historical K-line data from Binance."""
    url = f"{BINANCE_API}?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit=1000"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Error fetching data for {symbol}: {response.status}")
                return []

async def fetch_month_data(symbol: str, year: int, month: int, interval="1m") -> List[list]:
    """Fetches a full month of 1m data."""
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
        
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    all_klines = []
    current_start = start_ts
    
    while current_start < end_ts:
        # Binance limits to 1000 bars per request
        current_end = min(current_start + (1000 * 60 * 1000), end_ts) 
        klines = await download_historical_klines(symbol, interval, current_start, current_end)
        
        if not klines:
            break
            
        all_klines.extend(klines)
        current_start = klines[-1][0] + 1 # Next ms
        
        await asyncio.sleep(0.1) # Rate limit protection
        
    return all_klines

def save_split_datasets(symbol: str, klines: List[list], base_dir: str = "data/historical"):
    """
    Phase 5: Strict Time-Based Data Splits (Chronological)
    Never randomly shuffle temporal data.
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # K-line format: 
    # [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base, Taker buy quote, Ignore]
    
    total_len = len(klines)
    if total_len == 0:
        return
        
    # Temporal Split: 70% Train, 10% Val, 10% Test, 10% Forward-Walk
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.8)
    test_end = int(total_len * 0.9)
    
    splits = {
        "train": klines[:train_end],
        "validation": klines[train_end:val_end],
        "test": klines[val_end:test_end],
        "forward": klines[test_end:]
    }
    
    for split_name, data in splits.items():
        file_path = os.path.join(base_dir, f"{symbol}_{split_name}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print(f"Saved {len(data)} {split_name} bars for {symbol} to {file_path}")

async def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for sym in symbols:
        print(f"Downloading historical data for {sym}...")
        # Download e.g. January 2024
        klines = await fetch_month_data(sym, 2024, 1)
        if klines:
            save_split_datasets(sym, klines)
            
if __name__ == "__main__":
    asyncio.run(main())
