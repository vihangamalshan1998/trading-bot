# AI Trading Platform - Future Upgrade Plan

This document outlines the roadmap for upgrading the quantitative system to advanced, institutional-grade complexity.

## ✅ 1. Add "Time-Series Memory" (LSTMs) - COMPLETED
The neural network now actively tracks 10-minute sliding window momentum sequences using a powerful PyTorch LSTM core, allowing it to detect trends instead of just single 5-second snapshots.

## ✅ 2. Expand to Massive Coin Universe - COMPLETED
The bot has been officially expanded and handles **570+ symbols** concurrently, actively trading the entire Binance Futures market to hunt for mathematical setups globally.

## ✅ 3. Introduce a "Maker/Taker" Fee Optimizer - FOUNDATION COMPLETED
The AI Model now outputs a `price_offset` decision specifically designed for Limit Orders. 
**Missing:** We still need to update the `BinanceFuturesAdapter` to physically post the Limit Orders to the Binance API instead of falling back to Market Orders. This will require some architectural work to handle order tracking and cancellations if the Limit isn't filled.

## ⏳ 4. Add "Alternative Data" (Alt-Data) Sensors - MISSING
Right now, we use Gemini for News. We could make it more advanced by plugging in additional data streams:
- **X (Twitter) Firehose APIs:** To detect when major influencers (like Elon Musk) tweet about specific coins (e.g., DOGE) for instant momentum trading.
- **On-Chain Data APIs:** To track massive whale movements. For example, detecting when 10,000 BTC moves out of a cold wallet onto an exchange, allowing the AI to predict and short a massive dump before it happens.
