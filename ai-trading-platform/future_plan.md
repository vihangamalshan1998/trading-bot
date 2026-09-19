# AI Trading Platform - Future Upgrade Plan

This document outlines the roadmap for upgrading the current Phase 1 quantitative system to advanced, institutional-grade complexity once the core model has mastered Testnet trading.

## 1. Add "Time-Series Memory" (LSTMs)
Right now, the AI only looks at the current 5-second snapshot. The next major upgrade would be adding an **LSTM (Long Short-Term Memory)** layer to the neural network. This allows the AI to "remember" the sequence of the last 100 snapshots, helping it detect slow-building momentum and longer-term market trends, rather than just rapid 5-second scalps.

## 2. Expand to 50+ Coins
Because your AI is "coin-agnostic," we can easily expand the `settings.py` file to trade 50 or 100 different altcoins. By casting a wider net, the AI has a much higher chance of finding the perfect mathematical setup on at least one coin at any given minute of the day.

## 3. Add "Alternative Data" (Alt-Data) Sensors
Right now, we use Gemini for News. We could make it more advanced by plugging in additional data streams:
- **X (Twitter) Firehose APIs:** To detect when major influencers (like Elon Musk) tweet about specific coins (e.g., DOGE) for instant momentum trading.
- **On-Chain Data APIs:** To track massive whale movements. For example, detecting when 10,000 BTC moves out of a cold wallet onto an exchange, allowing the AI to predict and short a massive dump before it happens.

## 4. Introduce a "Maker/Taker" Fee Optimizer
Right now, the bot places "Market Orders" (Taker fees), which are expensive and eat into profits. A major upgrade is teaching the AI to place "Limit Orders" (Maker fees) exactly 1 tick away from the mid-price so that it gets paid rebates by Binance instead of paying fees. This dramatically improves the Sharpe ratio and long-term equity growth.
