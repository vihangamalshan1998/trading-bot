# Complete System Documentation: AI Trading Platform (A-Z)

This document provides a comprehensive, end-to-end explanation of every single component, subsystem, and logic flow engineered across the 15-Phase Master Prompt for the AI Trading Platform.

---

## 1. System Overview & Orchestration

The platform is a fully autonomous, deep-reinforcement-learning-driven trading system. It is designed to trade perpetual futures on the Binance Testnet. The architecture is split into three main layers:
1. **Data & Market Collection Layer**: Gathers live websocket data, macroeconomic news, and dynamically screens for the best trading pairs.
2. **Research & Simulation Layer**: Contains the strictly controlled Gym environments, the World Model (neural network), and the offline training/backtesting engines.
3. **Production Execution Layer**: The live asynchronous trading loop, strict risk management firewalls, and continuous online training systems.

At the very top sits `run_system.py`.
*   **Logic**: This is the Master Orchestrator. Instead of running 5 separate python scripts manually, this script uses `asyncio` and `subprocess` to boot up the Universe Screener, News Collector, Market Data WebSocket, Online Trainer, and the Trading Bot all at the same time. It pipes their output into a single, clean terminal view.

---

## 2. Market Collection & Data Layer

### 2.1 The Dynamic Universe Screener (`apps/market_collector/universe.py`)
*   **Purpose**: The cryptocurrency market changes constantly. A static list of symbols (like just BTC and ETH) is suboptimal.
*   **Logic**: Every hour, this script queries the Binance `/fapi/v1/ticker/24hr` REST endpoint. It filters for USDT perpetual contracts, sorts them by 24-hour quoting volume, and selects the absolute Top N (e.g., top 5 or 10) most liquid pairs. This dynamic list is passed to the rest of the system so the AI is always trading the hottest coins.

### 2.2 Live Market Data Collector (`apps/market_collector/market_data.py`)
*   **Purpose**: To feed the AI real-time market data.
*   **Logic**: It opens a persistent `websockets` connection to `wss://fstream.binance.com/ws`. It subscribes to three specific streams for every coin in our dynamic universe:
    1.  `@bookTicker`: Gives the best bid, best ask, and quantities. The logic instantly calculates the `spread`, the `mid_price`, and the orderbook `imbalance` (who has more size, buyers or sellers?).
    2.  `@markPrice`: Tracks the funding rate.
    3.  `@forceOrder`: Tracks liquidations (when other traders blow up their accounts).
    It compresses this data into a massive 200-dimensional V2 state array (108 active slots, 92 padding slots) and pushes it to Redis (`market:state:{symbol}`) at extremely high frequencies. Please refer to `V2_200_SLOT_BLUEPRINT.md` for the full slot mapping.

### 2.3 News & Macro Collector (`apps/market_collector/news_macro.py`)
*   **Purpose**: To give the AI "outside world" context.
*   **Logic**: Currently a synthetic mock service, it generates random macroeconomic shifts (like fake inflation reports or global sentiment scores). It broadcasts these scores over Redis so the neural network can factor global fear/greed into its trading decisions.

### 2.4 Event Memory System (`core/ai/memory.py`)
*   **Purpose**: To prevent the AI from instantly forgetting what it just read in the news.
*   **Logic**: It acts as a chronological database (using a Redis List). Every time a major macro event happens, it is pushed onto the list. The AI can pull the last 10 events to understand the *trajectory* of the economy (e.g., "Sentiment was 0.8 yesterday, but is 0.2 today. That's a crash.").

---

## 3. Exchange & Execution Adapters

### 3.1 Binance Futures Client (`core/exchange/binance_client.py`)
*   **Purpose**: The bridge that actually executes trades on your behalf.
*   **Logic**: It uses `aiohttp` to securely sign requests using your API Key and Secret using HMAC SHA256 cryptography. It connects to the Binance Testnet (`testnet.binancefuture.com`) to place market orders, limit orders, and check your live account balance.

### 3.2 Symbol Registry (`core/exchange/symbol_registry.py`)
*   **Purpose**: Binance is incredibly strict. If you try to buy 1.12345 BTC, but the precision limit is 3 decimals, Binance will reject the order with a `400` error.
*   **Logic**: On startup, it calls Binance's `exchangeInfo` endpoint, downloads the massive rulebook for every coin, and caches the `tickSize` (price precision) and `stepSize` (quantity precision). It provides helper functions (`format_quantity`, `format_price`) that dynamically chop off excess decimals before the bot attempts to place a trade, guaranteeing zero API rejections.

---

## 4. Research & Simulation Layer

### 4.1 Realistic Futures Environment (`apps/research/environment.py`)
*   **Purpose**: A brutal, highly realistic simulator used to train the AI before it touches real money.
*   **Logic**: Built on OpenAI's `Gymnasium` framework. It tracks the bot's fake `wallet_balance`. 
    *   If the bot goes LONG, it calculates margin based on leverage.
    *   Every 8 simulated hours, it deducts the `funding_rate` from the balance.
    *   If the unrealized loss drops below the `maintenance_margin_rate`, it simulates a **Liquidation**, instantly wiping the position and slapping the AI with a massive negative reward penalty. 
    *   It supports trading `N` symbols at the exact same time (cross-margin).

### 4.2 V3 Actor-Critic Brain (`apps/research/model.py`)
*   **Purpose**: The "Brain" of the operation. In V3, we use a **SingleSymbolActorCritic** architecture driven by an **LSTM (Long Short-Term Memory)** network. Instead of just reacting to the current price, this AI retains perfect chronological memory of the trading session.
*   **Logic**:
    1.  **3D Tensor Input**: Takes a massive `[Batch, 300, 200]` tensor representing a 5-minute sliding window (300 steps). Each step contains 200 features.
    2.  **LSTM Layer**: Takes the current state and the previous physical `hidden_state` to maintain a running memory of the chart's history, tracking velocity and momentum. It continuously passes this hidden state forward tick-by-tick.
    3.  **Actor Head**: The trading logic. It looks at the LSTM output and decides whether to BUY, SELL, or HOLD.
    4.  **Multi-Horizon Critic Head**: Guesses the exact future profit of the trade across 3 distinct time horizons (5-minute, 1-hour, 4-hour).

### 4.3 Offline PPO Training Loop (`apps/trainer/ppo.py`)
*   **Purpose**: To teach the Brain how to trade using historical data.
*   **Logic**: A background Teacher's Assistant calculates the *exact* future returns of old trades. The Proximal Policy Optimization (PPO) script bypasses traditional TD-learning math, forcing the Critic to target those exact, ground-truth returns. It uses a **PER SumTree** to sample the database, heavily prioritizing trades where the AI made massive financial mistakes, allowing for extremely fast convergence.

### 4.4 Forward Testing Engine (`apps/research/forward_test.py`)
*   **Purpose**: The final exam for the AI.
*   **Logic**: It takes the fully trained World Model and forces it to trade on completely *unseen*, out-of-sample data. It records every trade and calculates Wall Street institutional metrics: Total Return, Sharpe Ratio (risk-adjusted return), and Max Drawdown (the worst peak-to-trough loss). If the Sharpe Ratio is negative, the bot is not allowed to go live.

---

## 5. Production Execution Layer

### 5.1 Risk Manager (`core/risk/manager.py`)
*   **Purpose**: The ultimate safety net. We do not trust the AI completely.
*   **Logic**: It sits between the AI and Binance. When the AI says "BUY BTC", the Risk Manager intercepts the command. It checks:
    *   Have we lost more than 5% today? (Daily Drawdown Limit)
    *   Is the current macro sentiment apocalyptic? (Macro Sentiment Block)
    If any safety limit is breached, the Risk Manager overrides the AI, changes the action to `HOLD` or `CLOSE`, and prevents the trade from reaching Binance.

### 5.2 Live Replay Buffer (`core/ai/replay_buffer.py`)
*   **Purpose**: To record history as it happens for future offline training.
*   **Logic**: Every time the live Trading Bot acts, it compresses the massive 300x200 state matrix into a **Msgpack** binary string to save RAM/Network bandwidth. It pushes this to MySQL via bulk `add_all` SQLAlchemy inserts to prevent transaction locking.

### 5.3 Continuous Online Trainer / The Teacher's Assistant
*   **Purpose**: To adapt to the market in real-time and grade the AI.
*   **Logic**: The `reward_calculator.py` runs silently in the background. It finds trades that are older than 4 hours, analyzes the database to see what happened *after* the AI bought, and hardcodes the true 5m, 1h, and 4h returns. The AI trainer then uses this perfectly graded data to update its weights.

### 5.4 Production Trading Bot (`apps/trading_bot/main.py`)
*   **Purpose**: The heart that ties everything together and actually trades.
*   **Logic**: 
    1. It connects to the Redis streams fed by the `Market Data Collector`.
    2. It updates a sliding `collections.deque(maxlen=300)` for all 6 active coins.
    3. **Batched Inference:** It stacks all 6 coins into a single `[6, 300, 200]` tensor to save CPU cycles.
    4. **True Memory:** It retrieves the `hidden_states` (LSTM memory) for all 6 coins from the previous tick.
    5. It feeds the batched tensor and the hidden states into the PyTorch `SingleSymbolActorCritic`.
    6. It saves the newly generated `hidden_states` back into RAM for the next tick.
    7. It passes the resulting 6 actions through the `RiskManager`.
    8. If approved, it dynamically formats the price/quantity via `SymbolRegistry` and fires the trades to the exchange.
    9. It compresses the states via **Msgpack** and logs them for training.
    *(This entire loop runs asynchronously thousands of times per minute, perfectly retaining physical memory across time).*

---
**END OF DOCUMENTATION**
