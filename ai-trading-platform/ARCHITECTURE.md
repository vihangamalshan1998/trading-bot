# System Architecture

The AI Trading Platform is structured into discrete micro-apps that run concurrently. This design ensures that high-latency operations (like executing REST API calls or running ML inference) do not block the ultra-low latency operations (like processing WebSocket messages).

## Directory Structure

```text
ai-trading-platform/
├── apps/
│   ├── market_collector/    # Ingests live data from Binance
│   ├── feature_engine/      # Transforms raw data into ML features
│   ├── orderbook/           # Builds local Limit Order Book
│   ├── trading_bot/         # Consumes features & executes trades
│   └── research/            # Offline RL Gym environment & PyTorch Models
├── core/
│   ├── config/              # Pydantic Settings & Env vars
│   ├── db/                  # Async Redis & MySQL connections
│   ├── logging/             # Centralized structural logging
│   ├── exchange/            # Base exchange abstractions
│   └── risk/                # Risk Manager limits and safety checks
├── data/
│   └── migrations/          # Alembic MySQL schemas
└── start_system.ps1         # System orchestration script
```

## Component Breakdown

### 1. Market Collector (`apps/market_collector`)
The backbone of the system. It connects to the `wss://stream.testnet.binance.vision` endpoint. 
It utilizes Python's `asyncio` to simultaneously process `depthUpdate` events (which route into the `OrderBookBuilder`) and `aggTrade` events (which route into the `FeatureEngine`).

### 2. Feature Engine (`apps/feature_engine`)
Takes the raw Level 2 Order Book state and trades, and derives predictive features:
- **Spread BPS**: Distance between Best Ask and Best Bid.
- **Mid Price**: Average of Best Bid/Ask.
- **Micro Price**: Volume-weighted mid price.
- **Imbalance**: Ratio of Bid volume to total Bid/Ask volume at the top of the book.
- **VWAP**: Rolling Volume Weighted Average Price based on recent trades.

### 3. Redis State Bus (`core/db/redis.py`)
To prevent the Trading Bot from needing its own redundant WebSocket connections (which can trigger rate limits), the `MarketCollector` serializes the output of the `FeatureEngine` into a JSON dictionary and pushes it to Redis (`market:state:BTCUSDT`) every second. 

### 4. Experience MySQL Storage (`core/db/repository.py`)
Because line-by-line SQL inserts would crash the async event loop at high frequencies, the `MarketFeatureRepository` buffers the features into memory. Once `batch_size` (e.g. 50) is hit, it executes a high-speed bulk `session.add_all()` to push the history to MySQL for offline training.

### 5. Risk Manager (`core/risk`)
Positioned deliberately as a firewall between the `TradingBot` and the `BinanceSpotAdapter`. It tracks simulated daily PnL and total `max_position_usd` inventory. If the AI hallucinates a massive order, the `RiskManager.approve_order()` will reject it before it hits the network.

### 6. RL Engine (`apps/research`)
A self-contained Sandbox mirroring the live environment.
- **`SpotTradingEnv`**: An OpenAI `gymnasium` environment that calculates MTM portfolio value and standardizes the AI action space to 0, 1, 2.
- **`TradingNet`**: A PyTorch MLP neural network.
- **`train.py`**: A Policy Gradient (REINFORCE) loop that teaches the `TradingNet` to maximize profits and minimize drawdowns, saving the resulting `.pth` checkpoint.
