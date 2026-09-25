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

### 2. Feature Engine (`apps/feature_engine`) & State Vector
Takes the raw Level 2 Order Book state, global macro news, and whale trades, and derives predictive features into a **V2 200-Dimension Array (108 Active / 92 Padding)**:
- For a complete, mathematical breakdown of all 200 slots, see: `V2_200_SLOT_BLUEPRINT.md`
- For an easy-to-read explanation of the 200 slots, see: `V2_200_SLOT_TODDLER_EDITION.md`

### 3. Redis State Bus (`core/db/redis.py`)
To prevent the Trading Bot from needing its own redundant WebSocket connections (which can trigger rate limits), the `MarketCollector` serializes the output of the `FeatureEngine` into a JSON dictionary and pushes it to Redis (`market:state:BTCUSDT`) every second. 

### 4. Experience MySQL Storage (`core/db/repository.py`)
Because line-by-line SQL inserts would crash the async event loop at high frequencies, the `ReplayBuffer` utilizes a `collections.deque(maxlen=120)` to maintain a sliding window. Once a trade action occurs, it takes the entire 120x41 2D matrix (a 10-minute snapshot) and saves it as a JSON payload in MySQL for offline training.

### 5. Risk Manager (`core/risk`)
Positioned deliberately as a firewall between the `TradingBot` and the `BinanceSpotAdapter`. It tracks simulated daily PnL and total `max_position_usd` inventory. If the AI hallucinates a massive order, the `RiskManager.approve_order()` will reject it before it hits the network.

### 6. RL Engine (`apps/research` & `apps/trainer`)
A self-contained Sandbox mirroring the live environment.
- **`ReplayBuffer`**: Pulls historical 120-step matrices and dynamically pads any legacy records to ensure consistent sequence lengths.
- **`TradingNet (V2)`**: A PyTorch **LSTM (Long Short-Term Memory)** neural network that processes 3D Tensors of shape `[Batch, 120, 41]`. This allows the AI to learn velocity, momentum, and complex temporal patterns.
- **`ppo.py`**: A Proximal Policy Optimization (PPO) loop that teaches the `TradingNet` to maximize profits by learning from both historical simulations and live continuous trades, saving the resulting `.pth` checkpoint.
