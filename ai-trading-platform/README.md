# AI Adaptive Binance Futures Trading Platform

A research-grade AI algorithmic trading platform for Binance Futures, built with asynchronous Python, Redis, MySQL, and PyTorch. 

This platform is designed to collect high-frequency market data (Level 2 Order Book, Aggregate Trades), compute live micro-structural features, train Reinforcement Learning agents, and execute live trades on the Binance Testnet through an enforced Risk Management engine.

## Architecture Overview

The system is decoupled into discrete microservices connected via a Redis state-bus:

1. **Market Collector Service (`apps/market_collector`)**
   - Connects to the Binance Futures WebSocket streams (`depth@100ms`, `aggTrade`).
   - Maintains an in-memory Limit Order Book (LOB) synchronized via REST snapshots and WebSocket deltas.
   - Pushes live updates to the Feature Engine.
   
2. **Feature Engine (`apps/feature_engine`)**
   - Computes advanced mathematical market features in real-time using Pandas/Numpy (e.g., Spread BPS, Order Book Imbalance, rolling VWAP).
   - Broadcasts the finalized feature state to a Redis in-memory cache at ultra-low latency.
   - Flushes batched historical snapshots to MySQL for offline ML training.

3. **Trading Bot Service (`apps/trading_bot`)**
   - Subscribes to the live Redis state.
   - Runs the AI trading strategy (PyTorch neural network inference) to decide on actions (Buy/Sell/Hold).
   - Routes intent through the Risk Manager before executing on the Binance API.

4. **Risk Manager (`core/risk`)**
   - A strict safety envelope. Evaluates all orders before exchange submission.
   - Enforces Maximum Position Size, Daily Drawdown Limits, and prevents Fat-Finger errors.

5. **ML Research Environment (`apps/research`)**
   - An OpenAI Gymnasium-compatible environment (`SpotTradingEnv`) that simulates the market mechanics, fees, and PnL.
   - Contains PyTorch neural network architectures (`TradingNet`) and Reinforcement Learning training loops (REINFORCE algorithm) to train the AI offline using the MySQL dataset.

## Prerequisites

- **Python 3.10+**
- **MySQL 8.0+** (Running on port 3306)
- **Redis Server** (Running on port 6379 - **CRITICAL:** The dashboard and live event bus will not work without Redis)
- **Node.js & npm** (Required to run the visual React dashboard)
- **Binance Testnet API Keys**

## Setup & Installation

**1. Clone & Environment Setup**
```powershell
# Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**2. Configuration**
Update the `.env` file in the root directory (or create one based on `core/config/settings.py` defaults) to include your database credentials and Binance API keys.
```env
# Database
MYSQL_HOST=localhost
MYSQL_USER=trading_user
MYSQL_PASSWORD=trading_password
MYSQL_DATABASE=ai_trading

# Binance Testnet
BINANCE_API_KEY=your_testnet_api_key
BINANCE_API_SECRET=your_testnet_api_secret
BINANCE_TESTNET=True
TRADING_SYMBOL=BTCUSDT
```

**3. Database Initialization**
Run the Alembic migrations to construct the MySQL tables for the Feature Repository.
```powershell
alembic upgrade head
```

## Running the Platform

To orchestrate and launch both the Market Collector and the live Trading Bot simultaneously, run the included PowerShell script:

```powershell
.\start_system.ps1
```

*(This will spawn two separate background windows running the individual services).*

## Visual Dashboard

The platform includes a real-time React dashboard to visualize live trading states, active positions, and AI training metrics (loss).

1. **Start the Backend API:**
   ```powershell
   # Ensure your virtual environment is active
   uvicorn apps.dashboard_api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Start the Frontend Dashboard:**
   ```powershell
   cd apps/dashboard_frontend
   npm install  # (First time only)
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser.

> **Note:** The dashboard requires Redis to stream live metrics. If Redis is down, the dashboard will not receive updates.

## Machine Learning Pipeline

To train the reinforcement learning AI agent:

1. **Train Model**: Runs the RL algorithm against the PyTorch `TradingNet` using historical (or generated) data, saving the best `.pth` checkpoint to `models/best_model.pth`.
   ```powershell
   python apps/research/train.py
   ```
2. **Evaluate Model**: Runs the trained checkpoint against a holdout validation set and logs quantitative performance metrics (Cumulative PnL, Max Drawdown).
   ```powershell
   python apps/research/evaluate.py
   ```

## Disclaimer
This project is built for **research and engineering purposes only**. It does not guarantee profitability. The objective of this codebase is to discover whether a learned RL system can develop a robust trading edge after realistic fees, slippage, latency, and unseen-data testing. Do not use this in a live-money production environment without extensive backtesting and risk evaluation.
