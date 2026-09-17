# AI Trading Platform: Deployment & Operations Guide

This guide provides step-by-step instructions for setting up the AI Trading Platform from scratch on either a local PC (Windows/Mac/Linux) or a remote VPS server (e.g., AWS EC2, DigitalOcean, Ubuntu).

## 1. System Prerequisites
Before you begin, ensure your system has the following installed:
* **Python 3.10+** (The brain of the system)
* **MySQL Server** (Stores the historical experiences and trading data)
* **Redis** (Used for lightning-fast real-time messaging between microservices)
* **Git** (To clone the repository)

## 2. Server/PC Setup

> **Tip:** If you are setting this up on a VPS (like Ubuntu), it is highly recommended to use `tmux` or `screen` so the bot continues running even if you close your SSH terminal connection.

### A. Clone the Repository
```bash
git clone https://github.com/your-repo/ai-trading-platform.git
cd ai-trading-platform
```

### B. Setup Python Virtual Environment
Keep your dependencies isolated from the rest of your system.
```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Mac/Linux/VPS)
source venv/bin/activate
```

### C. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Database & Service Configuration

### A. Start Redis
Make sure Redis is running in the background. 
- On Windows: Use Memurai or WSL to run Redis.
- On Linux/VPS: `sudo systemctl start redis-server`

### B. Configure MySQL
Log into your MySQL server and create the database:
```sql
CREATE DATABASE ai_trading;
CREATE USER 'trading_user'@'localhost' IDENTIFIED BY 'trading_password';
GRANT ALL PRIVILEGES ON ai_trading.* TO 'trading_user'@'localhost';
FLUSH PRIVILEGES;
```

### C. Configure Environment Variables
Copy the template file to create your active `.env` file:
```bash
cp .env.template .env
```
Open `.env` and configure your keys:
1. **Binance Keys:** Add your `BINANCE_API_KEY` and `BINANCE_API_SECRET`.
2. **Gemini Key:** Add your `GEMINI_API_KEY` for AI news sentiment analysis.
3. **Database:** Ensure the MySQL and Redis credentials match your setup.
4. **Safety Gates:** Keep `BINANCE_TESTNET=True` and `ALLOW_LIVE_TRADING=False` until you are absolutely ready for real money.

---

## 4. Operational Workflow

The system is designed to be run in phases. Follow this exact order:

### Phase 1: Download Historical Data (The "Textbooks")
You only need to do this once (or whenever you want to update the AI's historical memory).
```bash
python -m apps.market_collector.historical_downloader
```
*This downloads the last 7 days of 1-minute candles and processes them into AI-readable features in your database. Wait for it to finish (approx. 2-5 minutes).*

### Phase 2: Train the AI Brain (The "School")
Run the reinforcement learning trainer so the bot can practice trading on the data you just downloaded.
```bash
python -m apps.trainer.ppo
```
*Let this run for a few hours. The AI will constantly practice, learn from its mistakes, and save improved versions of its brain (`latest_model.pt`). You can stop it manually with `Ctrl+C` when you are satisfied with its reward metrics.*

### Phase 3: Start Live Trading (The "Real World")
Once the AI has been trained and a model is saved, you can boot the entire live trading ecosystem.
```bash
python run_system.py
```
This orchestrator script will automatically launch:
1. **Market Data Stream:** Connects to Binance WebSocket.
2. **News Sentiment Engine:** Polls headlines and asks Gemini for macro sentiment.
3. **Trading Bot:** Loads the trained brain and makes real-time sub-second trading decisions.

> **Warning:** By default, the bot trades on the **Binance Testnet** with fake money. To switch to real money, you must open your `.env` file and set `BINANCE_TESTNET=False`, `ALLOW_TESTNET=False`, and `ALLOW_LIVE_TRADING=True`. Do this **only** when you are 100% confident in the bot's training.

### Phase 4: Launch the Visual Dashboard (Optional but Recommended)
To monitor live trades, equity, and AI training loss visually:

1. **Start the API Backend:**
   ```bash
   uvicorn apps.dashboard_api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Start the React Frontend:**
   ```bash
   cd apps/dashboard_frontend
   npm install
   npm run dev
   ```
   *Open the generated local URL (usually http://localhost:5173) in your web browser.*

> **Critical Note:** The dashboard absolutely requires **Redis** to be running, as it streams all real-time events over Redis Pub/Sub.

---

## 5. VPS Specifics (Running 24/7)
If running on a VPS, you don't want the bot to die when you close your laptop. Use `tmux`:

1. Start a new background session: `tmux new -s trading_bot`
2. Activate your environment and start the bot: `python run_system.py`
3. Detach from the session (leave it running in the background): Press `Ctrl+b` then press `d`.
4. (Optional) To check on the bot later, SSH back into your server and run: `tmux attach -t trading_bot`
