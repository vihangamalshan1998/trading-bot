# Hostinger VPS A-Z Deployment Guide

This guide will walk you through deploying the AI Trading Platform on a Hostinger Ubuntu VPS. By the end of this guide, the bot will run 24/7 in the background, and you will be able to view the premium live dashboard by visiting your VPS IP address in any web browser.

---

## Step 1: Initial VPS Setup
1. Log in to your Hostinger account, go to **VPS**, and ensure you have an **Ubuntu 22.04 (or newer)** operating system installed.
2. Open your local terminal (PowerShell or Mac Terminal) and SSH into your server:
   ```bash
   ssh root@<YOUR_VPS_IP_ADDRESS>
   ```
   > **Troubleshooting SSH Errors:** If you get a terrifying "REMOTE HOST IDENTIFICATION HAS CHANGED" error, it just means the VPS IP was recycled. Open a local PowerShell window and run `ssh-keygen -R <YOUR_VPS_IP_ADDRESS>` to fix it, then try logging in again.

3. Once logged in, update the server's package list:
   ```bash
   apt update && apt upgrade -y
   ```

## Step 2: Install Prerequisites
You need Python, Redis, MySQL, Node.js, and Nginx. Run the following commands:

> **Note:** While running these `apt install` commands, you might see red errors like `Failed to connect to system scope bus` or `Transport endpoint is not connected`. **These are completely harmless** on a VPS container. Just wait for the progress bar to reach 100%.

```bash
# Install Python (3.10+) and Virtual Environment tools
apt install -y python3 python3-venv python3-pip

# Install Redis (The messaging backbone)
apt install -y redis-server
systemctl enable redis-server
systemctl start redis-server

# Install MySQL Server (The database)
apt install -y mysql-server
systemctl enable mysql
systemctl start mysql

# Install Node.js (For the React frontend)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Install Nginx (To serve the dashboard to the internet)
apt install -y nginx

# Install PM2 globally (To keep the bots running forever)
npm install -g pm2
```

## Step 3: Configure MySQL Database
Log into the MySQL console:
```bash
mysql
```
Inside the MySQL console, copy and paste these commands to create the database:
```sql
CREATE DATABASE ai_trading;
CREATE USER 'trading_user'@'localhost' IDENTIFIED BY 'trading_password';
GRANT ALL PRIVILEGES ON ai_trading.* TO 'trading_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

## Step 4: Get the Code and Setup Python
You need to transfer your code to the VPS. You have two options:

### Option A: Transfer directly from your Windows PC (Easiest)
If you haven't uploaded your code to GitHub, simply copy it directly using SCP.
1. Open a **NEW** PowerShell window on your local Windows PC (do not run this in the VPS terminal) and run:
   ```powershell
   # Replace the IP address with your actual VPS IP
   scp -r d:\laragon\www\Traning-bot\ai-trading-platform root@<YOUR_VPS_IP_ADDRESS>:/var/www/
   ```
2. Once it finishes copying, go back to your **VPS terminal** and type:
   ```bash
   cd /var/www/trading-bot/ai-trading-platform
   ```

### Option B: Clone from GitHub (If already uploaded)
If you pushed your code to a private GitHub repository:
1. Generate a Personal Access Token (PAT) from GitHub Developer Settings.
2. Run this in your VPS terminal (replace with your actual username/repo):
   ```bash
   cd /var/www/
   git clone https://github.com/YOUR-USERNAME/ai-trading-platform.git
   cd ai-trading-platform
   ```
   *When prompted for a password, paste your Personal Access Token (PAT), not your GitHub account password!*

2. Setup the Python Virtual Environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Setup your Environment Variables:
   ```bash
   cp .env.template .env
   nano .env
   ```
   *Paste in your Binance API Keys and Gemini Key, then press `Ctrl+X`, `Y`, and `Enter` to save.*

4. Initialize the Database tables:
   ```bash
   alembic upgrade head
   ```

## Step 5: Run the Bots Continuously (PM2)
We use `pm2` so the bots never stop running, even if the server reboots or you close your terminal.

```bash
# Start the AI Training Bot
pm2 start venv/bin/python --name "ai-trainer" -- -m apps.trainer.ppo

# Start the Market Data Collector
pm2 start venv/bin/python --name "market-collector" -- -m apps.market_collector.main

# Start the Dashboard Backend API
pm2 start venv/bin/uvicorn --name "dashboard-api" -- apps.dashboard_api.main:app --host 127.0.0.1 --port 8000

# Start the Live Trading Execution Bot
pm2 start venv/bin/python --name "trading-bot" -- -m apps.trading_bot.main

# Save the PM2 configuration so it restarts on server reboot
pm2 save
pm2 startup
```
*You can view live logs anytime by typing: `pm2 logs`*

## Step 6: Build the React Dashboard
We need to compile the React code into static HTML/CSS files.
```bash
cd /var/www/trading-bot/ai-trading-platform/apps/dashboard_frontend
npm install
npm run build
```
*This creates a `dist` folder containing the compiled website.*

## Step 7: Configure Nginx
We will configure Nginx to serve the compiled React app on port 80, and reverse proxy the `/api/` requests to our Python backend running on port 8000.

1. Open the default Nginx configuration:
   ```bash
   nano /etc/nginx/sites-available/default
   ```
2. **Delete everything in the file**, and paste this exact configuration:
   ```nginx
   server {
       listen 80 default_server;
       listen [::]:80 default_server;

       root /var/www/trading-bot/ai-trading-platform/apps/dashboard_frontend/dist;
       index index.html index.htm index.nginx-debian.html;

       server_name _;

       # Route API requests to Python Uvicorn backend
       location /api/ {
           proxy_pass http://127.0.0.1:8000/api/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }

       # Serve React Frontend
       location / {
           try_files $uri $uri/ /index.html;
       }
   }
   ```
3. Save and close (`Ctrl+X`, `Y`, `Enter`).
4. Restart Nginx to apply changes:
   ```bash
   systemctl restart nginx
   ```

---

## 🎉 You're Done!
Open your web browser on your personal computer and type in your VPS IP Address (e.g., `http://192.168.x.x`). 

You will immediately see the premium dashboard displaying live market states, macro sentiment scores, total news analyzed, and the PPO training brain progress in real-time, completely hosted on your Hostinger VPS!
