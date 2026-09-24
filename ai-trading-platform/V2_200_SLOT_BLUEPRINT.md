# V2 God Mode: 200-Slot AI Neural Blueprint

This document contains the exact mathematical mapping of the 200 sensory inputs (slots) fed into the AI's Neural Network every single second. 

By feeding the AI this structured data, we eliminate random guessing and give it the ability to execute statistically significant trading strategies based on momentum, geometry, and cross-asset correlation.

---

## 1. Portfolio & Risk Metrics (Slots 0 - 9)
**Purpose:** Teaches the AI to manage your money, avoid margin calls, and understand its own buying power.

| Slot | Feature Name | What it Measures | How it Helps the AI |
| :--- | :--- | :--- | :--- |
| **0** | `wallet_balance` | Total raw cash in the account. | Baseline for absolute risk sizing. |
| **1** | `equity` | Total cash + active profits/losses. | Tells the AI the true live value of the account. |
| **2** | `free_margin` | Cash not currently locked in trades. | Tells the AI exactly how much it can still spend. |
| **3** | `margin_utilization` | % of account currently locked. | **Crucial:** Prevents the AI from over-trading. If this is 90%, it learns to stop opening new trades. |
| **4** | `total_exposure` | Total dollar value of all open bets. | Helps the AI balance its risk across multiple coins. |
| **5-9** | *Reserved Pad* | `0.0` | Reserved for future Value-at-Risk (VaR) math. |

## 2. Active Position Awareness (Slots 10 - 19)
**Purpose:** Teaches the AI how to manage a trade *after* it opens one (taking profits and cutting losses).

| Slot | Feature Name | What it Measures | How it Helps the AI |
| :--- | :--- | :--- | :--- |
| **10** | `quantity` | Size of the current open trade. | Tells the AI if it is flat, long, or short. |
| **11** | `entry_price` | Exact price the trade was opened at. | Baseline for profit taking. |
| **12** | `leverage` | Normalized leverage (e.g. 20x). | Teaches the AI that higher leverage = tighter stops needed. |
| **13** | `current_pnl_pct` | **Live Unrealized Profit/Loss %.** | **The Holy Grail:** The AI literally sees "I am up +5%." It learns to hit the SELL button to take the cash before it drops. |
| **14** | `dist_to_liq` | Distance to liquidation price. | If this gets too close to 0%, the AI learns to emergency close the trade. |
| **15-19** | *Reserved Pad* | `0.0` | Reserved for trailing stop-loss metrics. |

## 3. Core Market & Price Action (Slots 20 - 49)
**Purpose:** High-frequency data that tracks the immediate heartbeat of the price and volume.

| Slot | Feature Name | What it Measures | How it Helps the AI |
| :--- | :--- | :--- | :--- |
| **20-25** | `returns_multi` | Price change over 1s, 10s, 1m, 5m, 15m, 30m. | Identifies micro-breakouts and macro-trends instantly. |
| **28** | `price_range` | High minus Low over the last 30 mins. | Tells the AI if the market is choppy or trending smoothly. |
| **30-33** | `volume_metrics` | Total volume, Buy Vol, Sell Vol, Imbalance. | If Buy Vol vastly outweighs Sell Vol, it signals a massive pump is starting. |
| **34-38** | `vwap_metrics` | Volume-Weighted Avg Price (1m, 15m) & Deviation. | Shows the true "fair value" of the coin. If price drops way below VWAP, it's a dip-buy opportunity. |
| **39** | `realized_volatility`| Math calculation of recent price swings. | High volatility = AI takes smaller, safer positions. |
| **40** | `spread_bps` | Distance between best Bid and Ask. | If spread is wide, AI learns to avoid Market Orders to save on fees. |
| **43** | `ob_imbalance` | Top of the Order Book buy/sell pressure. | Predicts which direction the price will tick in the next second. |

## 4. Advanced Technical Indicators (Slots 55 - 89)
**Purpose:** Gives the AI the mathematical "cheat codes" used by human day traders.

| Slot | Feature Name | What it Measures | How it Helps the AI |
| :--- | :--- | :--- | :--- |
| **55-58** | `rsi_multi` | Relative Strength Index (1m, 5m, 15m). | **Overbought/Oversold.** Prevents the AI from buying the absolute top of a pump. |
| **59-61** | `macd_metrics` | MACD Line, Signal Line, Histogram. | Identifies trend reversals. When MACD crosses, the AI knows a dump/pump is reversing. |
| **62-65** | `ema_distances` | Distance to 9, 21, 50, 200 EMAs. | Dynamic Support/Resistance. The AI learns that price often bounces off the 21 EMA. |
| **68-69** | `bollinger_bands`| Band Width and Position (%B). | If position hits +1.0 (Upper Band), the AI knows a breakout or reversal is imminent. |

## 5. Candlestick Geometry (Slots 90 - 99)
**Purpose:** Teaches the AI to "read the charts" visually using math.

| Slot | Feature Name | What it Measures | How it Helps the AI |
| :--- | :--- | :--- | :--- |
| **90, 93, 96** | `upper_wicks` | Length of the top wicks (last 3 candles). | Giant upper wicks = Price Rejection (Bearish). AI learns to short. |
| **91, 94, 97** | `lower_wicks` | Length of the bottom wicks. | Giant lower wicks = "Hammer" pattern (Bullish). AI learns to long the dip. |
| **92, 95, 98** | `body_size` | Size of the colored candle body. | Detects "Engulfing" candles which signal massive momentum shifts. |

## 6. Derivatives & Liquidation Tracking (Slots 100 - 109)
**Purpose:** Tracking the "Smart Money" and betting against over-leveraged retail traders.

| Slot | Feature Name | What it Measures | How it Helps the AI |
| :--- | :--- | :--- | :--- |
| **100** | `funding_rate` | Are Longs paying Shorts? | High funding means Longs are over-leveraged. AI learns to Short and collect the free fees. |
| **101-102**| `liquidations` | Dollar amount of traders recently wiped out. | "Liquidation Hunting". When retail traders get wiped out, the AI buys their cheap bags. |

## 7. Macro & Cross-Asset Correlation (Slots 110 - 129)
**Purpose:** Gives the AI the "Big Picture" so it doesn't get tunnel vision on a single coin.

| Slot | Feature Name | What it Measures | How it Helps the AI |
| :--- | :--- | :--- | :--- |
| **110-113**| `time_encoding` | Sine/Cosine of the current Hour and Day. | Teaches the AI that 3 AM on a Sunday is lower volume than 9 AM on a Monday. |
| **116** | `gemini_sentiment`| Bearish vs Bullish NLP score. | Integrates macro news sentiment (if enabled). |
| **119** | `btc_mom_1m` | **Bitcoin 1-Minute Return.** | **Crucial:** If trading SOL, and SOL looks bullish, but BTC is dumping heavily, this slot tells the AI to cancel the SOL trade. |
| **120** | `btc_mom_15m` | Bitcoin 15-Minute Return. | Defines the overall market trend for the AI to follow. |

## 8. The Future-Proofing Canvas (Slots 130 - 199)
**Purpose:** Guarantees you never have to wipe the MySQL database again.

| Slot | Feature Name | What it Measures | How it Helps the AI |
| :--- | :--- | :--- | :--- |
| **130-199**| `PADDING` | Literally just `0.0`. | The AI mathematically ignores these. When Phase 5 arrives, we will silently replace these zeros with **Twitter Sentiment**, **Whale Wallet Tracking**, and **S&P 500 Data** without breaking the database structure. |
