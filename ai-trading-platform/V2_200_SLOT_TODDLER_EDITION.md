# 🧠 The Robot's Brain: V2 200-Slot God Mode (Toddler Edition!)

**Current Status:** 108 Active Senses | 92 Empty Spaces (For Future Growth)

Here is the exact mapping of every single slot (0 to 199) from your `V2_200_SLOT_BLUEPRINT.md`. Think of each slot as a specific sensor on the robot, telling it exactly what is happening in the world!

---

## 1. The Piggy Bank & Feelings (Slots 0 - 9)
*The robot checking its wallet and calming its nerves.*

| Slot | The Toddler Meaning | How It's Calculated | How The AI Uses It |
| :--- | :--- | :--- | :--- |
| **0 (`wallet_balance`)** | Total cash. *(Do I have $10 or $1,000?)* | Fetched from Binance account balance (USD). | To decide if it can afford a new trade. |
| **1 (`equity`)** | Cash + active trades. *(How much am I actually worth right now?)* | Wallet Balance + Unrealized PnL of open positions. | To track if it's growing or shrinking overall. |
| **2 (`free_margin`)** | Unlocked cash. *(How much money is left for shopping?)* | Equity minus margin locked in open trades. | To know exactly how much it can spend right now. |
| **3 (`margin_utilization`)** | % of money locked. *(Have I spent too much?)* | (Locked Margin / Equity) * 100. | To learn risk management (don't bet 100% at once). |
| **4 (`total_exposure`)** | Size of all bets. *(Am I holding too many heavy bags?)* | Sum of all open position sizes in USD. | To avoid over-leveraging across multiple coins. |
| **5 (`portfolio_heat`)** | Max capacity. *(Is my backpack full?)* | Exposure divided by Equity. | To trigger a "stop trading" rule if it gets too hot. |
| **6 (`session_win_rate`)** | Winning streak. *(Am I winning today?)* | Wins / Total Trades in the last 24h. | To boost confidence or force it to take a break. |
| **7 (`account_drawdown`)** | Money lost from the top. *(Did I just lose money?)* | (Peak Equity - Current Equity) / Peak Equity. | To trigger "defensive mode" and reduce trade sizes. |
| **8 (`portfolio_variance`)** | Money rollercoaster. *(Is my money bouncing fast?)* | Volatility of the portfolio's value over time. | To avoid chaotic, unpredictable market conditions. |
| **9 (`time_since_win`)** | Impatience timer. *(Has it been a long time since I won?)* | Minutes since the last profitable trade closed. | To prevent "revenge trading" when it gets frustrated. |

---

## 2. The Current Trade Watcher (Slots 10 - 19)
*The robot staring at the trade it just opened.*

| Slot | The Toddler Meaning | How It's Calculated | How The AI Uses It |
| :--- | :--- | :--- | :--- |
| **10 (`quantity`)** | Trade size. *(Did I buy 1 or 10 Bitcoins?)* | Amount of crypto currently held (from Binance). | To calculate potential profits/losses on price moves. |
| **11 (`entry_price`)** | Buying price. *(I bought it at $50k. Don't forget!)* | The average price of the open position. | To anchor its calculations for Take-Profit/Stop-Loss. |
| **12 (`leverage`)** | Borrowed money. *(Did I borrow 20x?)* | Leverage multiplier fetched from Binance API. | To understand how fast the PnL will move. |
| **13 (`current_pnl_pct`)** | Live score. *(Am I winning or losing right now?)* | (Current Price - Entry) / Entry * Leverage. | To decide if it should close the trade right now. |
| **14 (`dist_to_liq`)** | The Cliff. *(How close am I to losing everything?)* | % difference between Current Price and Liquidation Price. | To panic-close the trade before Binance liquidates it. |
| **15 (`max_unrealized_pnl`)** | The High Score. *(Highest profit I saw?)* | The highest `current_pnl_pct` recorded during the trade. | To learn "Trailing Stop Loss" (sell if it drops from the top). |
| **16 (`max_drawdown_pnl`)** | The Scary Score. *(Lowest my money went?)* | The lowest `current_pnl_pct` recorded during the trade. | To recognize dangerous volatility and avoid holding toxic bags. |
| **17 (`time_held_hours`)** | Stop-watch. *(I've held this for 5 hours?)* | Current Time - Trade Open Time. | To close "stuck" trades that aren't doing anything. |
| **18 (`entry_dist_vwap`)** | FOMO check. *(Did I buy when it was too expensive?)* | % difference between Entry Price and VWAP. | To learn not to buy at the very top of a pump. |
| **19 (`accumulated_funding`)** | Hidden fees. *(Is Binance secretly charging me?)* | Sum of all funding fees paid while holding. | To learn that holding forever slowly bleeds money. |

---

## 3. The Price Heartbeat (Slots 20 - 49)
*The robot feeling the pulse of the market.*

| Slot | The Toddler Meaning | How It's Calculated | How The AI Uses It |
| :--- | :--- | :--- | :--- |
| **20 (`ret_1s`)** | 1-second step. *(Did it just take a tiny step up?)* | % change in price over the last 1 second. | To detect high-frequency micro-pumps. |
| **21 (`ret_10s`)** | 10-second run. *(Is it starting to jog?)* | % change in price over the last 10 seconds. | To confirm short-term momentum. |
| **22 (`ret_1m`)** | 1-minute sprint. *(Is it sprinting now?)* | % change in price over the last 60 seconds. | To ride fast momentum waves. |
| **23 (`ret_5m`)** | 5-minute hike. *(Which way is the hill sloping?)* | % change in price over the last 5 minutes. | To find the local micro-trend. |
| **24 (`ret_15m`)** | 15-minute road. *(Highway up or down?)* | % change in price over the last 15 minutes. | To confirm the scalp trend direction. |
| **25 (`ret_30m`)** | 30-minute map. *(What does the city look like?)* | % change in price over the last 30 minutes. | To align trades with the macro swing. |
| **26 (`ret_24h`)** | The Daily News. *(Good day or bad day?)* | % change in price over the last 24 hours. | To know if the overall market is bullish or bearish. |
| **27 (`PADDING`)** | *Empty* | N/A | Reserved for future timeframes. |
| **28 (`price_range`)** | The Earthquake. *(Is the price shaking?)* | (High - Low) / Low of the last minute. | To know if it should use wider stop-losses. |
| **29 (`momentum`)** | The Rocket Booster. *(Speeding up?)* | Difference between fast and slow moving averages. | To buy into accelerating trends. |
| **30 (`vol_1m`)** | The Crowd Size. *(10 people or 10,000?)* | Total trading volume (USD) in the last minute. | To ensure there is enough liquidity to enter safely. |
| **31 (`buy_vol_1m`)** | The Happy Crowd. *(How many scream "BUY"?)* | Volume of Maker-Sell (Taker-Buy) trades. | To measure raw buyer aggression. |
| **32 (`sell_vol_1m`)** | The Scared Crowd. *(How many scream "SELL"?)* | Volume of Maker-Buy (Taker-Sell) trades. | To measure raw seller aggression. |
| **33 (`trade_imbalance`)** | Tug-of-war. *(Buyers or sellers winning?)* | (Buy Vol - Sell Vol) / Total Vol. | To predict the immediate next price tick. |
| **34 (`vwap_1m`)** | 1-minute Fair Price. *(Is this a fair deal?)* | Volume Weighted Average Price (1m). | As a baseline to mean-revert to. |
| **35 (`vwap_15m`)** | 15-minute Fair Price. *(Fair deal for the morning?)* | Volume Weighted Average Price (15m). | As a strong magnetic support/resistance level. |
| **36 (`vwap_dev_1m`)** | The Rubber Band (Small). *(Stretched too far?)* | % distance of Current Price from 1m VWAP. | To predict snap-backs (mean reversion scalps). |
| **37 (`vwap_dev_15m`)** | The Rubber Band (Big). *(Stretched WAY too far?)* | % distance of Current Price from 15m VWAP. | To predict major structural reversals. |
| **38 (`realized_vol`)** | The Danger Meter. *(Too dangerous to trade?)* | Standard deviation of 1-second returns. | To dynamically adjust trade sizes (high vol = small size). |
| **39 (`vol_24h`)** | The Ocean Size. *(Giant ocean or puddle?)* | Total 24h trading volume from Binance. | To filter out dead, illiquid shitcoins. |
| **40 (`spread_bps`)** | The Tax. *(Giant hidden tax?)* | (Ask - Bid) / Mid Price * 10000. | To avoid trading when slippage would destroy profits. |
| **41 (`bid_qty`)** | The Safety Net. *(People waiting below?)* | Total volume sitting on the orderbook Bids. | To identify strong price floors. |
| **42 (`ask_qty`)** | The Ceiling. *(People waiting above?)* | Total volume sitting on the orderbook Asks. | To identify strong price ceilings. |
| **43 (`ob_imbalance`)** | The Wall Weight. *(Is ceiling heavier than net?)* | (Bid Qty - Ask Qty) / (Bid Qty + Ask Qty). | To predict orderbook exhaustion. |
| **44 (`ob_skew_l2`)** | The Hidden Walls. *(Whales hiding in the dark?)* | Imbalance of the top 20 orderbook levels. | To see where the deep liquidity is stacked. |
| **45-49 (`PADDING`)**| *Empty* | N/A | Reserved for deeper L2 Orderbook analysis. |

---

## 4. The Math Nerd (Slots 50 - 89)
*The robot doing complex geometry on the chart.*

| Slot | The Toddler Meaning | How It's Calculated | How The AI Uses It |
| :--- | :--- | :--- | :--- |
| **50 (`hurst_exponent`)** | The Chopping Block. *(Sideways or moving?)* | Statistical test for long-term memory of a time series. | To choose between trend-following or mean-reversion strategies. |
| **51 (`autocorrelation`)** | The Copycat. *(Will next minute copy last?)* | Correlation of price returns with their own past. | To detect sticky, persistent momentum. |
| **52 (`return_skewness`)** | The Trapdoor. *(Secret risk of crashing?)* | Asymmetry of the return distribution. | To avoid coins that are prone to sudden flash crashes. |
| **53 (`return_kurtosis`)** | The Black Swan. *(Crazy event about to happen?)* | Fat-tailedness of the return distribution. | To detect extreme outlier behavior. |
| **54 (`PADDING`)** | *Empty* | N/A | Reserved. |
| **55 (`rsi_1m`)** | 1-minute Exhaustion. *(Buyers tired yet?)* | Relative Strength Index (1m). | To buy oversold dips and sell overbought rips. |
| **56 (`rsi_5m`)** | 5-minute Exhaustion. *(Really tired?)* | Relative Strength Index (5m). | To confirm reversals on a medium timeframe. |
| **57 (`rsi_15m`)** | 15-minute Exhaustion. *(Collapsing from exhaustion?)* | Relative Strength Index (15m). | To confirm major local tops and bottoms. |
| **58 (`rsi_1h`)** | 1-hour Exhaustion. *(Asleep! Time to sell!)* | Relative Strength Index (1h). | To align with the massive macro trend. |
| **59 (`macd_line`)** | The Train Speed. *(Up-train faster than down?)* | Difference between 12-EMA and 26-EMA. | To measure absolute trend strength. |
| **60 (`macd_sig`)** | The Train Track Switch. *(Train change directions?)* | 9-EMA of the MACD Line. | As a trigger for trend-reversal trades. |
| **61 (`macd_hist`)** | Train Acceleration. *(Train hit the brakes?)* | MACD Line - MACD Signal. | To detect momentum slowing down before price actually reverses. |
| **62 (`ema_9_dist`)** | The Tiny Bumper. *(Hit the first bumper?)* | % distance from the 9-period EMA. | To trade the "fast lane" breakouts. |
| **63 (`ema_21_dist`)** | The Small Wall. *(Hit the brick wall?)* | % distance from the 21-period EMA. | To trade standard pullbacks. |
| **64 (`ema_50_dist`)** | The Big Wall. *(Hit the steel wall?)* | % distance from the 50-period EMA. | To identify the medium-term support. |
| **65 (`ema_200_dist`)** | The Giant Wall. *(Hit the Titanium wall?)* | % distance from the 200-period EMA. | To identify the absolute baseline trend. |
| **66 (`sma_50_dist`)** | The Smooth Wall. *(Alternative steel wall)* | % distance from the 50-period Simple Moving Avg. | To confirm EMA signals with a less reactive line. |
| **67 (`sma_200_dist`)** | The Golden Wall. *(Cross this = crazy!)* | % distance from the 200-period SMA. | To detect legendary "Golden Crosses" or "Death Crosses". |
| **68 (`bb_width`)** | The Squeeze. *(Explosion about to happen?)* | Width between Upper and Lower Bollinger Bands. | To buy right before a massive volatility breakout. |
| **69 (`bb_pos`)** | The Pop. *(Balloon pop up or down?)* | Position of price relative to the Bands (0-1). | To detect when price breaks outside normal limits. |
| **70 (`ema_4h_dist`)** | The Bankers' Line. *(What is Wall St. looking at?)* | % distance from the 4-Hour EMA. | To align with institutional, slow-moving money. |
| **71 (`atr`)** | The Measuring Stick. *(How many $ per candle?)* | Average True Range over 14 periods. | To perfectly calculate Stop-Loss distances. |
| **72 (`stoch_k`)** | The Bouncy Ball. *(Ball at ceiling or floor?)* | Stochastic Oscillator %K line. | To find exact tops and bottoms in ranging markets. |
| **73 (`stoch_d`)** | The Ball's Shadow. *(Which way is ball heading?)* | Stochastic Oscillator %D line (SMA of %K). | To confirm Stochastic crossovers. |
| **74 (`adx`)** | The Trend King. *(Real trend, or fake trend?)* | Average Directional Index. | To tell the AI: "Stop using RSI, use moving averages instead!" |
| **75-89 (`PADDING`)** | *Empty* | N/A | Reserved for 15 future math indicators. |

---

## 5. The Picture Book (Slots 90 - 99)
*The robot looking at the shapes of the last 3 candles.*

| Slot | The Toddler Meaning | How It's Calculated | How The AI Uses It |
| :--- | :--- | :--- | :--- |
| **90 (`candle_1_upper`)**| The Head. *(Slapped down?)* | Length of the top wick of the last closed candle. | To detect seller rejection at highs. |
| **91 (`candle_1_lower`)**| The Legs. *(Pushed back up?)* | Length of the bottom wick of the last closed candle. | To detect buyer support at lows. |
| **92 (`candle_1_body`)** | The Belly. *(How fat was it?)* | Open-Close distance of the last closed candle. | To measure absolute buying/selling power. |
| **93 (`candle_2_upper`)**| Yesterday's Head. | Top wick of the candle before last. | To recognize 2-candle rejection patterns. |
| **94 (`candle_2_lower`)**| Yesterday's Legs. | Bottom wick of the candle before last. | To recognize 2-candle support patterns. |
| **95 (`candle_2_body`)** | Yesterday's Belly. | Body size of the candle before last. | To confirm momentum continuation. |
| **96 (`candle_3_upper`)**| Grandpa's Head. | Top wick of the 3rd candle back. | For 3-candle patterns (Morning Star, etc). |
| **97 (`candle_3_lower`)**| Grandpa's Legs. | Bottom wick of the 3rd candle back. | For 3-candle patterns. |
| **98 (`candle_3_body`)** | Grandpa's Belly. | Body size of the 3rd candle back. | For 3-candle patterns (Three White Soldiers, etc). |
| **99 (`PADDING`)** | *Empty* | N/A | Reserved. |

---

## 6. The Gambler Spy (Slots 100 - 109)
*The robot watching people who borrow too much money.*

| Slot | The Toddler Meaning | How It's Calculated | How The AI Uses It |
| :--- | :--- | :--- | :--- |
| **100 (`funding_rate`)** | The Rent. *(Up-Gamblers paying rent?)* | Binance Futures live funding rate. | To trade against the over-leveraged majority. |
| **101 (`liq_1m`)** | The Quick Wipeout. *(Someone lose everything?)* | Sum of liquidations in the last minute. | To buy the instant panic cascades. |
| **102 (`liq_15m`)** | The Big Wipeout. *(Whole group lose everything?)* | Sum of liquidations in the last 15 minutes. | To confirm a total market reset. |
| **103 (`oi_1m_delta`)**| The Fake Pump Check. *(Real or fake money?)* | Change in Open Interest over 1 minute. | If price rises but OI drops, it's a fake short-squeeze. |
| **104 (`ls_ratio`)** | The Sheep Tracker. *(Herd betting UP?)* | Binance Long/Short Accounts Ratio. | To do the exact opposite of retail traders (contrarian). |
| **105 (`mark_price_premium`)**| The Warning Siren. *(Exchange force-close?)* | Difference between Mark Price and Index Price. | To avoid trades right before massive liquidations. |
| **106-109 (`PADDING`)**| *Empty* | N/A | Reserved for future. |

---

## 7. The Global Weather (Slots 110 - 129)
*The robot checking the real world outside of Crypto.*

| Slot | The Toddler Meaning | How It's Calculated | How The AI Uses It |
| :--- | :--- | :--- | :--- |
| **110 (`time_sin`)** | The Sun. *(Morning in NY?)* | Sine wave of the current hour of day. | To learn time-of-day behaviors (NY Open vs Asia Open). |
| **111 (`time_cos`)** | The Moon. *(Midnight in Tokyo?)* | Cosine wave of the current hour of day. | Combined with sine to make a perfect 24h circle. |
| **112 (`day_sin`)** | The Workday. *(Is it Monday?)* | Sine wave of the day of the week. | To learn weekend vs weekday volume drops. |
| **113 (`day_cos`)** | The Weekend. *(Is it Sunday? Boring...)* | Cosine wave of the day of the week. | Combined with sine to make a 7-day circle. |
| **114 (`cross_coin_beta`)** | The Follower. *(Blindly following BTC?)* | Correlation of this coin to Bitcoin's moves. | If BTC dumps, to know if this coin will dump too. |
| **115 (`PADDING`)** | *Empty* | N/A | Reserved. |
| **116 (`gemini_sentiment`)**| The News Mood. *(Smiling or crying?)* | AI NLP analysis of latest Crypto News headlines. | To avoid shorting during highly positive news hype. |
| **117 (`vol_expectation`)** | The Warning Calendar. *(President speaking?)* | Economic Calendar flags (CPI, FOMC, etc). | To close trades before massive news drops. |
| **118 (`regime`)** | The Green/Red Light. *(Safe or dangerous?)* | Market Regime detection algorithm. | To switch entirely between Bull, Bear, or Crab strategies. |
| **119 (`btc_mom_1m`)** | The King's Mood (Fast). *(BTC running?)* | Bitcoin 1m return. | To use Bitcoin as a leading indicator for altcoins. |
| **120 (`btc_mom_15m`)** | The King's Mood (Slow). *(BTC walking?)* | Bitcoin 15m return. | To ensure the altcoin trend matches the macro BTC trend. |
| **121 (`sp500_momentum`)** | The Stock Market. *(Normal companies doing well?)* | S&P 500 futures momentum. | To detect macro risk-on vs risk-off days. |
| **122 (`dxy_momentum`)** | The US Dollar. *(Cash too strong?)* | DXY (Dollar Index) momentum. | To short crypto if the US dollar is skyrocketing. |
| **123 (`fear_greed_index`)**| The Ultimate Crowd Score. *(Incredibly greedy?)* | Global Crypto Fear & Greed Index. | To sell when everyone is greedy, buy when fearful. |
| **124 (`vix_momentum`)** | The Terror Gauge. *(Wall Street terrified?)* | VIX (Volatility Index) momentum. | To go to cash if the global stock market is crashing. |
| **125 (`gold_momentum`)** | The Safety Rock. *(Hiding money in Gold?)* | Gold futures momentum. | To detect global inflation panics. |
| **126 (`treasury_yield_mom`)**| The Government Bonds. *(Paying high interest?)* | US 10Y Treasury yield momentum. | To detect macro interest rate shifts. |
| **127 (`ndx_momentum`)** | The Tech Companies. *(Apple/Google going up?)* | Nasdaq 100 futures momentum. | Crypto is highly correlated to tech stocks. |
| **128 (`defi_tvl_momentum`)** | The Crypto Banks. *(Money locked in smart contracts?)* | Total Value Locked (TVL) in DeFi protocols. | To measure on-chain health of the crypto ecosystem. |
| **129 (`PADDING`)** | *Empty* | N/A | Reserved. |

---

## 8. The Future-Proofing Canvas (Slots 130 - 199)
*The blank spaces so we never run out of room for the next 5 years!*

| Slot | The Toddler Meaning | How It's Calculated | How The AI Uses It |
| :--- | :--- | :--- | :--- |
| **130 (`buy_wall_distance`)** | Whale Spotter (Below). *(Millionaire waiting to buy?)* | Distance to the largest orderbook bid > $100k. | To set a Stop-Loss right below the whale's wall. |
| **131 (`sell_wall_distance`)**| Whale Spotter (Above). *(Millionaire waiting to sell?)* | Distance to the largest orderbook ask > $100k. | To set a Take-Profit right before the whale's wall. |
| **132 (`whale_buy_pressure`)**| Whale Splashes. *(Millionaire bought a ton?)* | Sum of all market buys > $50k in the last second. | To front-run institutional buying pressure. |
| **133 (`whale_sell_pressure`)**| Whale Dumps. *(Millionaire sold a ton?)* | Sum of all market sells > $50k in the last second. | To escape immediately when institutions dump. |
| **134-139 (`PADDING`)** | *Empty* | N/A | Reserved for 6 more whale trackers. |
| **140 (`okx_premium`)** | The OKX Spy. *(Price higher on OKX?)* | Price diff between Binance and OKX. | For statistical arbitrage across exchanges. |
| **141 (`bybit_premium`)** | The Bybit Spy. *(Price higher on Bybit?)* | Price diff between Binance and Bybit. | For statistical arbitrage across exchanges. |
| **142 (`okx_momentum_lead`)** | The Time Machine. *(Did OKX move first?)* | 1-second lag correlation between OKX and Binance. | If OKX spikes first, the AI buys Binance instantly. |
| **143-199 (`PADDING`)** | *57 completely empty slots.* | N/A | Saved for whatever crazy new data we invent in the year 2030! |

---
**And that is how the robot sees the universe in 200 numbers!**
