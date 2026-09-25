# V2 God Mode: 200-Slot AI Neural Blueprint (Fully Expanded)

This document contains the exact mathematical mapping of the 200 sensory inputs (slots) fed into the AI's Neural Network every single second. **Every single slot from 0 to 199 is explicitly defined below.**

---

## 1. Portfolio & Risk Metrics (Slots 0 - 9)
**Purpose:** Teaches the AI to manage your money and dynamically adjust risk based on its own performance.

| Slot | Feature Name | Description | Status / Alpha |
| :--- | :--- | :--- | :--- |
| **0** | `wallet_balance` | Total raw cash in the account. | Baseline size. |
| **1** | `equity` | Total cash + active unrealized profits/losses. | True live value of the account. |
| **2** | `free_margin` | Cash not currently locked in trades. | Tells AI exactly how much it can still spend. |
| **3** | `margin_utilization` | % of account currently locked. | Prevents over-trading. |
| **4** | `total_exposure` | Total dollar value of all open bets. | Risk balancing. |
| **5** | `portfolio_heat` | **[NEW]** % of maximum symbol slots currently active. | Tells AI to stop opening longs if it's already full. |
| **6** | `session_win_rate` | **[NEW]** Rolling win rate of the current session. | Self-awareness: If low, AI learns to stop trading. |
| **7** | `account_drawdown` | **[NEW]** % distance from All-Time High equity. | Capital preservation trigger. |
| **8** | `portfolio_variance`| **[NEW]** Statistical volatility of the account balance. | Deleveraging trigger. |
| **9** | `time_since_win` | **[NEW]** Hours since the last profitable trade closed. | Prevents "revenge trading" (impatience metric). |

---

## 2. Active Position Awareness (Slots 10 - 19)
**Purpose:** Teaches the AI how to manage a trade *after* it opens one.

| Slot | Feature Name | Description | Status / Alpha |
| :--- | :--- | :--- | :--- |
| **10** | `quantity` | Size of the current open trade. | Informs direction (Long/Short/Flat). |
| **11** | `entry_price` | Exact price the trade was opened at. | Baseline for PnL. |
| **12** | `leverage` | Normalized leverage (e.g. 20x). | Dictates stop-loss tightness. |
| **13** | `current_pnl_pct` | Live Unrealized Profit/Loss %. | The core trigger for standard taking-profit. |
| **14** | `dist_to_liq` | Distance to liquidation price. | Emergency close trigger. |
| **15** | `max_unrealized_pnl`| **[NEW]** The absolute highest profit this trade reached. | **Trailing Stop Loss anchor.** |
| **16** | `max_drawdown_pnl` | **[NEW]** The absolute lowest PNL this trade reached. | Measures trade distress. |
| **17** | `time_held_hours` | **[NEW]** How long the trade has been open. | Prevents holding dead trades forever. |
| **18** | `entry_dist_vwap` | **[NEW]** Distance from VWAP at exact time of entry. | Measures if the AI chased a pump or bought a dip. |
| **19** | `accumulated_funding`| **[NEW]** Total funding fees paid/received on this trade. | AI learns to close trades bleeding too much funding. |

---

## 3. Core Market & Price Action (Slots 20 - 49)
**Purpose:** High-frequency data that tracks the immediate heartbeat of the price and volume.

| Slot | Feature Name | Description | Status / Alpha |
| :--- | :--- | :--- | :--- |
| **20** | `ret_1s` | 1-second price return. | Micro-momentum. |
| **21** | `ret_10s` | 10-second price return. | High-frequency momentum. |
| **22** | `ret_1m` | 1-minute price return. | Scalping momentum. |
| **23** | `ret_5m` | 5-minute price return. | Short-term trend. |
| **24** | `ret_15m` | 15-minute price return. | Medium-term trend. |
| **25** | `ret_30m` | 30-minute price return. | Baseline trend. |
| **26** | `ret_24h` | **[NEW]** 24-Hour Return (Binance Ticker). | **Daily Macro Trend.** |
| **27** | `PADDING` | Reserved 0.0 | Future Expansion. |
| **28** | `price_range` | High minus Low (30m) normalized. | Volatility context. |
| **29** | `momentum` | Raw momentum proxy. | Trend strength. |
| **30** | `vol_1m` | Total 1-minute volume. | Activity spike detection. |
| **31** | `buy_vol_1m` | Total 1-minute BUY volume. | Bullish pressure. |
| **32** | `sell_vol_1m` | Total 1-minute SELL volume. | Bearish pressure. |
| **33** | `trade_imbalance` | (Buy Vol - Sell Vol) / Total Vol. | **Taker Buy/Sell Ratio.** |
| **34** | `vwap_1m` | 1-minute VWAP. | Micro fair-value. |
| **35** | `vwap_15m` | 15-minute VWAP. | Macro fair-value. |
| **36** | `vwap_dev_1m` | Price deviation from 1m VWAP. | Mean-reversion trigger. |
| **37** | `vwap_dev_15m` | Price deviation from 15m VWAP. | Mean-reversion trigger. |
| **38** | `realized_vol` | Standard deviation of log returns. | Risk scaling. |
| **39** | `vol_24h` | **[NEW]** 24-Hour Quote Volume. | **Daily Liquidity Check.** |
| **40** | `spread_bps` | Distance between Bid and Ask. | Fee awareness. |
| **41** | `bid_qty` | Total size of best Bid. | Orderbook support. |
| **42** | `ask_qty` | Total size of best Ask. | Orderbook resistance. |
| **43** | `ob_imbalance` | (Bid - Ask) / (Bid + Ask). | **Whale Wall Detection.** |
| **44** | `ob_skew_l2` | **[NEW]** Depth 5 Level 2 Orderbook Skew. | **Hidden Whale Wall Detection.** |
| **45** | `PADDING` | Reserved 0.0 | Future L3 Orderbook. |
| **46** | `PADDING` | Reserved 0.0 | Future L3 Orderbook. |
| **47** | `PADDING` | Reserved 0.0 | Future L3 Orderbook. |
| **48** | `PADDING` | Reserved 0.0 | Future L3 Orderbook. |
| **49** | `PADDING` | Reserved 0.0 | Future L3 Orderbook. |

---

## 4. Advanced Technical Indicators (Slots 50 - 89)
**Purpose:** Mathematical definitions of trends and oscillators.

| Slot | Feature Name | Description | Status / Alpha |
| :--- | :--- | :--- | :--- |
| **50** | `hurst_exponent` | **[NEW]** Fractal math market regime calculation. | **If <0.5, market is chopping. If >0.5, trending.** |
| **51** | `autocorrelation`| **[NEW]** 1-min return auto-correlation (ACF). | **StatArb momentum continuation check.** |
| **52** | `return_skewness`| **[NEW]** Skewness of last 30 mins returns. | **Detects asymmetric crash risks.** |
| **53** | `return_kurtosis`| **[NEW]** Kurtosis of last 30 mins returns. | **Fat-Tail Risk Detection.** |
| **54** | `PADDING` | Reserved 0.0 | Future L3 Orderbook. |
| **55** | `rsi_1m` | 1-minute Relative Strength Index. | Overbought/Oversold. |
| **56** | `rsi_5m` | 5-minute RSI. | Medium trend exhaustion. |
| **57** | `rsi_15m` | 15-minute RSI. | Macro trend exhaustion. |
| **58** | `rsi_1h` | 1-hour RSI. | Baseline exhaustion. |
| **59** | `macd_line` | MACD Line. | Momentum shift. |
| **60** | `macd_sig` | MACD Signal Line. | Cross-over detection. |
| **61** | `macd_hist` | MACD Histogram. | Trend acceleration. |
| **62** | `ema_9_dist` | Price distance to 9 EMA. | Immediate support/resistance. |
| **63** | `ema_21_dist` | Price distance to 21 EMA. | Scalp trend boundary. |
| **64** | `ema_50_dist` | Price distance to 50 EMA. | Medium trend boundary. |
| **65** | `ema_200_dist` | Price distance to 200 EMA. | Macro trend boundary. |
| **66** | `sma_50_dist` | Price distance to 50 SMA. | Alternative smooth boundary. |
| **67** | `sma_200_dist` | Price distance to 200 SMA. | 'Golden Cross' baseline. |
| **68** | `bb_width` | Bollinger Band Width. | Volatility squeeze detection. |
| **69** | `bb_pos` | Position inside Bollinger Bands (%B). | Breakout indicator. |
| **70** | `ema_4h_dist` | **[NEW]** Price distance to 4-Hour EMA. | **Institutional Macro Trend Alignment.** |
| **71** | `atr` | **[NEW]** Average True Range. | **True absolute volatility metric.** |
| **72** | `stoch_k` | **[NEW]** Stochastic Oscillator %K. | **Momentum top/bottom catching.** |
| **73** | `stoch_d` | **[NEW]** Stochastic Oscillator %D. | **Momentum moving average.** |
| **74** | `adx` | **[NEW]** Average Directional Index. | **Absolute Trend Strength (0-100).** |
| **75-89**| `PADDING` | Reserved 0.0 | (15 Slots for Future Technicals). |

---

## 5. Candlestick Geometry (Slots 90 - 99)
**Purpose:** Teaches the AI to "read the charts" visually using exact math of the last 3 candles.

| Slot | Feature Name | Description | Status / Alpha |
| :--- | :--- | :--- | :--- |
| **90** | `candle_1_upper`| Upper wick of 1st candle. | Price rejection (Bearish). |
| **91** | `candle_1_lower`| Lower wick of 1st candle. | Dip absorption (Bullish). |
| **92** | `candle_1_body` | Body size of 1st candle. | Engulfing momentum. |
| **93** | `candle_2_upper`| Upper wick of 2nd candle. | Prior rejection history. |
| **94** | `candle_2_lower`| Lower wick of 2nd candle. | Prior absorption history. |
| **95** | `candle_2_body` | Body size of 2nd candle. | Prior momentum context. |
| **96** | `candle_3_upper`| Upper wick of 3rd candle. | Deep rejection history. |
| **97** | `candle_3_lower`| Lower wick of 3rd candle. | Deep absorption history. |
| **98** | `candle_3_body` | Body size of 3rd candle. | Deep momentum context. |
| **99** | `PADDING` | Reserved 0.0 | Future Expansion. |

---

## 6. Derivatives & Liquidation Tracking (Slots 100 - 109)
**Purpose:** Tracking the "Smart Money" and betting against over-leveraged retail traders.

| Slot | Feature Name | Description | Status / Alpha |
| :--- | :--- | :--- | :--- |
| **100** | `funding_rate` | Are Longs paying Shorts? | Indicates retail over-leverage. |
| **101** | `liq_1m` | 1-minute Liquidations. | Detects massive liquidation cascades immediately. |
| **102** | `liq_15m` | 15-minute Liquidations. | Identifies "cleared out" market bottoms. |
| **103** | `oi_1m_delta` | **[NEW]** Open Interest 1-min Change. | **Detects if a pump is backed by real money or fake volume.** |
| **104** | `ls_ratio` | **[NEW]** Binance Top Trader Long/Short Ratio. | **Follow the smart money whales.** |
| **105** | `mark_price_premium`| **[NEW]** Mark Price vs Index Price. | **Impending Liquidation Risk.** |
| **106** | `PADDING` | Reserved 0.0 | Future expansion. |
| **107** | `PADDING` | Reserved 0.0 | Future expansion. |
| **108** | `PADDING` | Reserved 0.0 | Future expansion. |
| **109** | `PADDING` | Reserved 0.0 | Future expansion. |

---

## 7. Macro & Cross-Asset Correlation (Slots 110 - 129)
**Purpose:** Gives the AI the "Big Picture" globally.

| Slot | Feature Name | Description | Status / Alpha |
| :--- | :--- | :--- | :--- |
| **110** | `time_sin` | Sine of minute of day. | Identifies daily volume cycles. |
| **111** | `time_cos` | Cosine of minute of day. | Identifies daily volume cycles. |
| **112** | `day_sin` | Sine of day of week. | Identifies weekend vs weekday volume. |
| **113** | `day_cos` | Cosine of day of week. | Identifies weekend vs weekday volume. |
| **114** | `cross_coin_beta`| **[NEW]** Altcoin momentum vs BTC momentum. | **Identifies extreme relative weakness/strength.** |
| **115** | `PADDING` | Reserved 0.0 | Future expansion. |
| **116** | `gemini_sentiment`| Bearish vs Bullish NLP score. | Live global news sentiment. |
| **117** | `vol_expectation`| Gemini Volatility Expectation. | Warns AI of upcoming FOMC or CPI spikes. |
| **118** | `regime` | Market Regime (-1, 0, 1). | Simplistic risk toggle. |
| **119** | `btc_mom_1m` | Bitcoin 1-Minute Return. | The gravitational pull of crypto. |
| **120** | `btc_mom_15m` | Bitcoin 15-Minute Return. | Baseline crypto macro trend. |
| **121** | `sp500_momentum` | **[NEW]** S&P 500 Stock Market Momentum. | **Avoids buying crypto if Wall Street is crashing.** |
| **122** | `dxy_momentum` | **[NEW]** US Dollar Index (DXY) Momentum. | **Avoids buying crypto if the US Dollar is spiking.** |
| **123** | `fear_greed_index`| **[NEW]** Global Crypto Fear & Greed Index. | **Tracks maximum retail euphoria and panic.** |
| **124** | `vix_momentum` | **[NEW]** CBOE Volatility Index (Fear Gauge). | **Predicts global Wall Street crashes.** |
| **125** | `gold_momentum` | **[NEW]** Gold (GC=F) Price Momentum. | **Inflation and flight-to-safety tracker.** |
| **126** | `treasury_yield_mom`| **[NEW]** US 10-Year Treasury Yield Momentum. | **Liquidity trap / Risk-off detection.** |
| **127** | `ndx_momentum`| **[NEW]** NASDAQ 100 (^NDX) Momentum. | **High-Beta Tech Stock Proxy for Crypto.** |
| **128** | `defi_tvl_momentum`| **[NEW]** DefiLlama Ethereum TVL Momentum. | **Tracks liquidity draining from smart contracts.** |
| **129** | `PADDING` | Reserved 0.0 | Future expansion. |

---

## 8. The Future-Proofing Canvas (Slots 130 - 199)
**Purpose:** 70 completely blank slots guaranteeing we never have to wipe the PyTorch Database for the next 5 years.

| Slot | Feature Name | Description | Status / Alpha |
| :--- | :--- | :--- | :--- |
| **130** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **131** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **132** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **133** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **134** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **135** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **136** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **137** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **138** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **139** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **140** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **141** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **142** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **143** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **144** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **145** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **146** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **147** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **148** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **149** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **150** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **151** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **152** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **153** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **154** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **155** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **156** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **157** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **158** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **159** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **160** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **161** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **162** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **163** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **164** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **165** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **166** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **167** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **168** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **169** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **170** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **171** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **172** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **173** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **174** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **175** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **176** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **177** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **178** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **179** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **180** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **181** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **182** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **183** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **184** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **185** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **186** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **187** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **188** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **189** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **190** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **191** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **192** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **193** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **194** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **195** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **196** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **197** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **198** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
| **199** | `PADDING` | Reserved 0.0 | Awaiting Institutional Data |
