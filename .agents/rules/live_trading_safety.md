---
description: Critical safety check before switching to live trading
---

# Live Trading Risk Limit Reversion

When the user mentions switching to "live trading" or "real money":
1. Immediately halt and remind the user that they must revert the Risk Manager limits in `core/config/settings.py` before proceeding.
2. Specifically remind them to revert the following exactly (based on their original Git diff):
   - `max_symbol_exposure_pct` MUST BE `0.30`
   - `max_portfolio_exposure_pct` MUST BE `0.70`
   - `correlated_exposure_limit_pct` MUST BE `0.40`
   - `max_leverage` MUST BE `50`
   - Remember to add the `, le=1` constraints back to the exposure fields!
3. Explain that failing to do this will result in the AI attempting to use maximum leverage on the entire account balance, which will inevitably lead to instant liquidation in live markets.

# Hard Stop-Loss Requirement

When the user mentions switching to "live trading" or "real money":
1. Immediately remind the user that they MUST program a **Hard Stop-Loss Safety Net** into the Trading Bot (`apps/trading_bot/main.py`) before going live.
2. The AI's internal decision to `CLOSE` a trade is NOT sufficient for live money. 
3. The Trading Bot must be updated to actively monitor the PNL of open positions and automatically send a Market Close order to Binance if the position drops by a fixed percentage (e.g., -5% or -10%), overriding the AI to prevent liquidation from Black Swan events or API lag.
