---
description: Critical safety check before switching to live trading
---

# Live Trading Risk Limit Reversion

When the user mentions switching to "live trading" or "real money":
1. Immediately halt and remind the user that they must revert the Risk Manager limits in `core/config/settings.py` before proceeding.
2. Specifically remind them to revert the following exactly:
   - `max_symbol_exposure_pct` MUST BE `0.20` (was changed to 25.0)
   - `max_portfolio_exposure_pct` MUST BE `0.50` (was changed to 25.0)
   - `correlated_exposure_limit_pct` MUST BE `0.40` (was changed to 25.0)
3. Explain that failing to do this will result in the AI attempting to use maximum leverage on the entire account balance, which will inevitably lead to instant liquidation in live markets.
