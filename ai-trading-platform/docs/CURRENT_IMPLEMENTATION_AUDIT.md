# Current Implementation Audit

## 1. Core Applications
*   **ProductionTradingBot** (`apps/trading_bot/main.py`)
    *   **Status**: `PARTIAL / BROKEN`
    *   **Dependencies**: Redis, ModelRegistry, BinanceFuturesClient, RiskManager
    *   **Problems**: Mismatch with RiskManager interface (`evaluate_risk` signatures differ). Doesn't load validated production checkpoint reliably. Hardcodes feature index `6` as `mid_price`.
    *   **Recommended Fix**: Refactor state construction to use canonical schemas. Fix RiskManager interface mismatch. Add strict model checkpoint loading validation.

*   **OnlineTrainer** (`apps/trading_bot/online_trainer.py`)
    *   **Status**: `MOCK / DANGEROUS`
    *   **Dependencies**: MySQL, Redis, PyTorch
    *   **Problems**: Contains `dummy_loss = torch.tensor(0.1, requires_grad=True)` and `dummy_loss.backward()`.
    *   **Recommended Fix**: Replace with actual batch sampling from persistent Replay Buffer and calculate real Temporal Difference / Policy Gradient losses.

## 2. Research & Environment Layer
*   **MultiAssetFuturesEnv** (`apps/research/environment.py`)
    *   **Status**: `PARTIAL`
    *   **Dependencies**: Gym, Data Pipeline
    *   **Problems**: Relies on dictionary-based state and generic Binance Tiered liquidation rules instead of fetching from exchange info. `macro_dim` handling is loose. 
    *   **Recommended Fix**: Use canonical `StateSchema` instead of ad-hoc dictionary passing. Integrate real reward calculations (drawdown penalties).

*   **World Model** (`apps/research/world_model.py`)
    *   **Status**: `IMPLEMENTED` (Prototype)
    *   **Dependencies**: PyTorch
    *   **Problems**: Uses mock data in training.
    *   **Recommended Fix**: Feed real replay buffer experiences into RSSM. Add strict tensor shape validation.

## 3. Data & Feature Engineering
*   **FeatureEngine / Market Collector**
    *   **Status**: `PARTIAL`
    *   **Dependencies**: Binance WebSocket
    *   **Problems**: Features are passed as opaque 25-dim arrays. Hard-coded index assumption for `mid_price` (`features[6]`).
    *   **Recommended Fix**: Define explicit Pydantic schemas for `MarketState`. Separate explicit metadata (prices, timestamps) from normalized feature vectors.

## 4. Execution & Risk Layer
*   **RiskManager** (`core/risk/risk_manager.py`)
    *   **Status**: `BROKEN` (Interface mismatch)
    *   **Dependencies**: None
    *   **Problems**: Expected arguments do not match caller arguments in `ProductionTradingBot`. Does not account for comprehensive portfolio-level correlation and isolated/cross margin fully.
    *   **Recommended Fix**: Standardize interface to `evaluate(order_request, portfolio_state, market_state)`. Add explicit unit tests for every rejection condition.

*   **BinanceFuturesClient** (`core/exchange/binance_client.py`)
    *   **Status**: `PARTIAL`
    *   **Dependencies**: aiohttp, config
    *   **Problems**: Lack of comprehensive Order/Execution state machine tracking. User Data Stream (account websocket) is missing.
    *   **Recommended Fix**: Implement User Data stream for real-time reconciliation. Build `ExchangeAdapter` hierarchy.

## 5. Persistence
*   **Experience Repository / MySQL**
    *   **Status**: `MISSING / PARTIAL`
    *   **Dependencies**: Redis, MySQL
    *   **Problems**: Real experience data is not reliably persisted in an `experiences` table with full state/action/reward reproducible context. Online trainer mocks the experience.
    *   **Recommended Fix**: Implement `Experience` schema and write explicitly to MySQL. Build an SQL-backed Replay Buffer cache.

## Execution Safety Summary
*   **Status**: `DANGEROUS` (without explicit gating).
*   Live execution is currently disabled via commented code (`# await self.binance.create_order`), but lacks structural multi-gate safety rules (`DRY_RUN` mode, `TRADING_ENABLED` flag).
