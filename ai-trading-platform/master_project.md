# MASTER PROJECT SPECIFICATION

# Autonomous Multi-Symbol Binance Futures AI Trading & World-Model Trading System

You are the primary senior AI/ML engineer, quantitative researcher, backend engineer, data engineer, MLOps engineer, and software architect for this project.

Your job is to transform the existing repository into a production-grade, research-grade autonomous Binance Futures AI trading platform.

Do NOT treat this as a simple trading bot.

The long-term goal is an AI trading system that:

* trades multiple Binance Futures symbols
* learns from historical and live experience
* understands market state
* predicts possible future market states
* evaluates possible actions
* learns from its own experience
* continuously improves through controlled training
* uses a world-model-style architecture for market simulation and planning
* has strict external risk controls
* never bypasses risk controls
* never blindly trusts its own predictions
* never directly overwrites the production model with an unvalidated model
* can eventually operate continuously on a VPS
* remains measurable, reproducible, auditable, and fail-safe

IMPORTANT:

Do not jump directly to the final world model.

Build the system progressively.

Every phase must create the foundation required by the next phase.

==================================================

1. FINAL OBJECTIVE
   ==================================================

The final system should look conceptually like:

MARKET DATA
↓
DATA QUALITY / TIME ALIGNMENT
↓
FEATURE / REPRESENTATION ENGINE
↓
MARKET STATE ENCODER
↓
TEMPORAL MEMORY
↓
WORLD MODEL
↓
PREDICT POSSIBLE FUTURES
↓
PLANNER / DECISION ENGINE
↓
AI POLICY
↓
RISK FIREWALL
↓
EXECUTION ENGINE
↓
BINANCE FUTURES
↓
NEW MARKET STATE
↓
EXPERIENCE
↓
MYSQL / REPLAY DATASET
↓
TRAINING
↓
EVALUATION
↓
FORWARD TEST
↓
MODEL PROMOTION
↓
PRODUCTION

The AI should eventually be able to answer:

"What is the current market state?"

"What could happen next?"

"What happens if I go LONG?"

"What happens if I go SHORT?"

"What happens if I HOLD?"

"What is the expected risk/reward of each action?"

"What is the probability that the current prediction is wrong?"

"What position size is appropriate?"

"Should I trade at all?"

==================================================
2. FINAL TRADING TARGET
=======================

Target exchange:

Binance Futures

Target market:

USDT-M Futures

The architecture must be Futures-first.

Do NOT build a Spot architecture and later try to convert it into Futures.

The system must understand:

* LONG
* SHORT
* HOLD
* CLOSE
* leverage
* margin
* maintenance margin
* liquidation
* funding
* maker fees
* taker fees
* spread
* slippage
* order-book liquidity
* position exposure
* portfolio exposure
* correlated exposure

Initial symbols:

BTCUSDT
ETHUSDT
SOLUSDT
BNBUSDT
XRPUSDT

Architecture must support expansion to:

DOGEUSDT
LINKUSDT
AVAXUSDT
ADAUSDT
SUIUSDT
AAVEUSDT

and additional symbols without rewriting the model architecture.

The final system must be MULTI-SYMBOL.

Do NOT create a completely independent AI brain for every coin.

The preferred architecture is:

```
            SHARED AI
                │
   ┌────────────┼────────────┐
   ↓            ↓            ↓
 BTC          ETH          SOL
   ↓            ↓            ↓
```

symbol info   symbol info   symbol info

The model should learn common market behavior while still understanding symbol identity and differences.

==================================================
3. CURRENT PROJECT CONTEXT
==========================

The repository already contains a partially implemented AI trading platform.

Existing areas include:

apps/
core/
models/
tests/
scripts/
docs/

There are already components for:

* Binance Futures connectivity
* market collection
* feature generation
* risk management
* model registry
* actor-critic models
* Futures simulation
* online training
* replay buffer
* MySQL
* Redis

However, existing implementations may be prototypes, incomplete, duplicated, inconsistent, unsafe, or architecturally weak.

You are allowed and expected to:

* refactor
* replace
* consolidate
* redesign
* remove obsolete implementations
* fix incorrect abstractions
* add missing modules
* add tests
* change schemas
* change model architecture

Do NOT preserve bad architecture merely because it already exists.

Correctness is more important than backward compatibility with prototype code.

==================================================
4. DEVELOPMENT PHILOSOPHY
=========================

Build this as a serious ML system.

Non-negotiable principles:

1. No data leakage.
2. No future information during training or inference.
3. No random production-model fallback.
4. No unvalidated model promotion.
5. No uncontrolled online learning.
6. No direct AI bypass of risk controls.
7. No live trading during development.
8. No hard-coded fake production account state.
9. No fake/random market/news data pretending to be real.
10. No meaningless training metrics.
11. No model promotion based only on profit.
12. Every important subsystem must be testable.
13. Every experiment must be reproducible.
14. Every production model must have version metadata.
15. Every trade decision must be explainable/auditable through stored state and decision metadata.

==================================================
5. COMPLETE DEVELOPMENT PHASES
==============================

Implement the project through these phases.

Do NOT skip phases.

==================================================
PHASE 1 — FOUNDATION AND SAFETY
===============================

Create a reliable application foundation.

Implement:

* configuration
* environment handling
* structured logging
* MySQL
* Redis
* database migrations
* application lifecycle
* dependency management
* test framework
* deterministic random seeds
* model registry foundation
* checkpoint management
* health checks
* service monitoring

Create clear configuration for:

TRADING_ENABLED
DRY_RUN
ALLOW_TESTNET
ALLOW_LIVE_TRADING
EMERGENCY_STOP

Default configuration MUST be safe.

Default:

TRADING_ENABLED=false
DRY_RUN=true
ALLOW_TESTNET=false
ALLOW_LIVE_TRADING=false
EMERGENCY_STOP=true

A production process must never accidentally start real trading.

==================================================
PHASE 2 — BINANCE FUTURES DATA AND EXCHANGE INFRASTRUCTURE
==========================================================

Build a reliable Binance Futures adapter.

Implement:

* exchangeInfo
* symbol metadata
* tick size
* quantity step
* min quantity
* max quantity
* min notional
* price precision
* quantity precision
* leverage information
* account information
* wallet balance
* available balance
* positions
* open orders
* order status
* create order
* cancel order
* modify/cancel-replace where supported
* user data stream
* websocket reconnect
* heartbeat
* retry
* timeout
* rate-limit handling

Separate:

MARKET DATA

from:

TRADING EXECUTION

from:

ACCOUNT STATE

from:

RISK

Never mix these responsibilities.

Exchange metadata must be retrieved from Binance rather than using fake hard-coded precision values.

Quantity calculations must correctly use Decimal/flooring against exchange step sizes.

Never use Python float rounding when exchange precision requires exact step handling.

==================================================
PHASE 3 — MARKET DATA ENGINE
============================

Collect historical and live data.

Minimum data:

OHLCV

* open
* high
* low
* close
* volume
* quote volume
* trades

Market microstructure:

* bid
* ask
* spread
* mid price
* bid volume
* ask volume
* order-book depth
* order-book imbalance
* trade flow
* buy volume
* sell volume

Derivatives:

* funding rate
* open interest
* liquidations
* mark price
* index price
* basis where available

Portfolio:

* wallet balance
* equity
* margin
* current positions
* unrealized PnL
* realized PnL
* exposure
* leverage

All observations must have precise timestamps.

Use event time whenever available.

Do NOT use arrival time as a replacement for event time.

==================================================
PHASE 4 — DATA QUALITY AND TIME ALIGNMENT
=========================================

Build a dedicated data-quality layer.

Check:

* stale data
* missing data
* duplicate events
* out-of-order events
* impossible prices
* negative volume
* invalid spread
* crossed book
* NaN
* Inf
* missing symbols
* timestamp problems
* websocket gaps
* exchange disconnects

The AI must NOT trade using invalid or stale data.

Implement strict time alignment.

At decision time T:

ONLY information with:

information_timestamp <= T

may be used.

Create automated leakage checks.

The system should be able to detect accidental future leakage during training.

==================================================
PHASE 5 — FEATURE AND MARKET REPRESENTATION ENGINE
==================================================

Build a proper feature pipeline.

Do NOT depend forever on a giant manually concatenated vector.

Initially support:

Returns:

* 1-step return
* 5-step return
* 15-step return
* 1m return
* 5m return
* 15m return
* 1h return

Trend:

* moving averages
* EMA relationships
* trend strength

Momentum:

* RSI
* rate of change
* momentum

Volatility:

* realized volatility
* ATR
* volatility regime

Volume:

* volume change
* volume anomaly
* buy/sell imbalance

Microstructure:

* spread
* order-book imbalance
* depth
* liquidity

Derivatives:

* funding
* funding change
* open interest
* OI change
* liquidation pressure

Portfolio state:

* current position
* entry price
* unrealized PnL
* exposure
* available margin
* leverage

Time:

* hour
* day
* session
* periodic encodings

Important:

Features must be calculated using only information available at that timestamp.

Create a canonical feature schema.

==================================================
PHASE 6 — BASELINE STRATEGIES
=============================

Implement traditional strategies as baselines.

These are NOT the final intelligence.

Implement:

* trend following
* momentum
* mean reversion
* breakout
* volatility-based strategy

Use them for:

* benchmarking
* feature generation
* debugging
* simulator validation
* evaluating whether AI actually adds value

Compare AI against these baselines.

Never claim AI success simply because it makes money.

==================================================
PHASE 7 — REALISTIC BINANCE FUTURES SIMULATOR
=============================================

Build a serious Futures simulator.

It must support:

LONG

SHORT

HOLD

CLOSE

Include:

* leverage
* initial margin
* maintenance margin
* liquidation
* funding payments
* maker fees
* taker fees
* spread
* slippage
* latency
* partial fills
* rejected orders
* exchange filters
* quantity step
* tick size
* minimum notional
* available liquidity
* mark price
* index price

Implement realistic position accounting.

The simulator must NOT allow impossible fills.

Use deterministic seeds.

The same seed + same dataset + same configuration should reproduce the same result.

Create tests for:

* opening long
* opening short
* closing long
* closing short
* profit
* loss
* leverage
* funding
* fees
* slippage
* liquidation
* insufficient margin
* invalid order
* partial fill
* exchange filters

==================================================
PHASE 8 — EXPERIENCE SYSTEM
===========================

Build one canonical experience system.

Remove duplicate replay-buffer implementations.

Experience must contain enough information to reproduce a decision.

Conceptually:

experience:

timestamp
symbol
environment
observation
latent_state if available
action
requested_quantity
executed_quantity
execution_price
reward
next_observation
done
portfolio_state
market_state
model_version
policy_version
risk_decision
execution_result
data_quality
metadata

Store durable experience in MySQL.

Use Redis for fast/recent state where appropriate.

MySQL is the durable source.

Replay/training cache may be derived from MySQL.

Experience must be queryable by:

* symbol
* time
* model version
* episode
* regime
* action
* reward
* event
* training dataset version

==================================================
PHASE 9 — FIRST AI PREDICTION MODEL
===================================

Before building the world model, build strong prediction models.

Predict:

* future return
* direction probability
* volatility
* probability of adverse movement
* probability of reaching target/stop regions

Use time-series-safe datasets.

Do NOT randomly shuffle temporal data across train/test.

Use:

TRAIN

↓

VALIDATION

↓

TEST

with chronological separation.

Also implement walk-forward evaluation.

Metrics:

* MAE
* RMSE
* directional accuracy
* precision
* recall
* calibration
* Brier score
* performance by regime
* performance by symbol

Prediction accuracy alone is not sufficient.

Measure trading usefulness.

==================================================
PHASE 10 — TEMPORAL MODEL
=========================

Upgrade from single-timestep features to sequences.

Input:

X[t-N : t]

Output:

market state / prediction

Experiment with:

* GRU
* LSTM
* Transformer
* temporal attention

The architecture should be modular so different models can be compared.

The model must understand:

* momentum
* trend
* volatility
* regime changes
* market acceleration
* reversals
* cross-symbol relationships

==================================================
PHASE 11 — SHARED MULTI-SYMBOL MODEL
====================================

Build a proper multi-asset architecture.

Do NOT simply flatten all symbols into one giant vector permanently.

Preferred architecture:

For each symbol:

market features
↓
symbol encoder
↓
symbol latent representation

Then:

BTC latent
ETH latent
SOL latent
BNB latent
XRP latent
↓
cross-asset attention / aggregation
↓
portfolio-level state

Include a learned symbol embedding.

The model should understand:

BTCUSDT ≠ ETHUSDT

but also learn common market relationships.

The architecture must support adding/removing symbols through configuration.

==================================================
PHASE 12 — DECISION / POLICY MODEL
==================================

Create a policy model.

Initial actions:

LONG
SHORT
HOLD
CLOSE

The policy should produce:

action probabilities
confidence
expected outcome estimates
risk-aware decision information

Do not allow the policy itself to bypass the risk engine.

Architecture:

STATE
↓
POLICY
↓
PROPOSED ACTION
↓
RISK ENGINE
↓
APPROVED / REJECTED / MODIFIED

The AI proposes.

The risk engine decides whether execution is allowed.

==================================================
PHASE 13 — REWARD ENGINE
========================

Create a proper trading reward.

Do NOT train only on raw profit.

Reward should consider:

net PnL

* trading fees
* funding cost
* slippage
* risk penalty
* drawdown penalty
* excessive turnover penalty
* liquidation penalty

Conceptually:

reward =
net_profit

* fees
* funding
* slippage
* risk_penalty
* drawdown_penalty
* turnover_penalty
* liquidation_penalty

Make reward components separately observable.

This is important for debugging.

==================================================
PHASE 14 — REINFORCEMENT LEARNING
=================================

Implement RL training using the realistic Futures simulator.

Experiment with appropriate algorithms such as:

* PPO
* SAC
* TD3
* actor-critic variants

Do NOT assume one algorithm is automatically best.

Use controlled experiments.

Every experiment must record:

* dataset
* symbols
* seed
* model architecture
* hyperparameters
* training period
* validation period
* simulator configuration
* reward configuration
* metrics
* checkpoint

==================================================
PHASE 15 — WORLD MODEL
======================

Only after the prediction, sequence, representation, and policy systems work should the true world model be developed.

The world model should learn:

P(next_state | current_state, action)

and ideally:

P(next_state, reward | current_state, action)

Architecture concept:

OBSERVATION
↓
ENCODER
↓
LATENT STATE Z_t
↓
WORLD MODEL
+
ACTION
↓
PREDICTED LATENT Z_t+1
↓
PREDICTED REWARD
↓
PREDICTED MARKET STATE

Possible architecture families can include:

* latent dynamics model
* recurrent state-space model
* transformer dynamics model
* latent state-space model
* ensemble dynamics models

Do not blindly copy a famous architecture.

Choose based on measured performance.

==================================================
PHASE 16 — WORLD MODEL VALIDATION
=================================

Do NOT call it a world model just because training loss decreases.

Validate whether it can actually predict useful futures.

Test:

1-step prediction

5-step prediction

15-step prediction

30-step prediction

longer horizon prediction

Measure:

* latent prediction error
* price prediction error
* volatility prediction error
* direction accuracy
* reward prediction
* regime prediction
* calibration
* multi-step degradation

Most importantly:

Does using the world model improve decision-making?

Compare:

Policy WITHOUT world model

vs

Policy WITH world model

If the world model does not improve useful decision metrics, do not promote it as the main architecture.

==================================================
PHASE 17 — IMAGINATION / PLANNING
=================================

Once the world model is reliable, allow the AI to simulate possible futures.

For each current state:

Current State

├── LONG
│     ↓
│   imagined future
│
├── SHORT
│     ↓
│   imagined future
│
├── HOLD
│     ↓
│   imagined future
│
└── CLOSE
↓
imagined future

Evaluate:

* expected return
* downside
* drawdown
* probability of adverse move
* fees
* funding
* slippage
* uncertainty

Then choose the best risk-adjusted action.

Conceptually:

CURRENT STATE
↓
WORLD MODEL
↓
IMAGINE FUTURE A
IMAGINE FUTURE B
IMAGINE FUTURE C
↓
EVALUATE
↓
POLICY / PLANNER
↓
RISK ENGINE
↓
ACTION

==================================================
PHASE 18 — UNCERTAINTY
======================

The system must know when it does not know.

Implement uncertainty estimates.

Possible approaches:

* ensembles
* dropout uncertainty
* probabilistic prediction
* distributional prediction
* confidence calibration

The AI should be able to produce something conceptually like:

LONG:
expected return = +0.8%
risk = medium
confidence = 0.72

SHORT:
expected return = -0.3%
risk = medium
confidence = 0.61

HOLD:
expected return = +0.1%
risk = low
confidence = 0.83

The final decision must incorporate uncertainty.

Low confidence should often result in HOLD.

==================================================
PHASE 19 — EVENT / NEWS / MACRO INTELLIGENCE
============================================

Add external information.

Sources can include:

* crypto news
* financial news
* economic calendar
* CPI
* PCE
* FOMC
* Fed decisions
* employment data
* GDP
* PMI
* interest rates
* geopolitical events
* regulation
* crypto-specific events
* token unlocks
* upgrades
* staking changes
* exchange events
* major liquidations

Store:

event time
source
headline
body
event type
entities
sentiment
importance
confidence

Then measure market response:

+5 seconds
+30 seconds
+1 minute
+5 minutes
+15 minutes
+30 minutes
+1 hour
+4 hours
+24 hours

Measure:

* return
* volume
* volatility
* spread
* order-book imbalance
* funding
* OI
* liquidation activity
* regime

Do NOT simply tell the AI:

"similar event previously caused price to rise."

The model must consider the current market state.

==================================================
PHASE 20 — CROSS-MARKET CONTEXT
===============================

Add relevant external market information:

* S&P 500
* Nasdaq
* VIX
* DXY
* gold
* oil
* Treasury yields where useful

The model should learn relationships rather than relying on hard-coded rules.

Example:

crypto risk-off state

may correlate with:

equity weakness
+
VIX increase
+
DXY increase

But these relationships must be learned and validated rather than assumed permanently.

==================================================
PHASE 21 — PORTFOLIO INTELLIGENCE
=================================

The model must eventually reason at portfolio level.

Example:

BTC LONG
ETH LONG
SOL LONG

may represent one large correlated risk exposure.

The system must calculate:

* symbol exposure
* total exposure
* directional exposure
* correlated exposure
* leverage
* margin utilization
* liquidation distance
* portfolio drawdown
* concentration

AI should not independently optimize each symbol while ignoring the portfolio.

The portfolio is the real decision unit.

==================================================
PHASE 22 — RISK FIREWALL
========================

Create a completely independent risk layer.

The AI cannot override it.

Enforce:

* maximum position size
* maximum order size
* maximum leverage
* maximum portfolio exposure
* maximum symbol exposure
* maximum correlated exposure
* maximum open positions
* maximum daily loss
* maximum drawdown
* maximum trade frequency
* cooldown
* maximum slippage
* stale data protection
* invalid data protection
* insufficient margin
* liquidation risk
* emergency stop
* manual kill switch

If AI says:

"BUY 100 BTC"

but risk says:

"maximum allowed = 0.01 BTC"

the result is:

REJECT or safely clamp according to the explicit risk policy.

The AI must never bypass the risk layer.

==================================================
PHASE 23 — EXECUTION ENGINE
===========================

Only after the simulator and risk engine are reliable should real execution be developed.

Execution engine must support:

* market order
* limit order
* reduce-only
* close position
* order status
* cancellation
* partial fills
* retry
* timeout
* websocket updates
* reconciliation

Never assume an order succeeded merely because the API request returned successfully.

Always reconcile actual exchange state.

Maintain:

local state

vs

exchange state

and detect divergence.

==================================================
PHASE 24 — PAPER TRADING / FORWARD TESTING
==========================================

Before Testnet:

run the complete system against live market data but without real execution.

Record:

* every observation
* every prediction
* every decision
* every risk decision
* every hypothetical order
* every hypothetical fill
* every reward
* every model version

This must run continuously.

Evaluate against:

* buy and hold
* baseline strategies
* previous model
* no-trade baseline

==================================================
PHASE 25 — BINANCE FUTURES TESTNET
==================================

After successful forward testing, connect to Binance Futures Testnet.

Still keep:

EMERGENCY_STOP=true

unless explicitly approved.

Use small controlled exposure.

Test:

* account synchronization
* order placement
* cancellation
* fills
* partial fills
* position updates
* funding
* leverage
* margin
* reconnection
* restart recovery
* state reconciliation

==================================================
PHASE 26 — PRODUCTION MODEL REGISTRY
====================================

Implement strict model lifecycle:

RESEARCH

↓

CANDIDATE

↓

VALIDATED

↓

FORWARD_TESTING

↓

APPROVED

↓

PRODUCTION

↓

RETIRED

A model must contain metadata:

model_id
version
architecture
training_dataset
training_period
validation_period
test_period
symbols
features
hyperparameters
metrics
checkpoint_path
checkpoint_hash
created_at
git_commit
random_seed
simulator_version
reward_version

Production must load a known production version.

Never randomly initialize a production model.

If the production model cannot be loaded:

FAIL CLOSED.

Do not silently start with random weights.

==================================================
PHASE 27 — AUTOMATED MODEL EVALUATION
=====================================

Every candidate must pass automated evaluation.

Metrics:

Net return
Sharpe
Sortino
Maximum drawdown
Profit factor
Win rate
Average trade
Trade count
Fees
Funding
Slippage
Turnover
Loss streak
Exposure
Liquidation count

Also evaluate by:

* symbol
* market regime
* month
* volatility regime
* trend regime
* event periods

Compare against:

* buy and hold
* baseline strategies
* previous production model

Do not promote if the model only performs well in one narrow period.

==================================================
PHASE 28 — WALK-FORWARD VALIDATION
==================================

Use rolling time windows.

Example:

TRAIN:
January → June

VALIDATE:
July

TEST:
August

Then move forward:

TRAIN:
February → July

VALIDATE:
August

TEST:
September

Continue.

This is much more representative of real trading.

Never use future test information to tune the model.

==================================================
PHASE 29 — CONTINUOUS LEARNING
==============================

The production system continuously collects experience.

Architecture:

LIVE MARKET
↓
PRODUCTION MODEL
↓
DECISION
↓
RISK
↓
EXECUTION
↓
EXPERIENCE
↓
MYSQL
↓
DATASET BUILDER
↓
CANDIDATE TRAINING
↓
EVALUATION
↓
WALK-FORWARD
↓
FORWARD TEST
↓
PROMOTION

Training must NEVER automatically replace production simply because new training finished.

==================================================
PHASE 30 — CATASTROPHIC FORGETTING PROTECTION
=============================================

The model should learn new market behavior without destroying previously learned useful behavior.

Maintain:

* historical replay
* recent replay
* rare-event replay
* regime-balanced replay
* symbol-balanced replay

Training dataset should contain a controlled mixture of:

recent experiences

*

historical experiences

*

rare market events

*

important failures

*

different regimes

Do not train only on the latest data.

==================================================
PHASE 31 — EXPERIENCE SAMPLING
==============================

Build intelligent replay sampling.

Possible categories:

RECENT
HISTORICAL
RARE EVENTS
HIGH LOSS
HIGH REWARD
REGIME CHANGE
LIQUIDATION
EXTREME VOLATILITY
NORMAL MARKET

Avoid letting the dataset become dominated by one market regime.

==================================================
PHASE 32 — SELF-IMPROVING TRADING LOOP
======================================

The final learning loop should be:

OBSERVE

↓

UNDERSTAND CURRENT MARKET STATE

↓

PREDICT

↓

WORLD MODEL

↓

IMAGINE POSSIBLE FUTURES

↓

EVALUATE ACTIONS

↓

POLICY

↓

RISK FIREWALL

↓

EXECUTE

↓

OBSERVE RESULT

↓

STORE EXPERIENCE

↓

LEARN

↓

EVALUATE

↓

PROMOTE IF BETTER

This is the final autonomous learning architecture.

==================================================
33. FINAL MODEL ARCHITECTURE
============================

The preferred long-term architecture is:

```
                MARKET DATA
                     ↓
            DATA QUALITY LAYER
                     ↓
            FEATURE ENGINE
                     ↓
         SYMBOL-SPECIFIC ENCODERS
                     ↓
            SYMBOL EMBEDDINGS
                     ↓
          CROSS-ASSET ATTENTION
                     ↓
             TEMPORAL MEMORY
                     ↓
              LATENT STATE
                     ↓
              WORLD MODEL
                     ↓
         ┌───────────┴───────────┐
         ↓                       ↓
   FUTURE PREDICTION       UNCERTAINTY
         │                       │
         └───────────┬───────────┘
                     ↓
               PLANNER
                     ↓
                 POLICY
                     ↓
             ACTION PROPOSAL
                     ↓
              RISK FIREWALL
                     ↓
                EXECUTION
                     ↓
             BINANCE FUTURES
                     ↓
                EXPERIENCE
                     ↓
               MYSQL/REPLAY
                     ↓
                TRAINING
                     ↓
               EVALUATION
                     ↓
                PROMOTION
                     ↓
                PRODUCTION
```

==================================================
34. DATABASE ARCHITECTURE
=========================

Use MySQL as durable storage.

At minimum create structured tables for:

symbols

market_ticks

candles

orderbook_snapshots

trades

funding_rates

open_interest

liquidations

market_states

portfolio_states

positions

orders

fills

experiences

episodes

events

event_responses

features

training_datasets

training_runs

evaluation_runs

models

model_versions

model_metrics

forward_tests

risk_decisions

system_events

audit_logs

The exact schema is up to you, but it must be normalized enough to remain maintainable and indexed for time-series access.

==================================================
35. REDIS
=========

Redis is for fast operational state.

Use it for things such as:

latest market state
latest orderbook state
latest portfolio state
latest position state
locks
cooldowns
temporary queues
service heartbeats

Do NOT treat Redis as the permanent historical source.

MySQL remains durable storage.

==================================================
36. OBSERVABILITY
=================

Build a monitoring system.

Monitor:

market-data health

websocket health

exchange connectivity

database connectivity

Redis connectivity

model inference latency

data freshness

CPU

RAM

GPU

disk

order latency

execution errors

risk rejections

model version

prediction confidence

PnL

drawdown

exposure

position count

training status

candidate model status

forward-test status

==================================================
37. DASHBOARD
=============

Build a local dashboard.

Show:

current market

symbols

positions

portfolio

PnL

drawdown

exposure

leverage

orders

fills

AI decisions

confidence

risk decisions

model version

prediction

world-model predictions

training status

candidate models

evaluation results

system health

The dashboard must make it possible to understand WHY the AI made a decision.

==================================================
38. FAIL-SAFE REQUIREMENTS
==========================

If any of these occur:

database unavailable

Redis unavailable

market data stale

invalid market data

exchange unavailable

model unavailable

model corruption

risk engine unavailable

portfolio state unknown

position reconciliation failure

unexpected exception

time synchronization problem

then:

DO NOT OPEN NEW POSITIONS.

Prefer:

HOLD

or

CLOSE/REDUCE ONLY when safely required and explicitly supported.

The system must fail closed.

==================================================
39. SECURITY
============

Never commit:

API keys

API secrets

database passwords

private credentials

production secrets

Use environment variables or secure secret storage.

Add secret scanning where practical.

Never log API secrets.

==================================================
40. TESTING
===========

Create tests for every critical subsystem.

Unit tests:

* feature calculations
* timestamp handling
* leakage detection
* symbol metadata
* quantity formatting
* risk limits
* reward calculation
* position accounting
* simulator
* model registry

Integration tests:

* MySQL
* Redis
* exchange adapter
* market collector
* execution engine
* risk engine
* model loading
* experience storage

Failure tests:

* stale data
* missing data
* invalid price
* invalid quantity
* NaN
* Inf
* exchange disconnect
* database failure
* Redis failure
* model failure
* order rejection
* partial fill
* liquidation
* insufficient margin

Regression tests:

Every discovered bug must receive a regression test.

==================================================
41. REPRODUCIBILITY
===================

Every training run must record:

git commit

random seed

dataset version

dataset time range

symbols

features

model architecture

hyperparameters

simulator version

reward version

environment configuration

hardware information where useful

checkpoint hash

The same configuration should be reproducible.

==================================================
42. NO DATA LEAKAGE
===================

This is one of the highest-priority requirements.

At time T:

ONLY data known at or before T may be used.

Be especially careful with:

* candle construction
* normalization
* rolling indicators
* train/test split
* event data
* news timestamps
* funding data
* open interest
* orderbook snapshots
* future fills
* target generation
* feature scaling

Fit scalers only on training data where appropriate.

Never normalize the entire dataset before chronological splitting.

Build automated leakage tests.

==================================================
43. MODEL SELECTION
===================

Do not assume:

"more complex = better."

Start simple.

Benchmark:

baseline

↓

ML prediction

↓

sequence model

↓

policy

↓

RL

↓

world model

↓

world-model planning

Only move to the next complexity level if it demonstrates measurable improvement.

==================================================
44. TRADING FREQUENCY
=====================

The AI observes continuously.

It does NOT need to trade continuously.

The policy must be allowed to choose:

HOLD

when no good opportunity exists.

A good model may trade less frequently if that improves risk-adjusted returns.

Do not optimize for trade count.

==================================================
45. POSITION SIZING
===================

Initially keep position sizing conservative and externally controlled.

Later the AI can estimate:

desired position

or

desired risk allocation.

But the final size must pass:

risk limits

margin limits

leverage limits

liquidity limits

portfolio exposure limits

correlation limits

exchange filters

==================================================
46. REGIME DETECTION
====================

Add market-regime representation.

Possible regimes:

trending

ranging

high volatility

low volatility

risk-on

risk-off

breakout

crash

recovery

Do not assume fixed regime labels are always correct.

Compare:

rule-based regime detection

vs

learned regime representation.

==================================================
47. RARE EVENT LEARNING
=======================

Pay special attention to:

flash crashes

large liquidations

extreme funding

sudden volatility

exchange outages

large market moves

major economic announcements

unexpected geopolitical events

Rare events must not be discarded merely because they are statistically uncommon.

Create a rare-event replay mechanism.

==================================================
48. FINAL DEPLOYMENT ARCHITECTURE
=================================

The final VPS deployment should resemble:

```
                ┌──────────────┐
                │ Binance      │
                │ Futures      │
                └──────┬───────┘
                       │
                WebSocket/REST
                       │
          ┌────────────┴────────────┐
          │                         │
    Market Collector          Exchange Adapter
          │                         │
          └────────────┬────────────┘
                       ↓
                State/Feature Engine
                       ↓
                AI Inference Service
                       ↓
              World Model / Planner
                       ↓
                  Policy
                       ↓
                Risk Firewall
                       ↓
                Execution Engine
                       ↓
                   Binance
```

Parallel:

Experience → MySQL

Fast State → Redis

Training → GPU/Training Worker

Evaluation → Evaluation Worker

Dashboard API → Frontend

Monitoring → Monitoring System

==================================================
49. PRODUCTION SAFETY
=====================

There must be explicit stages:

OFFLINE

↓

SIMULATION

↓

PAPER TRADING

↓

FORWARD TEST

↓

BINANCE TESTNET

↓

CONTROLLED LIVE

Never automatically jump between these stages.

Live trading must require explicit configuration.

Production default must remain disabled.

==================================================
50. PROJECT DIRECTORY
=====================

Use a clean architecture similar to:

ai-trading-platform/

```
apps/

    market_collector/
    orderbook/
    derivatives_collector/
    news_collector/
    macro_collector/
    sentiment_collector/
    crypto_data/

    feature_engine/

    experience/

    simulator/

    trainer/

    evaluator/

    world_model/

    policy/

    risk/

    portfolio_risk/

    executor/

    scheduler/

    monitor/

    dashboard_api/

    dashboard_frontend/

core/

    config/
    database/
    logging/
    events/
    schemas/
    time/
    metrics/
    security/
    exchange/

models/

    architectures/
    preprocessing/
    checkpoints/
    registry/
    production/
    candidates/
    rejected/

strategies/

    baseline/
        trend_following/
        momentum/
        mean_reversion/
        breakout/
        volatility/

    ai/

data/

    migrations/
    seed/
    fixtures/

tests/

    unit/
    integration/
    simulation/
    exchange/
    model/
    risk/
    portfolio/
    data_quality/
    regression/

scripts/

    backfill/
    training/
    evaluation/
    maintenance/

configs/

    symbols/
    risk/
    training/
    models/

docs/

notebooks/

docker/

docker-compose.yml

.env.example

README.md
```

==================================================
51. CODE QUALITY
================

Use:

Python 3.12+

FastAPI where APIs are needed

Pydantic

SQLAlchemy

Alembic

MySQL 8+

Redis

PyTorch

NumPy

Pandas or Polars

asyncio where appropriate

Docker

Linux-compatible deployment

Use type hints.

Use clear interfaces.

Avoid giant files.

Avoid duplicated models/schemas.

Avoid circular dependencies.

Use dependency injection where useful.

==================================================
52. IMPORTANT EXISTING CODE RULE
================================

The existing repository contains prototype implementations.

Audit them carefully.

Do not assume existing code is correct.

Specifically check for:

* duplicated ReplayBuffer
* fake/random data
* fake account state
* hard-coded observation dimensions
* incorrect feature indexing
* model dimension mismatch
* symbol-universe mismatch
* unsafe random model fallback
* incomplete exchange order lifecycle
* incorrect quantity rounding
* incomplete risk enforcement
* stale-data handling
* simulator realism
* model registry validation
* production model loading
* online trainer safety
* README/documentation inconsistencies

Fix root causes rather than patching symptoms.

==================================================
53. ONLINE TRAINING SAFETY
==========================

Online training must NEVER automatically replace the production model.

Correct:

Production Model

*

New Experience

↓

Candidate Training

↓

Candidate Model

↓

Evaluation

↓

Forward Testing

↓

Promotion

Incorrect:

Production Model

↓

modify weights continuously

↓

hope it improves

Never implement the second pattern.

==================================================
54. FINAL SUCCESS CRITERIA
==========================

The project is complete only when the system can:

1. Collect reliable Binance Futures data.

2. Maintain synchronized multi-symbol state.

3. Store durable experience.

4. Train reproducibly.

5. Predict future market behavior.

6. Understand temporal market structure.

7. Learn shared representations across multiple symbols.

8. Make LONG/SHORT/HOLD/CLOSE decisions.

9. Estimate uncertainty.

10. Respect external risk controls.

11. Simulate Futures realistically.

12. Compare against strong baselines.

13. Perform walk-forward validation.

14. Build and validate a useful world model.

15. Use the world model to imagine possible futures.

16. Use those imagined futures for risk-aware planning.

17. Continuously collect new experience.

18. Train candidate models.

19. Automatically evaluate candidates.

20. Protect against catastrophic forgetting.

21. Promote only validated models.

22. Recover safely after service failures.

23. Reconcile local state with Binance.

24. Run continuously on a VPS.

25. Provide a useful monitoring dashboard.

26. Remain disabled from real trading unless explicitly authorized.

==================================================
55. MOST IMPORTANT DEVELOPMENT ORDER
====================================

Follow this order:

1. Foundation
2. Safety
3. Binance Futures infrastructure
4. Market data
5. Data quality
6. Feature engine
7. Baseline strategies
8. Realistic simulator
9. Experience/MySQL system
10. Prediction models
11. Temporal models
12. Multi-symbol shared model
13. Policy
14. Reward system
15. Reinforcement learning
16. World model
17. World-model validation
18. Imagination/planning
19. Uncertainty
20. News/macro/events
21. Cross-market information
22. Portfolio intelligence
23. Risk firewall hardening
24. Execution engine
25. Paper trading
26. Forward testing
27. Binance Testnet
28. Model registry/promotion
29. Continuous learning
30. VPS production architecture
31. Controlled live deployment

==================================================
56. DEVELOPMENT RULE
====================

Do NOT implement the entire project blindly in one pass.

Work phase-by-phase.

For each phase:

1. Inspect the current repository.
2. Identify existing reusable code.
3. Identify incorrect/unsafe code.
4. Refactor where required.
5. Implement the phase.
6. Write tests.
7. Run tests.
8. Fix failures.
9. Perform integration checks.
10. Document the completed work.
11. Report:

* what changed
* files changed
* tests executed
* test results
* remaining risks
* next phase

Do not claim a phase is complete without evidence.

==================================================
57. FINAL RULE
==============

The goal is NOT to build a bot that simply makes trades.

The goal is to build a system that progressively learns:

MARKET STATE
↓
MARKET DYNAMICS
↓
PREDICTION
↓
DECISION
↓
RISK
↓
EXECUTION
↓
OUTCOME
↓
EXPERIENCE
↓
LEARNING
↓
BETTER MARKET MODEL
↓
BETTER PLANNING
↓
BETTER DECISIONS

The final architecture should therefore evolve from:

RULES

→ ML

→ SEQUENCE MODEL

→ POLICY

→ RL

→ WORLD MODEL

→ WORLD-MODEL PLANNING

→ CONTINUOUS LEARNING

while maintaining strict risk controls throughout.

Do not optimize for complexity.

Optimize for:

CORRECTNESS
RELIABILITY
GENERALIZATION
RISK-ADJUSTED PERFORMANCE
DATA QUALITY
REPRODUCIBILITY
MEASURABILITY
SAFETY

Start by auditing the current repository against this entire specification.

Then implement the project in the defined order.

Do not skip foundational work merely to reach the world-model stage faster.
