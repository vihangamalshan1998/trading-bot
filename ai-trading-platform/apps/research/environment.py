import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List

class FuturesTradingEnv(gym.Env):
    """
    A rigorous simulated futures trading environment supporting leverage, maintenance margin,
    liquidations, funding rates, and multi-directional (LONG/SHORT) trading.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 historical_data: List[Dict[str, Any]], 
                 initial_balance: float = 1000.0, 
                 leverage: int = 10,
                 maker_fee: float = 0.0002,  # 0.02%
                 taker_fee: float = 0.0004,  # 0.04%
                 maintenance_margin_rate: float = 0.004, # 0.4%
                 funding_rate: float = 0.0001, # per 8 hours
                 steps_per_funding: int = 28800 # assuming 1 step = 1 sec, 8 hours = 28800 secs
                 ):
        super(FuturesTradingEnv, self).__init__()
        
        self.data = historical_data
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.mmr = maintenance_margin_rate
        self.funding_rate = funding_rate
        self.steps_per_funding = steps_per_funding
        
        # State: wallet_balance, margin_balance, position_size, entry_price, 
        # unrealized_pnl, spread_bps, mid_price, imbalance, vwap_recent
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        
        # Actions: 
        # 0: HOLD
        # 1: OPEN_LONG (allocates available margin with leverage)
        # 2: OPEN_SHORT
        # 3: CLOSE_LONG
        # 4: CLOSE_SHORT
        self.action_space = spaces.Discrete(5)
        
        # Internal state
        self.current_step = 0
        self.wallet_balance = self.initial_balance
        self.position_size = 0.0 # positive for LONG, negative for SHORT
        self.entry_price = 0.0
        
    def _get_obs(self):
        current_data = self.data[self.current_step]
        mid_price = current_data.get("mid_price", 0.0)
        
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        if self.position_size != 0:
            if self.position_size > 0: # LONG
                unrealized_pnl = (mid_price - self.entry_price) * self.position_size
            else: # SHORT
                unrealized_pnl = (self.entry_price - mid_price) * abs(self.position_size)
                
        margin_balance = self.wallet_balance + unrealized_pnl
        
        obs = np.array([
            self.wallet_balance,
            margin_balance,
            self.position_size,
            self.entry_price,
            unrealized_pnl,
            current_data.get("spread_bps", 0.0),
            mid_price,
            current_data.get("imbalance", 0.0),
            current_data.get("vwap_recent", 0.0)
        ], dtype=np.float32)
        return obs
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.wallet_balance = self.initial_balance
        self.position_size = 0.0
        self.entry_price = 0.0
        
        return self._get_obs(), {}
        
    def step(self, action: int):
        current_data = self.data[self.current_step]
        best_bid = current_data.get("best_bid", 0.0)
        best_ask = current_data.get("best_ask", 0.0)
        mid_price = current_data.get("mid_price", 0.0)
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # 1. Process Actions
        if action == 1 and self.position_size <= 0: # OPEN_LONG
            # If we had a short, close it first
            if self.position_size < 0:
                self._close_position(best_ask) # buy to cover
            
            # Open Long with e.g. 10% of wallet balance
            margin_allocated = self.wallet_balance * 0.10
            notional = margin_allocated * self.leverage
            qty = notional / best_ask
            
            fee = notional * self.taker_fee
            self.wallet_balance -= fee
            reward -= fee
            
            self.position_size = qty
            self.entry_price = best_ask
            
        elif action == 2 and self.position_size >= 0: # OPEN_SHORT
            if self.position_size > 0:
                self._close_position(best_bid) # sell to close
                
            margin_allocated = self.wallet_balance * 0.10
            notional = margin_allocated * self.leverage
            qty = notional / best_bid
            
            fee = notional * self.taker_fee
            self.wallet_balance -= fee
            reward -= fee
            
            self.position_size = -qty
            self.entry_price = best_bid
            
        elif action == 3 and self.position_size > 0: # CLOSE_LONG
            revenue = self._close_position(best_bid)
            reward += revenue
            
        elif action == 4 and self.position_size < 0: # CLOSE_SHORT
            revenue = self._close_position(best_ask)
            reward += revenue
            
        # 2. Continuous State Updates (Funding & Liquidations)
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            terminated = True
            
        new_data = self.data[self.current_step]
        new_mid = new_data.get("mid_price", 0.0)
        
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        if self.position_size > 0:
            unrealized_pnl = (new_mid - self.entry_price) * self.position_size
        elif self.position_size < 0:
            unrealized_pnl = (self.entry_price - new_mid) * abs(self.position_size)
            
        margin_balance = self.wallet_balance + unrealized_pnl
        
        # Maintenance Margin Liquidation Check
        if self.position_size != 0:
            notional_value = abs(self.position_size) * new_mid
            maintenance_margin = notional_value * self.mmr
            
            if margin_balance <= maintenance_margin:
                # LIQUIDATED
                self.wallet_balance -= maintenance_margin # lose the remaining margin to clearing fund
                self.position_size = 0.0
                self.entry_price = 0.0
                reward -= 500.0 # Massive penalty for liquidation
                terminated = True
                info["liquidation"] = True
                
        # Funding rate deduction
        if self.current_step % self.steps_per_funding == 0 and self.position_size != 0:
            notional_value = abs(self.position_size) * new_mid
            # If funding is positive, longs pay shorts
            if self.position_size > 0:
                funding_payment = notional_value * self.funding_rate
            else:
                funding_payment = -notional_value * self.funding_rate
                
            self.wallet_balance -= funding_payment
            reward -= funding_payment
            
        # Base reward is the step-by-step change in margin balance
        # For a rigorous setup, we might only reward closed PnL, but continuous is better for RL
        info["margin_balance"] = margin_balance
        
        # Bankruptcy Check
        if self.wallet_balance < 0:
            terminated = True
            
        return self._get_obs(), float(reward), terminated, truncated, info

    def _close_position(self, exit_price: float) -> float:
        """Helper to cleanly close positions and realize PnL."""
        if self.position_size == 0:
            return 0.0
            
        notional = abs(self.position_size) * exit_price
        fee = notional * self.taker_fee
        
        pnl = 0.0
        if self.position_size > 0:
            pnl = (exit_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - exit_price) * abs(self.position_size)
            
        self.wallet_balance += (pnl - fee)
        self.position_size = 0.0
        self.entry_price = 0.0
        
        return (pnl - fee)

from core.ai.scaling import scale_portfolio_state, scale_position_state
from core.ai.memory import EventMemoryBuffer

class PositionState:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.side = "FLAT" # FLAT, LONG, SHORT
        self.quantity = 0.0
        self.entry_price = 0.0
        self.current_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.leverage = 10
        self.margin = 0.0
        self.initial_margin = 0.0
        self.maintenance_margin = 0.0
        self.liquidation_price = 0.0
        self.funding_paid = 0.0
        self.funding_received = 0.0
        
    def to_array(self, portfolio_equity: float = 1.0) -> np.ndarray:
        # 12-dim normalized position state
        return scale_position_state(
            side=self.side,
            quantity=self.quantity,
            entry_price=self.entry_price,
            current_price=self.current_price,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            leverage=self.leverage,
            margin=self.margin,
            liquidation_price=self.liquidation_price,
            funding_net=self.funding_paid - self.funding_received,
            portfolio_equity=portfolio_equity
        )

class PortfolioState:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.max_equity = initial_balance
        self.wallet_balance = initial_balance
        self.available_balance = initial_balance
        self.equity = initial_balance
        self.used_margin = 0.0
        self.free_margin = initial_balance
        self.total_unrealized_pnl = 0.0
        self.total_exposure = 0.0
        self.long_exposure = 0.0
        self.short_exposure = 0.0
        
    def to_array(self) -> np.ndarray:
        # 9-dim normalized portfolio state
        return scale_portfolio_state(
            wallet_balance=self.wallet_balance,
            equity=self.equity,
            used_margin=self.used_margin,
            free_margin=self.free_margin,
            total_unrealized_pnl=self.total_unrealized_pnl,
            total_exposure=self.total_exposure,
            initial_balance=self.initial_balance,
            max_equity=self.max_equity
        )

class MultiAssetFuturesEnv(gym.Env):
    """
    Phase 2: Upgraded simulated multi-asset futures trading environment supporting 
    normalized PortfolioState, PositionState, and advanced (action, confidence, size) action spaces.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 historical_data: Dict[str, List[Dict[str, Any]]], 
                 symbols: List[str],
                 macro_data: List[Dict[str, float]] = None,
                 initial_balance: float = 10000.0, 
                 leverage: int = 10,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.0004,
                 maintenance_margin_rate: float = 0.004,
                 funding_rate: float = 0.0001,
                 steps_per_funding: int = 480, # 8 hours at 1min steps
                 macro_dim: int = 8
                 ):
        super(MultiAssetFuturesEnv, self).__init__()
        
        self.data = historical_data
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.mmr = maintenance_margin_rate
        self.funding_rate = funding_rate
        self.steps_per_funding = steps_per_funding
        
        self.macro_data = macro_data
        self.macro_dim = 2 if macro_data else 0
        
        self.max_steps = min([len(data_list) for data_list in self.data.values()])
        if self.macro_data:
            self.max_steps = min(self.max_steps, len(self.macro_data))
        
        # New Observation Space: 
        # (PortfolioState: 9) + N * (MarketState: 25 + PositionState: 12) + macro_dim
        obs_dim = 9 + (self.num_symbols * (25 + 12)) + self.macro_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # New Action Space: continuous box of shape (num_symbols, 3)
        # For each symbol: [action_type_logits, confidence, target_position_size]
        # We constrain it between -1 and 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_symbols, 3), dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.portfolio = PortfolioState(initial_balance)
        self.positions = {sym: PositionState(sym) for sym in self.symbols}
        self.event_memory = EventMemoryBuffer(memory_dim=5, decay_rate=0.99)
        
    def _get_obs(self):
        obs_list = []
        
        # Calculate current equity and margin
        total_unrealized_pnl = 0.0
        total_exposure = 0.0
        used_margin = 0.0
        
        for sym in self.symbols:
            pos = self.positions[sym]
            current_data = self.data[sym][self.current_step]
            # Use mid_price if available, else extract it from the 25-dim vector (index 6 roughly)
            # For simulation, we assume `mid_price` is still provided separately for PnL logic
            mid_price = current_data.get("mid_price", 0.0) 
            
            pos.current_price = mid_price
            
            pnl = 0.0
            if pos.quantity > 0:
                pnl = (mid_price - pos.entry_price) * pos.quantity
            elif pos.quantity < 0:
                pnl = (pos.entry_price - mid_price) * abs(pos.quantity)
                
            pos.unrealized_pnl = pnl
            notional = abs(pos.quantity) * mid_price
            
            total_unrealized_pnl += pnl
            total_exposure += notional
            used_margin += notional / pos.leverage
            
        self.portfolio.total_unrealized_pnl = total_unrealized_pnl
        self.portfolio.equity = self.portfolio.wallet_balance + total_unrealized_pnl
        self.portfolio.used_margin = used_margin
        self.portfolio.free_margin = self.portfolio.equity - used_margin
        self.portfolio.total_exposure = total_exposure
        
        # Track max equity for drawdown calculation
        if self.portfolio.equity > self.portfolio.max_equity:
            self.portfolio.max_equity = self.portfolio.equity
        
        # 1. Append Portfolio State
        obs_list.extend(self.portfolio.to_array())
        
        # 2. Append Market + Position State per symbol
        for sym in self.symbols:
            current_data = self.data[sym][self.current_step]
            
            # Use the 25-dim market features vector
            market_state = current_data.get("market_features", [0.0]*25)
            # Ensure it's exactly 25
            if len(market_state) < 25:
                market_state = list(market_state) + [0.0]*(25-len(market_state))
            elif len(market_state) > 25:
                market_state = market_state[:25]
                
            obs_list.extend(market_state)
            obs_list.extend(self.positions[sym].to_array(portfolio_equity=self.portfolio.equity))
            
        # 3. Append macro state and regime
        if self.macro_data:
            macro_state = self.macro_data[self.current_step]
            
            # Extract basic regime (mocked if not present in macro_data yet)
            regime = macro_state.get("regime", 0.0)
            
            obs_list.extend([
                macro_state.get("sentiment_score", 0.0),
                macro_state.get("volatility_expectation", 0.0),
            ])
        # 4. Append Event Memory
        memory_vector = self.event_memory.step()
        obs_list.extend(memory_vector)
            
        return np.array(obs_list, dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio = PortfolioState(self.initial_balance)
        self.positions = {sym: PositionState(sym) for sym in self.symbols}
        return self._get_obs(), {}
        
    def _close_position(self, sym: str, exit_price: float) -> float:
        pos = self.positions[sym]
        if pos.quantity == 0:
            return 0.0
            
        notional = abs(pos.quantity) * exit_price
        fee = notional * self.taker_fee
        
        pnl = 0.0
        if pos.quantity > 0:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * abs(pos.quantity)
            
        self.portfolio.wallet_balance += (pnl - fee)
        
        # Reset position state
        pos.quantity = 0.0
        pos.entry_price = 0.0
        pos.side = "FLAT"
        pos.unrealized_pnl = 0.0
        pos.margin = 0.0
        
        return (pnl - fee)
        
    def _get_binance_mmr(self, notional: float) -> float:
        """Mock Binance Tiered Maintenance Margin Rate."""
        if notional < 50000: return 0.004 # Tier 1
        if notional < 250000: return 0.005 # Tier 2
        if notional < 1000000: return 0.01 # Tier 3
        return 0.025 # Tier 4+
        
    def step(self, actions):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Simulate network/processing latency (1-3 steps delay for execution)
        latency_ticks = np.random.randint(1, 4)
        execution_step = min(self.current_step + latency_ticks, self.max_steps - 1)
        
        # actions shape is (num_symbols, 3)
        for i, sym in enumerate(self.symbols):
            action_vec = actions[i]
            
            # Decode the action vector
            action_val = action_vec[0]
            confidence = (action_vec[1] + 1.0) / 2.0 # Scale to [0, 1]
            target_size = (action_vec[2] + 1.0) / 2.0 # Scale to [0, 1]
            
            action = 0 # HOLD
            is_maker = False
            
            # We treat extreme values as Taker (Market Order) and moderate values as Maker (Limit)
            if action_val < -0.8: action, is_maker = 4, False # Aggressive CLOSE_SHORT
            elif action_val < -0.5: action, is_maker = 4, True # Passive CLOSE_SHORT (Limit)
            elif action_val < -0.2: action, is_maker = 2, False # OPEN_SHORT
            elif action_val > 0.8: action, is_maker = 3, False # Aggressive CLOSE_LONG
            elif action_val > 0.5: action, is_maker = 3, True # Passive CLOSE_LONG (Limit)
            elif action_val > 0.2: action, is_maker = 1, False # OPEN_LONG
            
            # Execute at the delayed tick
            exec_data = self.data[sym][execution_step]
            best_bid = exec_data.get("best_bid", exec_data.get("mid_price", 0.0))
            best_ask = exec_data.get("best_ask", exec_data.get("mid_price", 0.0))
            pos = self.positions[sym]
            
            # Base fee
            applied_fee_rate = self.maker_fee if is_maker else self.taker_fee
            
            # Slippage modeling (worse execution for larger sizes, worse for takers)
            slippage_factor = 0.0 if is_maker else (np.random.uniform(0.0001, 0.0005) * target_size)
            
            # Apply confidence thresholding
            if confidence < 0.3:
                action = 0 # Too uncertain, force hold
                
            # Simulate Maker fill probability (Stochastic model)
            if is_maker and np.random.random() > 0.7:
                action = 0 # Limit order didn't get filled this step
                
            if action == 1 and pos.quantity <= 0: # OPEN_LONG
                if pos.quantity < 0:
                    self._close_position(sym, best_ask * (1 + slippage_factor))
                    
                # Use target_size * available free margin
                margin_allocated = max(0, self.portfolio.free_margin) * target_size
                if margin_allocated > 1.0:
                    notional = margin_allocated * self.leverage
                    exec_price = best_ask * (1 + slippage_factor)
                    qty = notional / exec_price
                    
                    fee = notional * applied_fee_rate
                    self.portfolio.wallet_balance -= fee
                    reward -= fee
                    
                    pos.quantity = qty
                    pos.entry_price = exec_price
                    pos.side = "LONG"
                    pos.leverage = self.leverage
                    pos.margin = margin_allocated
                
            elif action == 2 and pos.quantity >= 0: # OPEN_SHORT
                if pos.quantity > 0:
                    self._close_position(sym, best_bid * (1 - slippage_factor))
                    
                margin_allocated = max(0, self.portfolio.free_margin) * target_size
                if margin_allocated > 1.0:
                    notional = margin_allocated * self.leverage
                    exec_price = best_bid * (1 - slippage_factor)
                    qty = notional / exec_price
                    
                    fee = notional * applied_fee_rate
                    self.portfolio.wallet_balance -= fee
                    reward -= fee
                    
                    pos.quantity = -qty
                    pos.entry_price = exec_price
                    pos.side = "SHORT"
                    pos.leverage = self.leverage
                    pos.margin = margin_allocated
                
            elif action == 3 and pos.quantity > 0: # CLOSE_LONG
                rev = self._close_position(sym, best_bid * (1 - slippage_factor))
                reward += rev
                
            elif action == 4 and pos.quantity < 0: # CLOSE_SHORT
                rev = self._close_position(sym, best_ask * (1 + slippage_factor))
                reward += rev

        self.current_step += 1
        if self.current_step >= self.max_steps - 1:
            terminated = True
            
        # Get obs to recalculate portfolio equity
        obs = self._get_obs()
        
        # Rigorous Cross-Margin Liquidation Check with Tiered MMR
        total_mmr_requirement = 0.0
        for sym in self.symbols:
            pos = self.positions[sym]
            if pos.quantity != 0:
                notional = abs(pos.quantity) * pos.current_price
                tiered_mmr = self._get_binance_mmr(notional)
                pos.maintenance_margin = notional * tiered_mmr
                total_mmr_requirement += pos.maintenance_margin
                
        # Liquidation occurs if Margin Balance (Equity) <= Maintenance Margin
        if total_mmr_requirement > 0 and self.portfolio.equity <= total_mmr_requirement:
            self.portfolio.wallet_balance -= total_mmr_requirement # Clearing fund takes remainder
            for sym in self.symbols:
                self.positions[sym].quantity = 0.0
                self.positions[sym].side = "FLAT"
            reward -= 1000.0 # Massive penalty
            self.event_memory.add_event("liquidation", 1.0)
            terminated = True
            info["liquidation"] = True
            
        # Trigger massive win/loss memory
        if reward > 500.0:
            self.event_memory.add_event("massive_win", min(1.0, reward / 5000.0))
        elif reward < -500.0 and not terminated: # Not a liquidation
            self.event_memory.add_event("massive_loss", min(1.0, abs(reward) / 5000.0))
            
        # Funding rate deduction
        if self.current_step % self.steps_per_funding == 0:
            for sym in self.symbols:
                pos = self.positions[sym]
                if pos.quantity != 0:
                    new_mid = self.data[sym][self.current_step].get("mid_price", 0.0)
                    notional = abs(pos.quantity) * new_mid
                    payment = notional * self.funding_rate if pos.quantity > 0 else -notional * self.funding_rate
                    self.portfolio.wallet_balance -= payment
                    reward -= payment
                    
                    if pos.quantity > 0:
                        pos.funding_paid += max(0, payment)
                        pos.funding_received += max(0, -payment)
                    else:
                        pos.funding_paid += max(0, payment)
                        pos.funding_received += max(0, -payment)
                    
        if self.portfolio.wallet_balance < 0:
            terminated = True
            
        info["margin_balance"] = self.portfolio.equity
        return obs, float(reward), terminated, truncated, info
