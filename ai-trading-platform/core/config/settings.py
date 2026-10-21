from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # ============================================================
    # MySQL
    # ============================================================
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="trading_user")
    mysql_password: str = Field(default="trading_password")
    mysql_database: str = Field(default="ai_trading")

    # ============================================================
    # Redis
    # ============================================================
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)

    # ============================================================
    # Binance Futures
    # ============================================================
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    binance_testnet: bool = Field(default=True)
    # Market data (features, prices for stops) comes from the venue the bot trades on by default so
    # position PnL and exits use the same prices as the fills. Set True to read mainnet data while
    # trading testnet (more realistic order book, but prices can differ slightly from testnet fills).
    market_data_mainnet: bool = Field(default=False)

    # ============================================================
    # Phase 1 SAFETY GATES
    # All defaults must be safe.
    # ============================================================
    trading_enabled: bool = Field(default=False)
    dry_run: bool = Field(default=True)
    allow_testnet: bool = Field(default=False)
    allow_live_trading: bool = Field(default=False)
    emergency_stop: bool = Field(default=True)

    # ============================================================
    # Phase 1 RISK LIMITS
    # ============================================================
    max_position_size: float = Field(default=50000.0, gt=0)   # USD notional per coin (40% margin @20x is ~8x equity)
    # Portfolio & Symbol Exposure Limits (Leverage multiples of equity: 8.0 = 800% notional exposure / 8x leverage)
    max_symbol_exposure_pct: float = Field(default=8.0, gt=0)     # Max 8x equity notional per symbol (40% margin @ 20x)
    max_portfolio_exposure_pct: float = Field(default=10.0, gt=0) # Max 10x aggregate portfolio leverage

    max_leverage: int = Field(default=20, gt=0)
    max_position_margin_pct: float = Field(default=0.40, gt=0, le=1.0)  # Max 40% of equity margin per coin (up to 40% on majors, 35% on alts)
    min_close_confidence: float = Field(default=0.60, ge=0, le=1.0)     # Avoid closing on noise
    min_open_expected_return_pct: float = Field(default=0.22, ge=0)     # 0.22% min predicted move (ensures net profit after fees & spread)
    min_edge_cost_multiple: float = Field(default=2.0, ge=1.0)          # expected move must be >= 2x round-trip cost
    entry_cooldown_seconds: float = Field(default=600.0, ge=0.0)       # per-coin wait after ANY close before re-opening
    max_entries_per_hour: int = Field(default=6, ge=1)
    max_entries_per_cycle: int = Field(default=2, ge=1)                # new positions per 5s decision cycle (no bursts)
    entry_margin_reserve_pct: float = Field(default=0.25, ge=0.0, le=0.9)  # equity share new entries must leave free                # portfolio-wide new positions per rolling hour
    # Unfilled maker entries: after the 30s wait, buy/sell the rest at market (taker fee) only for strong signals
    entry_topup_min_move_pct: float = Field(default=0.30, ge=0.0)        # predicted move must be >= 0.30% (normal hurdle 0.22%)
    entry_topup_below_fill_frac: float = Field(default=0.5, ge=0.0, le=1.0)  # only if less than half of the order filled
    entry_topup_max_adverse_pct: float = Field(default=0.05, ge=0.0)     # skip if price already ran > 0.05% past our limit
    experience_snapshot_seconds: float = Field(default=600.0, ge=60.0)  # one training record per coin every 10 min, whatever the bot decides (~2,600/day)
    experience_retention_days: int = Field(default=21, ge=3)            # VPS keeps 3 weeks of records; the laptop trainer syncs them within minutes
    min_net_edge_hurdle_pct: float = Field(default=0.08, ge=0.0)        # 0.08% min net edge after fees, spread, slippage, and uncertainty penalty
    
    # Institutional Session Confidence Matrix (Middle Ground Calibration)
    conf_hurdle_ny_peak: float = Field(default=0.62, ge=0.0, le=1.0)   # US New York Peak Trend (13:30 - 21:00 UTC)
    conf_hurdle_london: float = Field(default=0.62, ge=0.0, le=1.0)    # London Trend (08:00 - 13:30 UTC)
    conf_hurdle_asia: float = Field(default=0.67, ge=0.0, le=1.0)      # Asia Off-Peak Chop (00:00 - 08:00 UTC)
    conf_hurdle_weekend: float = Field(default=0.68, ge=0.0, le=1.0)   # Weekend Low Liquidity (Saturday & Sunday)
    
    # Dynamic Margin Sizing Bounds (Enhanced Allocation: 25%-40% Majors, 20%-35% Alts)
    major_min_margin_pct: float = Field(default=0.25, ge=0.01, le=0.50)  # 25% min margin on majors (BTC, ETH, SOL, BNB)
    major_max_margin_pct: float = Field(default=0.40, ge=0.01, le=0.50)  # 40% max margin on majors
    alt_min_margin_pct: float = Field(default=0.20, ge=0.01, le=0.50)    # 20% min margin on altcoins (TAO, ADA, XRP, NEAR, etc.)
    alt_max_margin_pct: float = Field(default=0.35, ge=0.01, le=0.50)    # 35% max margin on altcoins
    
    taker_fee_pct: float = Field(default=0.0005, ge=0)                  # 0.05% taker fee
    maker_fee_pct: float = Field(default=0.0002, ge=0)                  # 0.02% maker fee

    max_order_size: float = Field(default=50000.0, gt=0)      # USD notional per order
    max_open_positions: int = Field(default=10, gt=0)
    # Institutional Drawdown & Circuit Breaker Gates (0.05 = 5.0% daily loss limit, 0.08 = 8.0% max peak drawdown)
    max_daily_loss_pct: float = Field(default=0.05, gt=0, le=1.0)
    max_drawdown_pct: float = Field(default=0.08, gt=0, le=1.0)

    max_market_data_age_seconds: float = Field(default=60.0, gt=0)

    correlated_exposure_limit_pct: float = Field(
        default=5.0,  # Max 5x correlated leverage multiple for BTC/ETH
        gt=0,
    )

    # ============================================================
    # Gemini AI Integration (For News/Macro State)
    # ============================================================
    gemini_api_key: str = Field(default="")
    gemini_model_version: str = Field(default="gemini-2.5-flash")

    # ============================================================
    # Trading Symbols
    # ============================================================
    trading_symbol: str = Field(default="BTCUSDT")

    symbol_universe: list[str] = Field(
        default_factory=lambda: [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "LINKUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "NEARUSDT",
            "LTCUSDT",
            "SUIUSDT",
            "DOGEUSDT",
            "RENDERUSDT",
            "1000PEPEUSDT",
            "APTUSDT",
            "TAOUSDT",
            "WIFUSDT",
            "INJUSDT",
        ]
    )

    # Excluded from active entry (thin testnet book depth / fee preservation)
    excluded_trading_symbols: list[str] = Field(
        default_factory=lambda: [
            "SUIUSDT",
        ]
    )

    # ============================================================
    # Institutional Upgrade 1: Macro & Flash-Crash Shield
    # ============================================================
    macro_flash_crash_threshold_pct: float = Field(default=0.025, gt=0.0, le=0.20)  # 2.5% drop in 5m
    macro_flash_crash_lockout_seconds: float = Field(default=1800.0, gt=0.0)        # 30-minute lockout
    macro_flash_crash_trailing_retention: float = Field(default=0.90, ge=0.5, le=1.0) # Ratchet trailing retention
    economic_shield_enabled: bool = Field(default=True)
    economic_shield_buffer_minutes: float = Field(default=15.0, ge=1.0, le=120.0)   # 15 mins before & after news

    # ============================================================
    # Institutional Upgrade 2: Multi-Exchange Liquidation Cascade Hunter
    # ============================================================
    liquidation_cascade_hunter_enabled: bool = Field(default=True)
    liquidation_cascade_threshold_usd: float = Field(default=20_000_000.0, gt=0.0)      # $20M aggregated cascade
    liquidation_cascade_alt_threshold_usd: float = Field(default=1_500_000.0, gt=0.0)   # $1.5M for single alt
    liquidation_exhaustion_decay_pct: float = Field(default=0.75, ge=0.1, le=0.99)      # 75% liquidation velocity decay

    # ============================================================
    # Institutional Upgrade 3: Market-Beta Hedger (Delta-Neutral) [DISABLED]
    # ============================================================
    beta_hedging_enabled: bool = Field(default=False)
    beta_hedge_ratio: float = Field(default=0.0, ge=0.0, le=2.0)                        # 0.0 = Disabled
    beta_hedge_rebalance_threshold_usd: float = Field(default=250.0, gt=0.0)            # $250 minimum rebalance delta
    beta_hedge_symbol: str = Field(default="BTCUSDT")

    # ============================================================
    # Institutional Upgrade 4: Interactive Telegram Command Center
    # ============================================================
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    telegram_enabled: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_safety_and_risk(self) -> "Settings":
        # 1. Leverage sanity bounds
        if self.max_leverage <= 0 or self.max_leverage > 125:
            raise ValueError(f"Invalid max_leverage: {self.max_leverage}. Must be between 1 and 125.")
        
        # 2. Position and order sizing
        if self.max_position_size <= 0:
            raise ValueError(f"max_position_size must be positive (got {self.max_position_size}).")
        if self.max_order_size <= 0:
            raise ValueError(f"max_order_size must be positive (got {self.max_order_size}).")
        if self.max_order_size > self.max_position_size:
            raise ValueError(f"max_order_size ({self.max_order_size}) cannot exceed max_position_size ({self.max_position_size}).")
            
        # 3. Exposure limits
        if not (0 < self.max_symbol_exposure_pct <= 20.0):
            raise ValueError(f"Invalid max_symbol_exposure_pct: {self.max_symbol_exposure_pct}.")
        if not (0 < self.max_portfolio_exposure_pct <= 20.0):
            raise ValueError(f"Invalid max_portfolio_exposure_pct: {self.max_portfolio_exposure_pct}.")
        if not (0 < self.correlated_exposure_limit_pct <= 20.0):
            raise ValueError(f"Invalid correlated_exposure_limit_pct: {self.correlated_exposure_limit_pct}.")
            
        # 4. Drawdown & loss percentages
        if not (0 < self.max_daily_loss_pct <= 1.0):
            raise ValueError(f"Invalid max_daily_loss_pct: {self.max_daily_loss_pct}. Must be in (0, 1.0].")
        if not (0 < self.max_drawdown_pct <= 1.0):
            raise ValueError(f"Invalid max_drawdown_pct: {self.max_drawdown_pct}. Must be in (0, 1.0].")
            
        # 5. Live trading safe defaults
        if self.allow_live_trading and self.dry_run:
            raise ValueError("Configuration conflict: allow_live_trading cannot be enabled while dry_run is True.")
            
        return self

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://"
            f"{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}"
            f"/{self.mysql_database}"
        )

    @property
    def async_database_url(self) -> str:
        return (
            f"mysql+aiomysql://"
            f"{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}"
            f"/{self.mysql_database}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"


settings = Settings()