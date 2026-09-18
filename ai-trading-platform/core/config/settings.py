from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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
    max_position_size: float = Field(default=9999999.0, gt=0)
    max_symbol_exposure_pct: float = Field(default=25.0, gt=0)
    max_portfolio_exposure_pct: float = Field(default=25.0, gt=0)

    max_leverage: int = Field(default=20, gt=0)
    max_order_size: float = Field(default=9999999.0, gt=0)
    max_open_positions: int = Field(default=5, gt=0)

    max_daily_loss_pct: float = Field(default=0.05, gt=0, le=1)
    max_drawdown_pct: float = Field(default=0.10, gt=0, le=1)

    max_market_data_age_seconds: float = Field(default=60.0, gt=0)

    correlated_exposure_limit_pct: float = Field(
        default=25.0,
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
            "DOGEUSDT",
        ]
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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