from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # MySQL
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="trading_user")
    mysql_password: str = Field(default="trading_password")
    mysql_database: str = Field(default="ai_trading")

    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)

    # Binance Testnet
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    binance_testnet: bool = Field(default=True)

    # Phase 1 Safety Gates
    trading_enabled: bool = Field(default=False)
    dry_run: bool = Field(default=True)
    allow_testnet: bool = Field(default=False)
    allow_live_trading: bool = Field(default=False)
    emergency_stop: bool = Field(default=True)

    # Trading Symbols
    trading_symbol: str = Field(default="BTCUSDT")
    symbol_universe: list[str] = Field(default=["BTCUSDT", "ETHUSDT"])

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def database_url(self) -> str:
        return f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        
    @property
    def async_database_url(self) -> str:
        return f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        
    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"

settings = Settings()
