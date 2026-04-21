"""Application settings using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.constants import (
    DEFAULT_ATTENTION_HEADS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL_MINUTES,
    DEFAULT_CLIP_EPSILON,
    DEFAULT_DROPOUT,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_GAE_LAMBDA,
    DEFAULT_GAMMA,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOT_SIZE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_LOSS_USD,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_PREDICTION_COEF,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    DEFAULT_STOP_LOSS_USD,
    DEFAULT_TAKE_PROFIT_USD,
    DEFAULT_VALUE_COEF,
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Application Settings
    # =========================================================================
    app_env: Literal["development", "production", "testing"] = "development"
    debug: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # =========================================================================
    # Server Settings
    # =========================================================================
    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = DEFAULT_SERVER_PORT
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )

    # =========================================================================
    # Match-Trader API Settings
    # =========================================================================
    match_trader_api_url: str = "https://api.match-trader.com"
    match_trader_api_key: str = ""
    match_trader_api_secret: str = ""
    match_trader_account_id: str = ""
    match_trader_demo_mode: bool = True

    # =========================================================================
    # Trading Settings
    # =========================================================================
    trading_symbol: str = "XAGUSD"
    trading_timeframe: str = "1m"
    trading_lot_size: float = DEFAULT_LOT_SIZE
    trading_stop_loss_usd: float = DEFAULT_STOP_LOSS_USD
    trading_take_profit_usd: float = DEFAULT_TAKE_PROFIT_USD
    trading_max_loss_usd: float = DEFAULT_MAX_LOSS_USD

    # =========================================================================
    # Model Settings
    # =========================================================================
    model_sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    model_input_dim: int = 19
    model_hidden_size: int = DEFAULT_HIDDEN_SIZE
    model_num_layers: int = DEFAULT_NUM_LAYERS
    model_attention_heads: int = DEFAULT_ATTENTION_HEADS
    model_dropout: float = DEFAULT_DROPOUT

    # =========================================================================
    # Training Settings
    # =========================================================================
    training_batch_size: int = DEFAULT_BATCH_SIZE
    training_learning_rate: float = DEFAULT_LEARNING_RATE
    training_gamma: float = DEFAULT_GAMMA
    training_gae_lambda: float = DEFAULT_GAE_LAMBDA
    training_clip_epsilon: float = DEFAULT_CLIP_EPSILON
    training_entropy_coef: float = DEFAULT_ENTROPY_COEF
    training_value_coef: float = DEFAULT_VALUE_COEF
    training_prediction_coef: float = DEFAULT_PREDICTION_COEF
    training_max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    training_num_epochs: int = DEFAULT_NUM_EPOCHS
    training_checkpoint_interval_minutes: int = DEFAULT_CHECKPOINT_INTERVAL_MINUTES
    training_use_adaptive_reward: bool = False

    # =========================================================================
    # Path Settings
    # =========================================================================
    data_dir: Path = Path("./data")
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")
    tensorboard_dir: Path = Path("./logs/tensorboard")
    trade_log_dir: Path = Path("./logs/trades")

    # =========================================================================
    # Device Settings
    # =========================================================================
    device: Literal["auto", "cuda", "cpu"] = "auto"

    # =========================================================================
    # Validators
    # =========================================================================
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Handle JSON-like string from env
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Handle comma-separated string
                return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("data_dir", "checkpoint_dir", "log_dir", "tensorboard_dir", "trade_log_dir", mode="before")
    @classmethod
    def parse_path(cls, v: str | Path) -> Path:
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v

    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def historical_data_dir(self) -> Path:
        """Get historical data directory."""
        return self.data_dir / "historical"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.app_env == "production"

    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.data_dir,
            self.historical_data_dir,
            self.checkpoint_dir,
            self.log_dir,
            self.tensorboard_dir,
            self.trade_log_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

