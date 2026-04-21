"""Constants used throughout the trading system."""

from enum import IntEnum
from typing import Final

# =============================================================================
# Trading Actions
# =============================================================================

class Action(IntEnum):
    """Trading actions available to the agent."""
    
    NONE = 0    # Do nothing
    BUY = 1     # Open long position
    SELL = 2    # Open short position
    CLOSE = 3   # Close current position


# =============================================================================
# Data Constants
# =============================================================================

# OHLCV column names
OHLCV_COLUMNS: Final[list[str]] = ["open", "high", "low", "close", "volume"]

# Number of raw features from the data source (OHLCV + Gold)
NUM_RAW_FEATURES: Final[int] = 6

# Number of total features (raw + technical indicators + external)
# 5 (OHLCV) + 11 (indicators) + 1 (gold) + 1 (hour) + 1 (day) = 19
NUM_TOTAL_FEATURES: Final[int] = 19

# Position info features: [has_position, is_long, unrealized_pnl_normalized]
POSITION_FEATURES: Final[list[str]] = ["has_position", "is_long", "unrealized_pnl_norm"]
NUM_POSITION_FEATURES: Final[int] = 3

# Account info features: [total_loss_normalized, available_margin_normalized]
ACCOUNT_FEATURES: Final[list[str]] = ["total_loss_norm", "margin_norm"]
NUM_ACCOUNT_FEATURES: Final[int] = 2


# =============================================================================
# Model Constants
# =============================================================================

# Default model architecture
DEFAULT_SEQUENCE_LENGTH: Final[int] = 120
DEFAULT_HIDDEN_SIZE: Final[int] = 256
DEFAULT_NUM_LAYERS: Final[int] = 2
DEFAULT_ATTENTION_HEADS: Final[int] = 4
DEFAULT_EMBEDDING_DIM: Final[int] = 64
DEFAULT_DROPOUT: Final[float] = 0.1


# =============================================================================
# Training Constants
# =============================================================================

# PPO hyperparameters
DEFAULT_LEARNING_RATE: Final[float] = 3e-4
DEFAULT_GAMMA: Final[float] = 0.99
DEFAULT_GAE_LAMBDA: Final[float] = 0.95
DEFAULT_CLIP_EPSILON: Final[float] = 0.2
DEFAULT_ENTROPY_COEF: Final[float] = 0.01
DEFAULT_VALUE_COEF: Final[float] = 0.5
DEFAULT_PREDICTION_COEF: Final[float] = 0.1
DEFAULT_MAX_GRAD_NORM: Final[float] = 0.5
DEFAULT_NUM_EPOCHS: Final[int] = 10
DEFAULT_BATCH_SIZE: Final[int] = 64


# =============================================================================
# Trading Constants
# =============================================================================

# Default trading parameters
DEFAULT_LOT_SIZE: Final[float] = 0.3
DEFAULT_STOP_LOSS_USD: Final[float] = 300.0
DEFAULT_TAKE_PROFIT_USD: Final[float] = 500.0
DEFAULT_MAX_LOSS_USD: Final[float] = 1000.0

# XAGUSD contract specifications (typical values, may vary by broker)
XAGUSD_CONTRACT_SIZE: Final[float] = 5000.0  # 1 lot = 5000 oz
XAGUSD_TICK_SIZE: Final[float] = 0.001
XAGUSD_TICK_VALUE: Final[float] = 5.0  # USD per tick per lot


# =============================================================================
# Reward Constants
# =============================================================================

# Reward scaling factors
PREDICTION_PENALTY_SCALE: Final[float] = 1.0
PNL_REWARD_SCALE: Final[float] = 0.01
LOSS_PENALTY_DENOMINATOR: Final[float] = 500.0  # For penalty_weight = 1 + (loss / this)


# =============================================================================
# Server Constants
# =============================================================================

DEFAULT_SERVER_HOST: Final[str] = "0.0.0.0"
DEFAULT_SERVER_PORT: Final[int] = 8000


# =============================================================================
# Checkpoint Constants
# =============================================================================

DEFAULT_CHECKPOINT_INTERVAL_MINUTES: Final[int] = 360  # 6 hours
CHECKPOINT_FILENAME_FORMAT: Final[str] = "checkpoint_epoch{epoch:04d}_{timestamp}.pt"
BEST_MODEL_FILENAME: Final[str] = "best_model.pt"
LATEST_MODEL_FILENAME: Final[str] = "latest_model.pt"

