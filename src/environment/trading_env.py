"""Trading environment compatible with Gymnasium API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, SupportsFloat, TypeAlias

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config.constants import (
    Action,
    NUM_OHLCV_FEATURES,
    NUM_POSITION_FEATURES,
    NUM_ACCOUNT_FEATURES,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_LOT_SIZE,
    DEFAULT_STOP_LOSS_USD,
    DEFAULT_TAKE_PROFIT_USD,
    DEFAULT_MAX_LOSS_USD,
)
from src.data.candle import Candle, CandleBuffer
from src.data.preprocessor import Preprocessor
from src.environment.position_manager import (
    PositionManager,
    PositionSide,
    CloseReason,
    ClosedPosition,
)
from src.environment.reward_calculator import RewardCalculator, RewardComponents


# Type aliases
ObsType: TypeAlias = dict[str, np.ndarray]
ActType: TypeAlias = dict[str, np.ndarray]


@dataclass
class StepResult:
    """Result of an environment step."""
    
    observation: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    reward_components: RewardComponents | None = None


class TradingEnvironment(gym.Env):
    """Trading environment for XAGUSD with hybrid action space.
    
    Observation Space:
    - candles: [sequence_length, 16] - Normalized OHLCV + Technical Indicators
    - position: [3] - (has_position, is_long, unrealized_pnl_norm)
    - account: [2] - (total_loss_norm, margin_norm)
    
    Action Space (Hybrid):
    - prediction: [16] - Predicted next candle features (continuous)
    - trading_action: Discrete(4) - NONE, BUY, SELL, CLOSE
    
    Reward:
    - Prediction penalty (MAPE) weighted by total loss
    - PnL from trading (realized + unrealized delta)
    
    Episode ends when:
    - Total loss + unrealized PnL > max_loss_usd (failure)
    - End of data (training) or manual stop (live)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        lot_size: float = DEFAULT_LOT_SIZE,
        stop_loss_usd: float = DEFAULT_STOP_LOSS_USD,
        take_profit_usd: float = DEFAULT_TAKE_PROFIT_USD,
        max_loss_usd: float = DEFAULT_MAX_LOSS_USD,
        initial_balance: float = 10000.0,
        input_dim: int = 16, # Extended features
        normalize_obs: bool = True,
        reward_calculator: RewardCalculator | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the trading environment.
        
        Args:
            sequence_length: Number of candles in observation
            lot_size: Position size in lots
            stop_loss_usd: Stop loss per position
            take_profit_usd: Take profit per position
            max_loss_usd: Maximum total loss before episode fails
            initial_balance: Starting balance for margin calculation
            normalize_obs: Whether to normalize observations
            reward_calculator: Custom reward calculator (optional)
            seed: Random seed
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        self.max_loss_usd = max_loss_usd
        self.initial_balance = initial_balance
        self.normalize_obs = normalize_obs
        self.lot_size = lot_size
        self.trading_stop_loss_usd = stop_loss_usd
        self.trading_take_profit_usd = take_profit_usd
        
        # Components
        self.position_manager = PositionManager(
            lot_size=lot_size,
            stop_loss_usd=stop_loss_usd,
            take_profit_usd=take_profit_usd,
        )
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.preprocessor = Preprocessor(method="rolling_zscore")
        from src.utils.news_filter import NewsFilter
        self.news_filter = NewsFilter()
        self.news_filter.load_mock_events()
        self.candle_buffer = CandleBuffer(max_size=sequence_length)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "candles": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(sequence_length, 16),
                dtype=np.float32,
            ),
            "position": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(NUM_POSITION_FEATURES,),
                dtype=np.float32,
            ),
            "account": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(NUM_ACCOUNT_FEATURES,),
                dtype=np.float32,
            ),
        })
        
        # Define action space (hybrid)
        self.action_space = spaces.Dict({
            "prediction": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(16,),
                dtype=np.float32,
            ),
            "trading_action": spaces.Discrete(4),  # NONE, BUY, SELL, CLOSE
        })
        
        # State tracking
        self.input_dim = 16
        self._current_candle: Candle | None = None
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_reward_components: RewardComponents | None = None
        
        # Set seed if provided
        if seed is not None:
            self.reset(seed=seed)
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options (can include initial candles)
            
        Returns:
            Tuple of (initial observation, info dict)
        """
        super().reset(seed=seed)
        
        # Reset components
        self.position_manager.reset()
        self.reward_calculator.reset()
        self.preprocessor.reset_rolling_state()
        self.candle_buffer.clear()
        
        # Reset state
        self._current_candle = None
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_reward_components = None
        
        # Handle initial candles if provided
        if options and "initial_candles" in options:
            initial_candles = options["initial_candles"]
            for candle in initial_candles:
                if isinstance(candle, Candle):
                    self.candle_buffer.add(candle)
                elif isinstance(candle, np.ndarray):
                    self.candle_buffer.add(Candle.from_array(candle))
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Dict with 'prediction' and 'trading_action'
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        prediction = action["prediction"]
        trading_action = Action(int(action["trading_action"]))
        
        # This should be called after receiving a new candle
        if self._current_candle is None:
            raise RuntimeError("Must call receive_candle() before step()")
        
        actual_candle = self._current_candle
        
        # Execute trading action
        realized_pnl = self._execute_trading_action(
            trading_action, actual_candle.close, actual_candle.timestamp
        )
        
        # Check auto SL/TP
        auto_closed = self.position_manager.process_price_update(
            actual_candle.close, actual_candle.timestamp
        )
        if auto_closed:
            realized_pnl += auto_closed.realized_pnl
        
        # Calculate reward
        reward_components = self.reward_calculator.calculate_reward(
            predicted=prediction,
            actual=actual_candle.to_array(),
            total_loss=self.position_manager.get_total_loss(),
            realized_pnl=realized_pnl,
            current_unrealized_pnl=self.position_manager.get_unrealized_pnl(),
        )
        
        reward = reward_components.total_reward
        self._episode_reward += reward
        self._last_reward_components = reward_components
        self._step_count += 1
        
        # Check termination
        terminated = self._check_termination()
        truncated = False  # Will be set by data source
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info(
            trading_action=trading_action,
            prediction=prediction,
            actual=actual_candle.to_array(),
            realized_pnl=realized_pnl,
            reward_components=reward_components,
            auto_closed=auto_closed,
        )
        
        # Clear current candle (must receive new one before next step)
        self._current_candle = None
        
        return obs, reward, terminated, truncated, info
    
    def receive_candle(self, candle: Candle | np.ndarray) -> None:
        """Receive a new candle from data source.
        
        This should be called before step() to provide the new market data.
        
        Args:
            candle: New candle data
        """
        if isinstance(candle, np.ndarray):
            candle = Candle.from_array(candle)
        
        self._current_candle = candle
        self.candle_buffer.add(candle)
        
        # Update position with new price
        if self.position_manager.has_position():
            self.position_manager.update_price(candle.close)
    
    def _execute_trading_action(
        self,
        action: Action,
        price: float,
        timestamp: datetime,
    ) -> float:
        """Execute the trading action.
        
        Args:
            action: Trading action to execute
            price: Current price
            timestamp: Current timestamp
            
        Returns:
            Realized PnL (if position was closed)
        """
        realized_pnl = 0.0
        
        if action == Action.NONE:
            pass
        
        elif action == Action.BUY:
            if not self.position_manager.has_position():
                if self.news_filter.is_safe_to_trade(timestamp):
                    sl, tp = self._calculate_dynamic_sl_tp(price)
                    self.position_manager.open_position(
                        PositionSide.LONG, price, timestamp,
                        stop_loss_usd=sl, take_profit_usd=tp
                    )
                else:
                    logger.info("BUY action blocked by NewsFilter")
        
        elif action == Action.SELL:
            if not self.position_manager.has_position():
                if self.news_filter.is_safe_to_trade(timestamp):
                    sl, tp = self._calculate_dynamic_sl_tp(price)
                    self.position_manager.open_position(
                        PositionSide.SHORT, price, timestamp,
                        stop_loss_usd=sl, take_profit_usd=tp
                    )
                else:
                    logger.info("SELL action blocked by NewsFilter")
        
        elif action == Action.CLOSE:
            if self.position_manager.has_position():
                closed = self.position_manager.close_position(
                    price, CloseReason.MANUAL, timestamp
                )
                if closed:
                    realized_pnl = closed.realized_pnl
        
        return realized_pnl

    def _calculate_dynamic_sl_tp(self, current_price: float) -> tuple[float, float]:
        """Calculate ATR-based stop loss and take profit.
        
        Returns:
            Tuple of (stop_loss_usd, take_profit_usd)
        """
        # Get last 20 candles for ATR
        raw_candles = self.candle_buffer.to_array()
        if len(raw_candles) < 20:
            return self.trading_stop_loss_usd, self.trading_take_profit_usd
        
        from src.data.preprocessor import Preprocessor
        from src.config.constants import XAGUSD_CONTRACT_SIZE
        
        high = raw_candles[:, 1]
        low = raw_candles[:, 2]
        close = raw_candles[:, 3]
        
        atr_values = Preprocessor.calculate_atr(high, low, close, window=14)
        current_atr = atr_values[-1]
        
        # Default if ATR is invalid
        if np.isnan(current_atr) or current_atr <= 0:
            return self.trading_stop_loss_usd, self.trading_take_profit_usd
            
        # Stop Loss: 1.5x ATR
        # Take Profit: 2.5x ATR (Providing a good Risk:Reward ratio)
        sl_price_diff = 1.5 * current_atr
        tp_price_diff = 2.5 * current_atr
        
        # Convert price diff to USD
        # PnL = diff * volume * contract_size
        sl_usd = sl_price_diff * self.lot_size * XAGUSD_CONTRACT_SIZE
        tp_usd = tp_price_diff * self.lot_size * XAGUSD_CONTRACT_SIZE
        
        # Ensure minimums to avoid micro-stops
        sl_usd = max(sl_usd, 100.0)
        tp_usd = max(tp_usd, 150.0)
        
        return float(sl_usd), float(tp_usd)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate.
        
        Returns:
            True if episode should end (failure)
        """
        total_pnl = self.position_manager.get_total_pnl()
        
        # Episode fails if total loss exceeds max_loss_usd
        if total_pnl < -self.max_loss_usd:
            return True
        
        return False
    
    def _get_observation(self) -> ObsType:
        """Get current observation.
        
        Returns:
            Observation dictionary
        """
        # Get raw data [seq, 6] (OHLCV + Gold)
        raw_data = np.zeros(
            (self.sequence_length, NUM_RAW_FEATURES),
            dtype=np.float32,
        )
        buffer_data = self.candle_buffer.to_array()
        if len(buffer_data) > 0:
            raw_data[-len(buffer_data):] = buffer_data
        
        # Create extended features [seq, 19]
        from src.data.preprocessor import create_features
        timestamps = self.candle_buffer.timestamps
        # Pad timestamps to match sequence length
        if len(timestamps) < self.sequence_length:
            pad_len = self.sequence_length - len(timestamps)
            first_ts = timestamps[0] if timestamps else datetime.now()
            timestamps = [first_ts] * pad_len + list(timestamps)
            
        candles = create_features(raw_data, timestamps=timestamps)
        
        # Normalize candles
        if self.normalize_obs:
            # Fit preprocessor if not fitted
            if not self.preprocessor.is_fitted:
                self.preprocessor.fit(candles)
            candles = self.preprocessor.transform(candles)
        
        # Get position info
        position_info = np.array(
            self.position_manager.get_position_info(),
            dtype=np.float32,
        )
        
        # Get account info
        total_loss = self.position_manager.get_total_loss()
        total_loss_norm = min(total_loss / self.max_loss_usd, 1.0)
        margin_norm = 1.0 - total_loss_norm  # Simplified margin calculation
        
        account_info = np.array(
            [total_loss_norm, margin_norm],
            dtype=np.float32,
        )
        
        return {
            "candles": candles,
            "position": position_info,
            "account": account_info,
        }
    
    def _get_info(
        self,
        trading_action: Action | None = None,
        prediction: np.ndarray | None = None,
        actual: np.ndarray | None = None,
        realized_pnl: float = 0.0,
        reward_components: RewardComponents | None = None,
        auto_closed: ClosedPosition | None = None,
    ) -> dict[str, Any]:
        """Get info dictionary.
        
        Args:
            trading_action: Action taken
            prediction: Prediction made
            actual: Actual candle values
            realized_pnl: Realized PnL this step
            reward_components: Reward breakdown
            auto_closed: Auto-closed position info
            
        Returns:
            Info dictionary
        """
        info: dict[str, Any] = {
            "step": self._step_count,
            "episode_reward": self._episode_reward,
            "total_pnl": self.position_manager.get_total_pnl(),
            "total_loss": self.position_manager.get_total_loss(),
            "has_position": self.position_manager.has_position(),
            "buffer_size": len(self.candle_buffer),
        }
        
        if trading_action is not None:
            info["trading_action"] = trading_action.name
        
        if prediction is not None:
            info["prediction"] = prediction.tolist()
        
        if actual is not None:
            info["actual"] = actual.tolist()
        
        if realized_pnl != 0:
            info["realized_pnl"] = realized_pnl
        
        if reward_components is not None:
            info["reward_components"] = reward_components.to_dict()
        
        if auto_closed is not None:
            info["auto_closed"] = auto_closed.to_dict()
        
        # Position details
        position = self.position_manager.get_position()
        if position is not None:
            info["position_details"] = position.to_dict()
        
        # Current candle
        if self._current_candle is not None:
            info["current_candle"] = self._current_candle.to_dict()
        
        return info
    
    def get_statistics(self) -> dict[str, Any]:
        """Get episode statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "steps": self._step_count,
            "episode_reward": self._episode_reward,
            "trading_stats": self.position_manager.get_statistics(),
        }
    
    def render(self) -> None:
        """Render the environment (placeholder)."""
        if self.render_mode == "human":
            stats = self.get_statistics()
            print(f"Step {self._step_count}: PnL={stats['trading_stats']['total_pnl']:.2f}")
    
    def close(self) -> None:
        """Clean up resources."""
        pass


class VectorizedTradingEnv:
    """Wrapper for running multiple environments in parallel.
    
    This is a simple synchronous implementation. For true parallelism,
    consider using gymnasium's AsyncVectorEnv or SubprocVectorEnv.
    """
    
    def __init__(
        self,
        num_envs: int,
        **env_kwargs: Any,
    ) -> None:
        """Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments
            **env_kwargs: Arguments passed to TradingEnvironment
        """
        self.num_envs = num_envs
        self.envs = [TradingEnvironment(**env_kwargs) for _ in range(num_envs)]
        
        # Get spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(
        self,
        seed: int | None = None,
        options: list[dict] | None = None,
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        """Reset all environments."""
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            env_options = options[i] if options else None
            env_seed = seed + i if seed else None
            obs, info = env.reset(seed=env_seed, options=env_options)
            observations.append(obs)
            infos.append(info)
        
        # Stack observations
        stacked_obs = {
            key: np.stack([o[key] for o in observations])
            for key in observations[0].keys()
        }
        
        return stacked_obs, infos
    
    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all environments."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, env in enumerate(self.envs):
            action = {
                "prediction": actions["prediction"][i],
                "trading_action": actions["trading_action"][i],
            }
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        stacked_obs = {
            key: np.stack([o[key] for o in observations])
            for key in observations[0].keys()
        }
        
        return (
            stacked_obs,
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )
    
    def receive_candles(self, candles: list[Candle | np.ndarray]) -> None:
        """Receive candles for all environments."""
        for env, candle in zip(self.envs, candles):
            env.receive_candle(candle)
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()

