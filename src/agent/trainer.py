"""Training orchestrator for PPO agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from src.agent.ppo_agent import PPOAgent, PPOConfig, UpdateResult
from src.agent.rollout_buffer import RolloutBuffer
from src.data.candle import Candle
from src.data.csv_loader import CSVDataLoader, CSVDataset
from src.environment.trading_env import TradingEnvironment
from src.models.actor_critic import HybridActorCritic
from src.utils.checkpoint import CheckpointManager
from src.utils.metrics import MetricsWriter
from src.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    total_timesteps: int = 1_000_000
    rollout_steps: int = 2048
    batch_size: int = 64
    num_epochs: int = 10
    
    # Checkpointing
    checkpoint_interval_minutes: int = 360  # 6 hours
    checkpoint_dir: Path = Path("./checkpoints")
    
    # Logging
    log_interval: int = 10  # Log every N rollouts
    tensorboard_dir: Path = Path("./logs/tensorboard")
    
    # Early stopping
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.001
    
    # Learning rate scheduling
    lr_schedule: str = "constant"  # "constant", "linear", "cosine"
    
    # Curriculum learning
    use_curriculum: bool = False
    curriculum_stages: list[dict] = field(default_factory=list)


@dataclass
class TrainingState:
    """Current training state."""
    
    timesteps: int = 0
    rollouts: int = 0
    episodes: int = 0
    updates: int = 0
    
    best_reward: float = float("-inf")
    best_reward_timestep: int = 0
    
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    
    start_time: datetime = field(default_factory=datetime.now)
    last_checkpoint_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timesteps": self.timesteps,
            "rollouts": self.rollouts,
            "episodes": self.episodes,
            "updates": self.updates,
            "best_reward": self.best_reward,
            "best_reward_timestep": self.best_reward_timestep,
            "start_time": self.start_time.isoformat(),
        }


class Trainer:
    """Training orchestrator for PPO agent.
    
    Handles:
    - Rollout collection
    - Policy updates
    - Logging and metrics
    - Checkpointing
    - Early stopping
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        env: TradingEnvironment,
        config: TrainingConfig | None = None,
    ) -> None:
        """Initialize the trainer.
        
        Args:
            agent: PPO agent to train
            env: Trading environment
            config: Training configuration
        """
        self.agent = agent
        self.env = env
        self.config = config or TrainingConfig()
        
        # State
        self.state = TrainingState()
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.config.rollout_steps,
            sequence_length=env.sequence_length,
            num_ohlcv_features=env.input_dim,
            gamma=agent.config.gamma,
            gae_lambda=agent.config.gae_lambda,
            device=str(agent.device),
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
        )
        
        # Metrics writer
        self.metrics_writer = MetricsWriter(
            log_dir=self.config.tensorboard_dir,
        )
        
        # Training control
        self._should_stop = False
        self._pause = False
        
        # Callbacks
        self._callbacks: list[Callable[[TrainingState], None]] = []
    
    def train(
        self,
        data_source: CSVDataLoader | Iterator[Candle],
        callback: Callable[[TrainingState], None] | None = None,
    ) -> TrainingState:
        """Train the agent.
        
        Args:
            data_source: Source of candle data
            callback: Optional callback called after each rollout
            
        Returns:
            Final training state
        """
        logger.info("Starting training", total_timesteps=self.config.total_timesteps)
        
        if callback:
            self._callbacks.append(callback)
        
        try:
            self._training_loop(data_source)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            self._cleanup()
        
        return self.state
    
    def _training_loop(self, data_source: CSVDataLoader | Iterator[Candle]) -> None:
        """Main training loop."""
        # Setup data iteration
        if isinstance(data_source, CSVDataLoader):
            data_iter = self._create_data_iterator(data_source)
        else:
            data_iter = data_source
        
        # Reset environment with initial data
        initial_candles = self._get_initial_candles(data_iter)
        obs, info = self.env.reset(options={"initial_candles": initial_candles})
        self.agent.reset_hidden()
        
        # Episode tracking
        episode_reward = 0.0
        episode_length = 0
        
        while self.state.timesteps < self.config.total_timesteps and not self._should_stop:
            if self._pause:
                time.sleep(0.1)
                continue
            
            # Collect rollout
            rollout_start = time.time()
            
            for _ in range(self.config.rollout_steps):
                # Get new candle
                try:
                    candle = next(data_iter)
                except StopIteration:
                    # Reset iterator for continuous training
                    if isinstance(data_source, CSVDataLoader):
                        data_iter = self._create_data_iterator(data_source)
                        candle = next(data_iter)
                    else:
                        break
                
                self.env.receive_candle(candle)
                
                # Get action from agent
                action, action_info = self.agent.get_action(obs, deterministic=False)
                
                # Create action dict for environment
                env_action = {
                    "prediction": action_info["prediction"],
                    "trading_action": action_info["trading_action"],
                }
                
                # Step environment
                next_obs, reward, terminated, truncated, step_info = self.env.step(env_action)
                done = terminated or truncated
                
                # Store transition
                self.buffer.add(
                    candles=obs["candles"],
                    position_info=obs["position"],
                    account_info=obs["account"],
                    prediction=action_info["prediction"],
                    trading_action=action_info["trading_action"],
                    pred_log_prob=action_info["pred_log_prob"],
                    action_log_prob=action_info["action_log_prob"],
                    reward=reward,
                    value=action_info["value"],
                    done=done,
                )
                
                # Update tracking
                episode_reward += reward
                episode_length += 1
                self.state.timesteps += 1
                
                # Handle episode end
                if done:
                    self.state.episodes += 1
                    self.state.episode_rewards.append(episode_reward)
                    self.state.episode_lengths.append(episode_length)
                    
                    # Check for best reward
                    if episode_reward > self.state.best_reward:
                        self.state.best_reward = episode_reward
                        self.state.best_reward_timestep = self.state.timesteps
                        self.checkpoint_manager.save_best(
                            self.agent.model,
                            self.agent.optimizer,
                            self.state.to_dict(),
                        )
                    
                    # Log episode
                    logger.info(
                        "Episode completed",
                        episode=self.state.episodes,
                        reward=episode_reward,
                        length=episode_length,
                        total_pnl=step_info.get("total_pnl", 0),
                    )
                    
                    # Reset for new episode
                    initial_candles = self._get_initial_candles(data_iter)
                    obs, info = self.env.reset(options={"initial_candles": initial_candles})
                    self.agent.reset_hidden()
                    episode_reward = 0.0
                    episode_length = 0
                else:
                    obs = next_obs
                    self.agent.detach_hidden()
            
            # Compute advantages
            _, final_info = self.agent.get_action(obs, deterministic=True)
            self.buffer.compute_advantages(final_info["value"])
            
            # Update policy
            update_result = self.agent.update(self.buffer)
            self.state.updates += 1
            self.state.rollouts += 1
            
            rollout_time = time.time() - rollout_start
            
            # Log metrics
            self._log_metrics(update_result, rollout_time)
            
            # Clear buffer for next rollout
            self.buffer.clear()
            
            # Checkpointing
            self._maybe_checkpoint()
            
            # Run callbacks
            for cb in self._callbacks:
                cb(self.state)
    
    def _create_data_iterator(self, loader: CSVDataLoader) -> Iterator[Candle]:
        """Create an iterator over candle data."""
        if not loader.is_loaded:
            loader.load()
        return iter(loader.candles)
    
    def _get_initial_candles(
        self,
        data_iter: Iterator[Candle],
        count: int | None = None,
    ) -> list[Candle]:
        """Get initial candles for environment reset."""
        count = count or self.env.sequence_length
        candles = []
        for _ in range(count):
            try:
                candles.append(next(data_iter))
            except StopIteration:
                break
        return candles
    
    def _log_metrics(self, update_result: UpdateResult, rollout_time: float) -> None:
        """Log training metrics."""
        step = self.state.timesteps
        
        # Training metrics
        self.metrics_writer.add_scalar("train/policy_loss", update_result.policy_loss, step)
        self.metrics_writer.add_scalar("train/value_loss", update_result.value_loss, step)
        self.metrics_writer.add_scalar("train/prediction_loss", update_result.prediction_loss, step)
        self.metrics_writer.add_scalar("train/entropy", update_result.entropy, step)
        self.metrics_writer.add_scalar("train/approx_kl", update_result.approx_kl, step)
        self.metrics_writer.add_scalar("train/clip_fraction", update_result.clip_fraction, step)
        self.metrics_writer.add_scalar("train/explained_variance", update_result.explained_variance, step)
        
        # Episode metrics
        if self.state.episode_rewards:
            recent_rewards = self.state.episode_rewards[-100:]
            self.metrics_writer.add_scalar("episode/mean_reward", np.mean(recent_rewards), step)
            self.metrics_writer.add_scalar("episode/std_reward", np.std(recent_rewards), step)
        
        if self.state.episode_lengths:
            recent_lengths = self.state.episode_lengths[-100:]
            self.metrics_writer.add_scalar("episode/mean_length", np.mean(recent_lengths), step)
        
        # Performance metrics
        fps = self.config.rollout_steps / rollout_time
        self.metrics_writer.add_scalar("performance/fps", fps, step)
        self.metrics_writer.add_scalar("performance/rollout_time", rollout_time, step)
        
        # Log to console periodically
        if self.state.rollouts % self.config.log_interval == 0:
            logger.info(
                "Training progress",
                timesteps=self.state.timesteps,
                rollouts=self.state.rollouts,
                episodes=self.state.episodes,
                mean_reward=np.mean(self.state.episode_rewards[-100:]) if self.state.episode_rewards else 0,
                fps=fps,
            )
    
    def _maybe_checkpoint(self) -> None:
        """Save checkpoint if interval has passed."""
        now = datetime.now()
        elapsed = now - self.state.last_checkpoint_time
        
        if elapsed >= timedelta(minutes=self.config.checkpoint_interval_minutes):
            self.checkpoint_manager.save(
                self.agent.model,
                self.agent.optimizer,
                self.state.to_dict(),
                epoch=self.state.rollouts,
            )
            self.state.last_checkpoint_time = now
            logger.info("Checkpoint saved", rollouts=self.state.rollouts)
    
    def _cleanup(self) -> None:
        """Cleanup after training."""
        # Save final checkpoint
        self.checkpoint_manager.save(
            self.agent.model,
            self.agent.optimizer,
            self.state.to_dict(),
            epoch=self.state.rollouts,
            is_final=True,
        )
        
        # Close metrics writer
        self.metrics_writer.close()
        
        logger.info(
            "Training completed",
            total_timesteps=self.state.timesteps,
            total_episodes=self.state.episodes,
            best_reward=self.state.best_reward,
        )
    
    def stop(self) -> None:
        """Stop training."""
        self._should_stop = True
    
    def pause(self) -> None:
        """Pause training."""
        self._pause = True
    
    def resume(self) -> None:
        """Resume training."""
        self._pause = False
    
    @property
    def is_running(self) -> bool:
        """Check if training is running."""
        return not self._should_stop and not self._pause
    
    def get_progress(self) -> dict:
        """Get training progress."""
        return {
            "timesteps": self.state.timesteps,
            "total_timesteps": self.config.total_timesteps,
            "progress": self.state.timesteps / self.config.total_timesteps,
            "episodes": self.state.episodes,
            "rollouts": self.state.rollouts,
            "updates": self.state.updates,
            "best_reward": self.state.best_reward,
            "is_running": self.is_running,
        }

