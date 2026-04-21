"""Rollout buffer for storing experience during PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, NamedTuple

import numpy as np
import torch


class RolloutBatch(NamedTuple):
    """A batch of rollout data for training."""
    
    candles: torch.Tensor          # [batch, seq_len, N]
    position_info: torch.Tensor    # [batch, 3]
    account_info: torch.Tensor     # [batch, 2]
    predictions: torch.Tensor      # [batch, N]
    trading_actions: torch.Tensor  # [batch]
    old_pred_log_probs: torch.Tensor  # [batch]
    old_action_log_probs: torch.Tensor  # [batch]
    advantages: torch.Tensor       # [batch]
    returns: torch.Tensor          # [batch]
    values: torch.Tensor           # [batch]


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data during PPO training.
    
    Stores transitions and computes advantages using GAE.
    Supports mini-batch iteration for PPO updates.
    """
    
    buffer_size: int
    sequence_length: int
    num_ohlcv_features: int = 5
    num_position_features: int = 3
    num_account_features: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    device: str = "cpu"
    
    # Storage arrays
    candles: np.ndarray = field(init=False)
    position_info: np.ndarray = field(init=False)
    account_info: np.ndarray = field(init=False)
    predictions: np.ndarray = field(init=False)
    trading_actions: np.ndarray = field(init=False)
    pred_log_probs: np.ndarray = field(init=False)
    action_log_probs: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    
    # Computed values
    advantages: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)
    
    # State tracking
    _ptr: int = field(default=0, init=False)
    _full: bool = field(default=False, init=False)
    _advantages_computed: bool = field(default=False, init=False)
    
    def __post_init__(self) -> None:
        """Initialize storage arrays."""
        self._allocate_storage()
    
    def _allocate_storage(self) -> None:
        """Allocate numpy arrays for storage."""
        self.candles = np.zeros(
            (self.buffer_size, self.sequence_length, self.num_ohlcv_features),
            dtype=np.float32,
        )
        self.position_info = np.zeros(
            (self.buffer_size, self.num_position_features),
            dtype=np.float32,
        )
        self.account_info = np.zeros(
            (self.buffer_size, self.num_account_features),
            dtype=np.float32,
        )
        self.predictions = np.zeros(
            (self.buffer_size, self.num_ohlcv_features),
            dtype=np.float32,
        )
        self.trading_actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.pred_log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.action_log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
    
    def add(
        self,
        candles: np.ndarray,
        position_info: np.ndarray,
        account_info: np.ndarray,
        prediction: np.ndarray,
        trading_action: int,
        pred_log_prob: float,
        action_log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.
        
        Args:
            candles: Candle sequence [seq_len, N]
            position_info: Position features [3]
            account_info: Account features [2]
            prediction: Predicted candle [N]
            trading_action: Discrete action index
            pred_log_prob: Log probability of prediction
            action_log_prob: Log probability of action
            reward: Reward received
            value: Value estimate
            done: Whether episode ended
        """
        self.candles[self._ptr] = candles
        self.position_info[self._ptr] = position_info
        self.account_info[self._ptr] = account_info
        self.predictions[self._ptr] = prediction
        self.trading_actions[self._ptr] = trading_action
        self.pred_log_probs[self._ptr] = pred_log_prob
        self.action_log_probs[self._ptr] = action_log_prob
        self.rewards[self._ptr] = reward
        self.values[self._ptr] = value
        self.dones[self._ptr] = float(done)
        
        self._ptr += 1
        if self._ptr >= self.buffer_size:
            self._full = True
            self._ptr = 0
        
        self._advantages_computed = False
    
    def compute_advantages(self, last_value: float) -> None:
        """Compute advantages using Generalized Advantage Estimation (GAE).
        
        GAE: A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        
        Args:
            last_value: Value estimate for the last state
        """
        size = self.buffer_size if self._full else self._ptr
        
        last_gae = 0.0
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            # TD error
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            
            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        # Compute returns (advantages + values)
        self.returns[:size] = self.advantages[:size] + self.values[:size]
        
        self._advantages_computed = True
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[RolloutBatch, None, None]:
        """Generate mini-batches for training.
        
        Args:
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle indices
            
        Yields:
            RolloutBatch instances
        """
        if not self._advantages_computed:
            raise RuntimeError("Must call compute_advantages() before get_batches()")
        
        size = self.buffer_size if self._full else self._ptr
        indices = np.arange(size)
        
        if shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_indices = indices[start:end]
            
            yield self._get_batch(batch_indices)
    
    def _get_batch(self, indices: np.ndarray) -> RolloutBatch:
        """Create a batch from given indices.
        
        Args:
            indices: Array of indices to include
            
        Returns:
            RolloutBatch with tensors
        """
        device = torch.device(self.device)
        
        return RolloutBatch(
            candles=torch.tensor(self.candles[indices], device=device),
            position_info=torch.tensor(self.position_info[indices], device=device),
            account_info=torch.tensor(self.account_info[indices], device=device),
            predictions=torch.tensor(self.predictions[indices], device=device),
            trading_actions=torch.tensor(self.trading_actions[indices], device=device),
            old_pred_log_probs=torch.tensor(self.pred_log_probs[indices], device=device),
            old_action_log_probs=torch.tensor(self.action_log_probs[indices], device=device),
            advantages=torch.tensor(self.advantages[indices], device=device),
            returns=torch.tensor(self.returns[indices], device=device),
            values=torch.tensor(self.values[indices], device=device),
        )
    
    def get_all(self) -> RolloutBatch:
        """Get all data as a single batch.
        
        Returns:
            RolloutBatch with all data
        """
        if not self._advantages_computed:
            raise RuntimeError("Must call compute_advantages() before get_all()")
        
        size = self.buffer_size if self._full else self._ptr
        indices = np.arange(size)
        return self._get_batch(indices)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._ptr = 0
        self._full = False
        self._advantages_computed = False
    
    def normalize_advantages(self) -> None:
        """Normalize advantages to have zero mean and unit variance."""
        if not self._advantages_computed:
            return
        
        size = self.buffer_size if self._full else self._ptr
        advantages = self.advantages[:size]
        
        mean = np.mean(advantages)
        std = np.std(advantages) + 1e-8
        
        self.advantages[:size] = (advantages - mean) / std
    
    @property
    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer_size if self._full else self._ptr
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._full
    
    def get_statistics(self) -> dict:
        """Get buffer statistics for logging."""
        size = self.size
        if size == 0:
            return {}
        
        return {
            "buffer_size": size,
            "mean_reward": float(np.mean(self.rewards[:size])),
            "std_reward": float(np.std(self.rewards[:size])),
            "mean_value": float(np.mean(self.values[:size])),
            "mean_advantage": float(np.mean(self.advantages[:size])) if self._advantages_computed else 0,
            "done_fraction": float(np.mean(self.dones[:size])),
        }

