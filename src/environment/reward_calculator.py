"""Reward calculation for the trading environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config.constants import (
    PREDICTION_PENALTY_SCALE,
    PNL_REWARD_SCALE,
    LOSS_PENALTY_DENOMINATOR,
)


@dataclass
class RewardComponents:
    """Breakdown of reward components for logging/debugging."""
    
    prediction_penalty: float
    penalty_weight: float
    pnl_reward: float
    total_reward: float
    
    # Optional detailed metrics
    mape: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl_delta: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prediction_penalty": self.prediction_penalty,
            "penalty_weight": self.penalty_weight,
            "pnl_reward": self.pnl_reward,
            "total_reward": self.total_reward,
            "mape": self.mape,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl_delta": self.unrealized_pnl_delta,
        }


class RewardCalculator:
    """Calculate rewards for the trading environment.
    
    Reward = -prediction_penalty * penalty_weight + pnl_reward * pnl_scale
    
    Where:
    - prediction_penalty = MAPE(predicted, actual)
    - penalty_weight = 1 + (total_loss / loss_penalty_denominator)
    - pnl_reward = realized_pnl + unrealized_pnl_delta
    """
    
    def __init__(
        self,
        prediction_scale: float = PREDICTION_PENALTY_SCALE,
        pnl_scale: float = PNL_REWARD_SCALE,
        loss_penalty_denominator: float = LOSS_PENALTY_DENOMINATOR,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize reward calculator.
        
        Args:
            prediction_scale: Scale factor for prediction penalty
            pnl_scale: Scale factor for PnL reward
            loss_penalty_denominator: Denominator for loss penalty weight
            epsilon: Small value for numerical stability
        """
        self.prediction_scale = prediction_scale
        self.pnl_scale = pnl_scale
        self.loss_penalty_denominator = loss_penalty_denominator
        self.epsilon = epsilon
        
        # Track previous unrealized PnL for delta calculation
        self._prev_unrealized_pnl = 0.0
    
    def calculate_mape(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
    ) -> float:
        """Calculate Mean Absolute Percentage Error.
        
        MAPE = mean(|predicted - actual| / |actual|) * 100
        
        Args:
            predicted: Predicted values [5] (OHLCV)
            actual: Actual values [5] (OHLCV)
        Returns:
            MAPE value
        """
        if len(predicted) > len(actual):
            predicted = predicted[:len(actual)]
        elif len(actual) > len(predicted):
            actual = actual[:len(predicted)]
            
        # Higher epsilon for better numerical stability in financial data
        actual_abs = np.abs(actual) + 1e-5
        errors = np.abs(predicted - actual) / actual_abs
        mape = float(np.mean(errors) * 100)
        
        # Cap MAPE to prevent reward explosions
        return min(mape, 1000.0)
    
    def calculate_prediction_penalty(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        total_loss: float,
    ) -> tuple[float, float, float]:
        """Calculate the prediction penalty with loss weighting.
        
        Args:
            predicted: Predicted OHLCV values
            actual: Actual OHLCV values
            total_loss: Current total loss (positive value)
            
        Returns:
            Tuple of (penalty, weight, mape)
        """
        mape = self.calculate_mape(predicted, actual)
        
        # Weight increases with loss
        weight = 1.0 + (total_loss / self.loss_penalty_denominator)
        
        penalty = mape * weight * self.prediction_scale
        
        return penalty, weight, mape
    
    def calculate_pnl_reward(
        self,
        realized_pnl: float,
        current_unrealized_pnl: float,
    ) -> tuple[float, float, float]:
        """Calculate PnL-based reward.
        
        Args:
            realized_pnl: PnL from closed position (if any)
            current_unrealized_pnl: Current unrealized PnL
            
        Returns:
            Tuple of (reward, realized, delta)
        """
        # Calculate change in unrealized PnL
        unrealized_delta = current_unrealized_pnl - self._prev_unrealized_pnl
        self._prev_unrealized_pnl = current_unrealized_pnl
        
        # Total PnL reward
        pnl_reward = (realized_pnl + unrealized_delta) * self.pnl_scale
        
        return pnl_reward, realized_pnl, unrealized_delta
    
    def calculate_reward(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        total_loss: float,
        realized_pnl: float = 0.0,
        current_unrealized_pnl: float = 0.0,
    ) -> RewardComponents:
        """Calculate the total reward.
        
        Args:
            predicted: Predicted OHLCV values
            actual: Actual OHLCV values
            total_loss: Current total loss
            realized_pnl: PnL from closed position
            current_unrealized_pnl: Current unrealized PnL
            
        Returns:
            RewardComponents with full breakdown
        """
        # Prediction penalty
        pred_penalty, penalty_weight, mape = self.calculate_prediction_penalty(
            predicted, actual, total_loss
        )
        
        # PnL reward
        pnl_reward, realized, unrealized_delta = self.calculate_pnl_reward(
            realized_pnl, current_unrealized_pnl
        )
        
        # Total reward
        total_reward = -pred_penalty + pnl_reward
        
        return RewardComponents(
            prediction_penalty=pred_penalty,
            penalty_weight=penalty_weight,
            pnl_reward=pnl_reward,
            total_reward=total_reward,
            mape=mape,
            realized_pnl=realized,
            unrealized_pnl_delta=unrealized_delta,
        )
    
    def reset(self) -> None:
        """Reset internal state."""
        self._prev_unrealized_pnl = 0.0
    
    def set_unrealized_pnl(self, pnl: float) -> None:
        """Set the unrealized PnL tracking value.
        
        Useful when resuming from a saved state.
        
        Args:
            pnl: Unrealized PnL value
        """
        self._prev_unrealized_pnl = pnl


class AdaptiveRewardCalculator(RewardCalculator):
    """Adaptive reward calculator that adjusts scales based on statistics.
    
    This version tracks reward statistics and can normalize rewards
    for more stable training.
    """
    
    def __init__(
        self,
        prediction_scale: float = PREDICTION_PENALTY_SCALE,
        pnl_scale: float = PNL_REWARD_SCALE,
        loss_penalty_denominator: float = LOSS_PENALTY_DENOMINATOR,
        normalize_rewards: bool = True,
        reward_clip: float = 10.0,
        ema_alpha: float = 0.01,
    ) -> None:
        """Initialize adaptive reward calculator.
        
        Args:
            prediction_scale: Scale factor for prediction penalty
            pnl_scale: Scale factor for PnL reward
            loss_penalty_denominator: Denominator for loss penalty weight
            normalize_rewards: Whether to normalize rewards
            reward_clip: Clip rewards to [-clip, clip]
            ema_alpha: EMA smoothing factor for statistics
        """
        super().__init__(prediction_scale, pnl_scale, loss_penalty_denominator)
        
        self.normalize_rewards = normalize_rewards
        self.reward_clip = reward_clip
        self.ema_alpha = ema_alpha
        
        # Running statistics for normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0
    
    def calculate_reward(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        total_loss: float,
        realized_pnl: float = 0.0,
        current_unrealized_pnl: float = 0.0,
    ) -> RewardComponents:
        """Calculate reward with optional normalization."""
        # Get base reward
        components = super().calculate_reward(
            predicted, actual, total_loss, realized_pnl, current_unrealized_pnl
        )
        
        raw_reward = components.total_reward
        
        if self.normalize_rewards:
            # Update running statistics
            self._reward_count += 1
            delta = raw_reward - self._reward_mean
            self._reward_mean += self.ema_alpha * delta
            self._reward_var = (1 - self.ema_alpha) * (
                self._reward_var + self.ema_alpha * delta * delta
            )
            
            # Normalize
            std = np.sqrt(self._reward_var) + self.epsilon
            normalized_reward = (raw_reward - self._reward_mean) / std
            
            # Clip
            normalized_reward = np.clip(
                normalized_reward, -self.reward_clip, self.reward_clip
            )
            
            components = RewardComponents(
                prediction_penalty=components.prediction_penalty,
                penalty_weight=components.penalty_weight,
                pnl_reward=components.pnl_reward,
                total_reward=float(normalized_reward),
                mape=components.mape,
                realized_pnl=components.realized_pnl,
                unrealized_pnl_delta=components.unrealized_pnl_delta,
            )
        
        return components
    
    def get_statistics(self) -> dict:
        """Get reward statistics."""
        return {
            "mean": self._reward_mean,
            "var": self._reward_var,
            "std": np.sqrt(self._reward_var),
            "count": self._reward_count,
        }
    
    def reset(self) -> None:
        """Reset internal state."""
        super().reset()
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

