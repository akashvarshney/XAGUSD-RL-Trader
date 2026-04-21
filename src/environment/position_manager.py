"""Position management for the trading environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from src.config.constants import (
    DEFAULT_LOT_SIZE,
    DEFAULT_STOP_LOSS_USD,
    DEFAULT_TAKE_PROFIT_USD,
    XAGUSD_CONTRACT_SIZE,
)


class PositionSide(str, Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


class CloseReason(str, Enum):
    """Reason for position closure."""
    MANUAL = "manual"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    LIQUIDATION = "liquidation"


@dataclass
class Position:
    """Represents an open trading position."""
    
    position_id: str
    side: PositionSide
    entry_price: float
    volume: float  # In lots
    open_time: datetime
    stop_loss_usd: float
    take_profit_usd: float
    
    # Current state
    current_price: float = field(default=0.0)
    unrealized_pnl: float = field(default=0.0)
    
    def update_price(self, price: float) -> None:
        """Update current price and recalculate PnL.
        
        Args:
            price: Current market price
        """
        self.current_price = price
        self.unrealized_pnl = self._calculate_pnl(price)
    
    def _calculate_pnl(self, price: float) -> float:
        """Calculate profit/loss at given price.
        
        For XAGUSD:
        - 1 lot = 5000 oz
        - PnL = (exit_price - entry_price) * volume * contract_size
        """
        if self.side == PositionSide.LONG:
            price_diff = price - self.entry_price
        else:
            price_diff = self.entry_price - price
        
        return price_diff * self.volume * XAGUSD_CONTRACT_SIZE
    
    def check_stop_loss(self) -> bool:
        """Check if stop loss is hit.
        
        Returns:
            True if stop loss is triggered
        """
        return self.unrealized_pnl <= -self.stop_loss_usd
    
    def check_take_profit(self) -> bool:
        """Check if take profit is hit.
        
        Returns:
            True if take profit is triggered
        """
        return self.unrealized_pnl >= self.take_profit_usd
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT
    
    @property
    def duration_seconds(self) -> float:
        """Get position duration in seconds."""
        return (datetime.now() - self.open_time).total_seconds()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "position_id": self.position_id,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "volume": self.volume,
            "open_time": self.open_time.isoformat(),
            "stop_loss_usd": self.stop_loss_usd,
            "take_profit_usd": self.take_profit_usd,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
        }


@dataclass
class ClosedPosition:
    """Record of a closed position."""
    
    position_id: str
    side: PositionSide
    entry_price: float
    exit_price: float
    volume: float
    open_time: datetime
    close_time: datetime
    realized_pnl: float
    close_reason: CloseReason
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "position_id": self.position_id,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "volume": self.volume,
            "open_time": self.open_time.isoformat(),
            "close_time": self.close_time.isoformat(),
            "realized_pnl": self.realized_pnl,
            "close_reason": self.close_reason.value,
        }


class PositionManager:
    """Manages trading positions with SL/TP logic.
    
    Features:
    - Only one position at a time
    - Automatic stop loss and take profit
    - PnL tracking and history
    """
    
    def __init__(
        self,
        lot_size: float = DEFAULT_LOT_SIZE,
        stop_loss_usd: float = DEFAULT_STOP_LOSS_USD,
        take_profit_usd: float = DEFAULT_TAKE_PROFIT_USD,
    ) -> None:
        """Initialize position manager.
        
        Args:
            lot_size: Position size in lots
            stop_loss_usd: Stop loss in USD
            take_profit_usd: Take profit in USD
        """
        self.lot_size = lot_size
        self.stop_loss_usd = stop_loss_usd
        self.take_profit_usd = take_profit_usd
        
        self._position: Position | None = None
        self._position_counter = 0
        self._total_realized_pnl = 0.0
        self._history: List[ClosedPosition] = []
    
    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self._position is not None
    
    def get_position(self) -> Position | None:
        """Get the current position."""
        return self._position
    
    def open_position(
        self,
        side: PositionSide,
        price: float,
        timestamp: datetime | None = None,
        stop_loss_usd: float | None = None,
        take_profit_usd: float | None = None,
    ) -> Position | None:
        """Open a new position.
        
        Args:
            side: Long or short
            price: Entry price
            timestamp: Open time
            
        Returns:
            The opened position, or None if position already exists
        """
        if self._position is not None:
            return None  # Can't open new position
        
        self._position_counter += 1
        timestamp = timestamp or datetime.now()
        
        self._position = Position(
            position_id=f"POS-{self._position_counter:06d}",
            side=side,
            entry_price=price,
            volume=self.lot_size,
            open_time=timestamp,
            stop_loss_usd=stop_loss_usd or self.stop_loss_usd,
            take_profit_usd=take_profit_usd or self.take_profit_usd,
            current_price=price,
            unrealized_pnl=0.0,
        )
        
        return self._position
    
    def close_position(
        self,
        price: float,
        reason: CloseReason = CloseReason.MANUAL,
        timestamp: datetime | None = None,
    ) -> ClosedPosition | None:
        """Close the current position.
        
        Args:
            price: Exit price
            reason: Reason for closing
            timestamp: Close time
            
        Returns:
            The closed position record, or None if no position
        """
        if self._position is None:
            return None
        
        timestamp = timestamp or datetime.now()
        
        # Calculate final PnL
        self._position.update_price(price)
        realized_pnl = self._position.unrealized_pnl
        
        # Create closed position record
        closed = ClosedPosition(
            position_id=self._position.position_id,
            side=self._position.side,
            entry_price=self._position.entry_price,
            exit_price=price,
            volume=self._position.volume,
            open_time=self._position.open_time,
            close_time=timestamp,
            realized_pnl=realized_pnl,
            close_reason=reason,
        )
        
        # Update totals
        self._total_realized_pnl += realized_pnl
        self._history.append(closed)
        
        # Clear position
        self._position = None
        
        return closed
    
    def update_price(self, price: float) -> tuple[bool, CloseReason | None]:
        """Update current price and check SL/TP.
        
        Args:
            price: Current market price
            
        Returns:
            Tuple of (should_close, close_reason)
        """
        if self._position is None:
            return False, None
        
        self._position.update_price(price)
        
        # Check stop loss
        if self._position.check_stop_loss():
            return True, CloseReason.STOP_LOSS
        
        # Check take profit
        if self._position.check_take_profit():
            return True, CloseReason.TAKE_PROFIT
        
        return False, None
    
    def process_price_update(
        self,
        price: float,
        timestamp: datetime | None = None,
    ) -> ClosedPosition | None:
        """Update price and auto-close if SL/TP hit.
        
        Args:
            price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Closed position if SL/TP triggered, None otherwise
        """
        should_close, reason = self.update_price(price)
        
        if should_close and reason is not None:
            return self.close_position(price, reason, timestamp)
        
        return None
    
    def get_unrealized_pnl(self) -> float:
        """Get unrealized PnL of current position."""
        if self._position is None:
            return 0.0
        return self._position.unrealized_pnl
    
    def get_total_pnl(self) -> float:
        """Get total PnL (realized + unrealized)."""
        return self._total_realized_pnl + self.get_unrealized_pnl()
    
    def get_total_loss(self) -> float:
        """Get total loss (positive value, 0 if profitable)."""
        total = self.get_total_pnl()
        return max(0, -total)
    
    def get_position_info(self) -> tuple[float, float, float]:
        """Get position info as tuple for environment state.
        
        Returns:
            Tuple of (has_position, is_long, unrealized_pnl_normalized)
        """
        if self._position is None:
            return 0.0, 0.0, 0.0
        
        has_position = 1.0
        is_long = 1.0 if self._position.is_long else -1.0
        
        # Normalize PnL to [-1, 1] range based on SL/TP
        pnl_normalized = self._position.unrealized_pnl / max(
            self.stop_loss_usd, self.take_profit_usd
        )
        pnl_normalized = max(-1.0, min(1.0, pnl_normalized))
        
        return has_position, is_long, pnl_normalized
    
    def get_history(self) -> List[ClosedPosition]:
        """Get position history."""
        return self._history.copy()
    
    def get_statistics(self) -> dict:
        """Get trading statistics."""
        if not self._history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": self._total_realized_pnl,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
            }
        
        wins = [p for p in self._history if p.realized_pnl > 0]
        losses = [p for p in self._history if p.realized_pnl <= 0]
        
        total_wins = sum(p.realized_pnl for p in wins) if wins else 0
        total_losses = abs(sum(p.realized_pnl for p in losses)) if losses else 0
        
        return {
            "total_trades": len(self._history),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self._history) if self._history else 0,
            "total_pnl": self._total_realized_pnl,
            "avg_win": total_wins / len(wins) if wins else 0,
            "avg_loss": total_losses / len(losses) if losses else 0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
        }
    
    def reset(self) -> None:
        """Reset the position manager state."""
        self._position = None
        self._total_realized_pnl = 0.0
        self._history.clear()
        # Don't reset position counter to maintain unique IDs

