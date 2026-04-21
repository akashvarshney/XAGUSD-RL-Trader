"""Candle data structures and buffer management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Sequence
from collections import deque

import numpy as np


@dataclass(slots=True)
class Candle:
    """Represents a single OHLCV candle with optional external features."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    gold_close: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert candle to numpy array [O, H, L, C, V, Gold]."""
        return np.array(
            [self.open, self.high, self.low, self.close, self.volume, self.gold_close],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray, timestamp: datetime | None = None) -> Candle:
        """Create candle from numpy array."""
        if timestamp is None:
            timestamp = datetime.now()
        
        gold_val = float(arr[5]) if len(arr) > 5 else 0.0
        return cls(
            timestamp=timestamp,
            open=float(arr[0]),
            high=float(arr[1]),
            low=float(arr[2]),
            close=float(arr[3]),
            volume=float(arr[4]),
            gold_close=gold_val
        )

    @classmethod
    def from_dict(cls, data: dict) -> Candle:
        """Create candle from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()
            
        return cls(
            timestamp=timestamp,
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data.get("volume", 0)),
            gold_close=float(data.get("gold_close", 0.0)),
        )

    def to_dict(self) -> dict:
        """Convert candle to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "gold_close": self.gold_close,
        }

    @property
    def ohlcv(self) -> tuple[float, float, float, float, float]:
        """Get OHLCV as tuple."""
        return (self.open, self.high, self.low, self.close, self.volume)

    @property
    def mid_price(self) -> float:
        """Get mid price (average of high and low)."""
        return (self.high + self.low) / 2

    @property
    def typical_price(self) -> float:
        """Get typical price (average of high, low, close)."""
        return (self.high + self.low + self.close) / 3

    @property
    def body_size(self) -> float:
        """Get candle body size (absolute difference between open and close)."""
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        """Get candle range (high - low)."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (close > open)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (close < open)."""
        return self.close < self.open


@dataclass
class CandleBuffer:
    """Rolling buffer for storing candles with fixed maximum size.
    
    This buffer maintains a fixed-size window of the most recent candles,
    automatically discarding old candles when new ones are added.
    """

    max_size: int
    _candles: deque[Candle] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        """Initialize the deque with max length."""
        self._candles = deque(maxlen=self.max_size)

    def add(self, candle: Candle) -> None:
        """Add a candle to the buffer."""
        self._candles.append(candle)

    def add_many(self, candles: Sequence[Candle]) -> None:
        """Add multiple candles to the buffer."""
        for candle in candles:
            self._candles.append(candle)

    def clear(self) -> None:
        """Clear all candles from the buffer."""
        self._candles.clear()

    def is_full(self) -> bool:
        """Check if buffer has reached max size."""
        return len(self._candles) == self.max_size

    def to_array(self) -> np.ndarray:
        """Convert all candles to numpy array of shape [N, 6]."""
        if not self._candles:
            return np.empty((0, 6), dtype=np.float32)
        return np.array([c.to_array() for c in self._candles], dtype=np.float32)

    def get_latest(self, n: int = 1) -> list[Candle]:
        """Get the n most recent candles."""
        return list(self._candles)[-n:]

    def get_prices(self, price_type: str = "close") -> np.ndarray:
        """Get array of prices of specified type."""
        if not self._candles:
            return np.empty(0, dtype=np.float32)
        
        price_map = {
            "open": lambda c: c.open,
            "high": lambda c: c.high,
            "low": lambda c: c.low,
            "close": lambda c: c.close,
            "typical": lambda c: c.typical_price,
            "mid": lambda c: c.mid_price,
        }
        
        getter = price_map.get(price_type, price_map["close"])
        return np.array([getter(c) for c in self._candles], dtype=np.float32)

    def get_volumes(self) -> np.ndarray:
        """Get array of volumes."""
        if not self._candles:
            return np.empty(0, dtype=np.float32)
        return np.array([c.volume for c in self._candles], dtype=np.float32)

    @property
    def timestamps(self) -> list[datetime]:
        """Get list of timestamps."""
        return [c.timestamp for c in self._candles]

    @property
    def latest(self) -> Candle | None:
        """Get the most recent candle."""
        return self._candles[-1] if self._candles else None

    @property
    def oldest(self) -> Candle | None:
        """Get the oldest candle in buffer."""
        return self._candles[0] if self._candles else None

    def __len__(self) -> int:
        """Return number of candles in buffer."""
        return len(self._candles)

    def __iter__(self) -> Iterator[Candle]:
        """Iterate over candles."""
        return iter(self._candles)

    def __getitem__(self, idx: int) -> Candle:
        """Get candle by index."""
        return self._candles[idx]

    def __repr__(self) -> str:
        """String representation."""
        return f"CandleBuffer(size={len(self)}/{self.max_size})"

