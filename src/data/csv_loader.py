"""CSV data loader for historical candle data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.data.candle import Candle, CandleBuffer
from src.config.constants import OHLCV_COLUMNS


class CSVDataLoader:
    """Load and iterate over historical candle data from CSV files.
    
    Expected CSV format:
        timestamp,open,high,low,close,volume
        2024-01-01 00:00:00,30.123,30.150,30.100,30.145,1234
        ...
    
    The loader supports:
    - Lazy loading with chunked iteration
    - Window-based iteration for sequence models
    - Data validation and cleaning
    """

    def __init__(
        self,
        file_path: str | Path,
        timestamp_column: str = "timestamp",
        timestamp_format: str | None = None,
    ) -> None:
        """Initialize the CSV loader.
        
        Args:
            file_path: Path to the CSV file
            timestamp_column: Name of the timestamp column
            timestamp_format: Optional strftime format for parsing timestamps
        """
        self.file_path = Path(file_path)
        self.timestamp_column = timestamp_column
        self.timestamp_format = timestamp_format
        
        self._df: pd.DataFrame | None = None
        self._candles: list[Candle] | None = None

    def load(self, validate: bool = True) -> None:
        """Load the entire CSV into memory.
        
        Args:
            validate: Whether to validate and clean the data
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        # Read CSV
        self._df = pd.read_csv(self.file_path)

        # Parse timestamp
        if self.timestamp_column in self._df.columns:
            if self.timestamp_format:
                self._df[self.timestamp_column] = pd.to_datetime(
                    self._df[self.timestamp_column],
                    format=self.timestamp_format,
                )
            else:
                self._df[self.timestamp_column] = pd.to_datetime(
                    self._df[self.timestamp_column]
                )
            # Sort by timestamp
            self._df = self._df.sort_values(self.timestamp_column).reset_index(drop=True)

        if validate:
            self._validate_and_clean()

        # Convert to candles
        self._candles = self._dataframe_to_candles(self._df)

    def _validate_and_clean(self) -> None:
        """Validate and clean the loaded data."""
        if self._df is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        # Check required columns
        required_columns = OHLCV_COLUMNS
        missing = set(required_columns) - set(self._df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Remove rows with NaN values in OHLCV columns
        original_len = len(self._df)
        self._df = self._df.dropna(subset=required_columns)
        dropped = original_len - len(self._df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with NaN values")

        # Validate OHLC relationships
        invalid_mask = (
            (self._df["high"] < self._df["low"]) |
            (self._df["high"] < self._df["open"]) |
            (self._df["high"] < self._df["close"]) |
            (self._df["low"] > self._df["open"]) |
            (self._df["low"] > self._df["close"])
        )
        
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            print(f"Warning: {invalid_count} rows have invalid OHLC relationships")
            # Fix by adjusting high/low
            self._df.loc[invalid_mask, "high"] = self._df.loc[
                invalid_mask, ["open", "high", "low", "close"]
            ].max(axis=1)
            self._df.loc[invalid_mask, "low"] = self._df.loc[
                invalid_mask, ["open", "high", "low", "close"]
            ].min(axis=1)

        # Ensure positive volumes
        self._df["volume"] = self._df["volume"].clip(lower=0)

        # Ensure numeric types
        for col in required_columns:
            self._df[col] = pd.to_numeric(self._df[col], errors="coerce")

    def _dataframe_to_candles(self, df: pd.DataFrame) -> list[Candle]:
        """Convert DataFrame to list of Candle objects."""
        candles = []
        for _, row in df.iterrows():
            timestamp = row.get(self.timestamp_column)
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            elif timestamp is None:
                timestamp = datetime.now()
                
            candle = Candle(
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                gold_close=float(row.get("gold_close", 0.0)),
            )
            candles.append(candle)
        return candles

    def iter_candles(self) -> Iterator[Candle]:
        """Iterate over all candles."""
        if self._candles is None:
            self.load()
        yield from self._candles  # type: ignore

    def iter_windows(
        self,
        window_size: int,
        stride: int = 1,
        drop_last: bool = False,
    ) -> Iterator[tuple[list[Candle], Candle]]:
        """Iterate over sliding windows of candles.
        
        Yields tuples of (window_candles, next_candle) for training.
        
        Args:
            window_size: Number of candles in each window
            stride: Step size between windows
            drop_last: Whether to drop the last incomplete window
            
        Yields:
            Tuple of (list of window candles, next candle after window)
        """
        if self._candles is None:
            self.load()
        
        candles = self._candles  # type: ignore
        n = len(candles)
        
        # Need at least window_size + 1 candles (window + next)
        if n < window_size + 1:
            raise ValueError(
                f"Not enough candles ({n}) for window size {window_size}"
            )

        for i in range(0, n - window_size, stride):
            window = candles[i : i + window_size]
            next_candle = candles[i + window_size]
            yield window, next_candle

    def iter_arrays(
        self,
        window_size: int,
        stride: int = 1,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over sliding windows as numpy arrays.
        
        Args:
            window_size: Number of candles in each window
            stride: Step size between windows
            
        Yields:
            Tuple of (window array [window_size, 5], next candle array [5])
        """
        for window, next_candle in self.iter_windows(window_size, stride):
            window_arr = np.array([c.to_array() for c in window], dtype=np.float32)
            next_arr = next_candle.to_array()
            yield window_arr, next_arr

    def to_numpy(self) -> np.ndarray:
        """Convert all data to numpy array of shape [N, 5]."""
        if self._candles is None:
            self.load()
        return np.array([c.to_array() for c in self._candles], dtype=np.float32)  # type: ignore

    def to_dataframe(self) -> pd.DataFrame:
        """Get the loaded DataFrame."""
        if self._df is None:
            self.load()
        return self._df.copy()  # type: ignore

    def fill_buffer(self, buffer: CandleBuffer, start_idx: int = 0) -> int:
        """Fill a CandleBuffer starting from given index.
        
        Args:
            buffer: The buffer to fill
            start_idx: Starting index in the candle list
            
        Returns:
            Index of the next candle after filling the buffer
        """
        if self._candles is None:
            self.load()
        
        candles = self._candles  # type: ignore
        end_idx = min(start_idx + buffer.max_size, len(candles))
        
        buffer.clear()
        buffer.add_many(candles[start_idx:end_idx])
        
        return end_idx

    def __len__(self) -> int:
        """Return total number of candles."""
        if self._candles is None:
            self.load()
        return len(self._candles)  # type: ignore

    def __getitem__(self, idx: int) -> Candle:
        """Get candle by index."""
        if self._candles is None:
            self.load()
        return self._candles[idx]  # type: ignore

    @property
    def candles(self) -> list[Candle]:
        """Get all candles."""
        if self._candles is None:
            self.load()
        return self._candles  # type: ignore

    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._candles is not None


class CSVDataset:
    """PyTorch-compatible dataset wrapper for CSV candle data.
    
    Provides indexed access to (window, next_candle) pairs for training.
    """

    def __init__(
        self,
        loader: CSVDataLoader,
        window_size: int,
        stride: int = 1,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            loader: CSVDataLoader instance with loaded data
            window_size: Size of the observation window
            stride: Stride between consecutive samples
        """
        self.loader = loader
        self.window_size = window_size
        self.stride = stride
        
        if not loader.is_loaded:
            loader.load()
        
        # Pre-compute valid indices
        n_candles = len(loader)
        self._indices = list(range(0, n_candles - window_size, stride))

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a sample by index.
        
        Returns:
            Tuple of (window array [window_size, 5], next candle array [5])
        """
        start_idx = self._indices[idx]
        candles = self.loader.candles
        
        # Get window
        window = np.array(
            [candles[i].to_array() for i in range(start_idx, start_idx + self.window_size)],
            dtype=np.float32,
        )
        
        # Get next candle
        next_candle = candles[start_idx + self.window_size].to_array()
        
        return window, next_candle

    def get_candle_at(self, idx: int) -> Candle:
        """Get the target candle for a sample index."""
        start_idx = self._indices[idx]
        return self.loader.candles[start_idx + self.window_size]

