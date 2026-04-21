"""News filtering logic for high-impact eco events."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class NewsEvent:
    """Represents a scheduled economic event."""
    def __init__(self, title: str, timestamp: datetime, impact: str):
        self.title = title
        self.timestamp = timestamp
        self.impact = impact  # "High", "Medium", "Low"

class NewsFilter:
    """Filters trading during high-impact news events."""

    def __init__(self, observation_window_minutes: int = 60):
        self.observation_window = observation_window_minutes
        self.mock_events: List[NewsEvent] = []

    def load_mock_events(self):
        """Simulate high-impact events like Fed meetings or CPI."""
        # For demonstration purposes
        now = datetime.now()
        self.mock_events = [
            NewsEvent("FOMC Interest Rate Decision", now + timedelta(hours=2), "High"),
            NewsEvent("US Core CPI m/m", now + timedelta(days=1), "High"),
            NewsEvent("Non-Farm Employment Change", now + timedelta(days=5), "High")
        ]

    def is_safe_to_trade(self, current_time: datetime | None = None) -> bool:
        """Check if any high-impact news is coming up.
        
        Args:
            current_time: The time to check against.
            
        Returns:
            False if a High-Impact event is within the observation window.
        """
        if not current_time:
            current_time = datetime.now()
            
        # Ensure current_time is naive for comparison if events are naive
        if current_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=None)

        for event in self.mock_events:
            time_until = event.timestamp - current_time
            if event.impact == "High" and timedelta(0) <= time_until <= timedelta(minutes=self.observation_window):
                logger.warning(
                    f"TRADING PAUSED: High impact news detected: {event.title} in {time_until}"
                )
                return False
        
        return True

    def get_event_at(self, current_time: datetime) -> NewsEvent | None:
        """Find news at a specific time (useful for backtesting)."""
        for event in self.mock_events:
            if abs(event.timestamp - current_time) < timedelta(minutes=15):
                return event
        return None
