from __future__ import annotations

from .in_memory_log_handler import InMemoryLogHandler
from .in_memory_log_tracker import InMemoryLogTracker


_log_tracker = InMemoryLogTracker()


def get_log_tracker() -> InMemoryLogTracker:
    return _log_tracker
