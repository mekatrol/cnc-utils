from __future__ import annotations

from datetime import datetime, timezone
import logging

from .diagnostic_entry import DiagnosticEntry
from .in_memory_log_tracker import InMemoryLogTracker


class InMemoryLogHandler(logging.Handler):
    def __init__(self, tracker: InMemoryLogTracker) -> None:
        super().__init__(level=logging.NOTSET)
        self._tracker = tracker

    def emit(self, record: logging.LogRecord) -> None:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        self._tracker.add_entry(
            DiagnosticEntry(
                timestamp=timestamp,
                level_no=record.levelno,
                level_name=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
            )
        )
