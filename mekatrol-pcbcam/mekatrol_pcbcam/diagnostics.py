from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from threading import Lock


@dataclass(frozen=True)
class DiagnosticEntry:
    timestamp: datetime
    level_no: int
    level_name: str
    logger_name: str
    message: str


class InMemoryLogTracker:
    def __init__(self) -> None:
        self._entries: list[DiagnosticEntry] = []
        self._lock = Lock()

    def add_entry(self, entry: DiagnosticEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def entries(self) -> list[DiagnosticEntry]:
        with self._lock:
            return list(self._entries)

    def warning_entries(self) -> list[DiagnosticEntry]:
        with self._lock:
            return [entry for entry in self._entries if entry.level_no >= logging.WARNING]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


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


_log_tracker = InMemoryLogTracker()


def get_log_tracker() -> InMemoryLogTracker:
    return _log_tracker
