from __future__ import annotations

import logging
from threading import Lock

from .diagnostic_entry import DiagnosticEntry


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
