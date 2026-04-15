from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DiagnosticEntry:
    timestamp: datetime
    level_no: int
    level_name: str
    logger_name: str
    message: str
