from dataclasses import dataclass, field

from .app_constants import (
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_MAX_BYTES,
)


@dataclass
class LoggingConfig:
    level: str = DEFAULT_LOG_LEVEL
    path: str = ""
    max_bytes: int = DEFAULT_LOG_MAX_BYTES
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT
    loggers: dict[str, str] = field(default_factory=dict)
