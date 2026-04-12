from dataclasses import dataclass, field

from .app_constants import DEFAULT_SPLASH_MINIMUM_VISIBLE_MS
from .logging_config import LoggingConfig
from .ui_save_state import UiSaveState


@dataclass
class AppConfig:
    splash_minimum_visible_ms: int = DEFAULT_SPLASH_MINIMUM_VISIBLE_MS
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui_save_state: UiSaveState = field(default_factory=UiSaveState)
