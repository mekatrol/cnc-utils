from dataclasses import dataclass, field

from .app_constants import DEFAULT_SPLASH_MINIMUM_VISIBLE_MS
from .file_locations import FileLocations
from .logging_config import LoggingConfig
from .ui_save_state import UiSaveState


@dataclass
class AppConfig:
    splash_minimum_visible_ms: int = DEFAULT_SPLASH_MINIMUM_VISIBLE_MS
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    file_locations: FileLocations = field(default_factory=FileLocations)
    ui_save_state: UiSaveState = field(default_factory=UiSaveState)
