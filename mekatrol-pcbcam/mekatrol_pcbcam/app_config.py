from dataclasses import dataclass, field

from .app_constants import DEFAULT_SPLASH_MINIMUM_VISIBLE_MS
from .file_locations import FileLocations
from .logging_config import LoggingConfig
from .nc_origin import DEFAULT_NC_ORIGIN
from .theme import AppTheme
from .ui_save_state import UiSaveState


@dataclass
class AppConfig:
    splash_minimum_visible_ms: int = DEFAULT_SPLASH_MINIMUM_VISIBLE_MS
    theme_file: str = "dark.yaml"
    default_nc_origin: str = DEFAULT_NC_ORIGIN
    default_file_alignment_horizontal_offset: float = 5.0
    default_file_alignment_vertical_offset: float = 5.0
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    file_locations: FileLocations = field(default_factory=FileLocations)
    ui_save_state: UiSaveState = field(default_factory=UiSaveState)
    theme: AppTheme = field(default_factory=AppTheme)
