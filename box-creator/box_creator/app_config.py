from dataclasses import dataclass, field

from .file_locations import FileLocations
from .ui_save_state import UiSaveState


@dataclass
class AppConfig:
    file_locations: FileLocations = field(default_factory=FileLocations)
    ui_save_state: UiSaveState = field(default_factory=UiSaveState)
