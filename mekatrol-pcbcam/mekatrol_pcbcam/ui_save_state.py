from dataclasses import dataclass

from .app_constants import DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH


@dataclass
class UiSaveState:
    last_screen_name: str = ""
    window_state: str = "normal"
    window_x: int | None = None
    window_y: int | None = None
    window_width: int = DEFAULT_WINDOW_WIDTH
    window_height: int = DEFAULT_WINDOW_HEIGHT
