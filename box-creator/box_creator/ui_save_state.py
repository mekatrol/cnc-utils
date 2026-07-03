from dataclasses import dataclass


@dataclass
class UiSaveState:
    last_screen_name: str = ""
    window_state: str = "normal"
    window_x: int | None = None
    window_y: int | None = None
    window_width: int = 1180
    window_height: int = 760
