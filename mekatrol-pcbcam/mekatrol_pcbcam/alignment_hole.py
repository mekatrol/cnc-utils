from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignmentHole:
    position_mode: str = "board_xy"
    x_offset: float = 0.0
    y_offset: float = 0.0
    diameter: float = 1.0
    mirror_direction: str = "horizontal"
    enabled: bool = True
    edge: str = ""
    offset_along_edge: float = 0.0
    offset_from_edge: float = 0.0
