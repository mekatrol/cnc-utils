from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignmentHole:
    x_offset: float = 0.0
    y_offset: float = 0.0
    diameter: float = 1.0
    mirror_direction: str = "horizontal"
    enabled: bool = True
