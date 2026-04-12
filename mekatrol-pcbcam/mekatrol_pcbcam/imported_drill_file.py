from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .board_bounds import BoardBounds

DrillHit = tuple[float, float, float]


@dataclass
class ImportedDrillFile:
    path: Path
    display_name: str
    holes: list[DrillHit]
    bounds: BoardBounds
