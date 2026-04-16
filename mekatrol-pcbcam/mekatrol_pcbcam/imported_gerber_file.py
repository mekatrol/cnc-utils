from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .board_bounds import BoardBounds

TraceSegment = tuple[tuple[float, float], tuple[float, float], float]
LineSegment = tuple[tuple[float, float], tuple[float, float]]
PadDefinition = dict[str, float | str]
PadFlash = tuple[tuple[float, float], PadDefinition]
Polygon = list[tuple[float, float]]


@dataclass
class ImportedGerberFile:
    path: Path
    display_name: str
    traces: list[TraceSegment]
    segments: list[LineSegment]
    pads: list[PadFlash]
    regions: list[Polygon]
    outline: Polygon
    bounds: BoardBounds

    @property
    def has_visible_geometry(self) -> bool:
        return bool(self.traces or self.segments or self.pads or self.regions or self.outline)
