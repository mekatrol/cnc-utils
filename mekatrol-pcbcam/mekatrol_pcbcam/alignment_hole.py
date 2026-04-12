from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignmentHole:
    edge: str
    offset_along_edge: float
    offset_from_edge: float
    diameter: float
