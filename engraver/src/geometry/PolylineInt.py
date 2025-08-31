from typing import List
from geometry.PointInt import PointInt
from dataclasses import dataclass


@dataclass
class PolylineInt:
    """Integer-scaled polyline."""
    pts: List[PointInt]
