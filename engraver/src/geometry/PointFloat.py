from dataclasses import dataclass
import math
from typing import Tuple

from geometry.VectorFloat import VectorFloat


@dataclass(frozen=True, slots=True)
class PointFloat:
    x: float
    y: float
    def as_tuple(self) -> Tuple[float, float]: return (self.x, self.y)
    def __add__(self, v: VectorFloat) -> "PointFloat": return PointFloat(self.x + v.x, self.y + v.y)
    def __sub__(self, p: "PointFloat") -> VectorFloat: return VectorFloat(self.x - p.x, self.y - p.y)
    def __abs__(self) -> float: return math.hypot(self.x, self.y)
