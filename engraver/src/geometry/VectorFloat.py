from dataclasses import dataclass
import math
from typing import Tuple


@dataclass(frozen=True, slots=True)
class VectorFloat:
    x: float
    y: float
    def as_tuple(self) -> Tuple[float, float]: return (self.x, self.y)
    def __add__(self, o: "VectorFloat") -> "VectorFloat": return VectorFloat(self.x + o.x, self.y + o.y)
    def __sub__(self, o: "VectorFloat") -> "VectorFloat": return VectorFloat(self.x - o.x, self.y - o.y)
    def __mul__(self, k: float) -> "VectorFloat": return VectorFloat(self.x * k, self.y * k)
    __rmul__ = __mul__
    def dot(self, o: "VectorFloat") -> float: return self.x * o.x + self.y * o.y
    def cross(self, o: "VectorFloat") -> float: return self.x * o.y - self.y * o.x
    def __abs__(self) -> float: return math.hypot(self.x, self.y)
    def scale(self, k: float) -> "VectorFloat": return VectorFloat(self.x * k, self.y * k)
