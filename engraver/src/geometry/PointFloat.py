from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PointFloat:
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __add__(self, other: "PointFloat") -> "PointFloat":
        return PointFloat(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "PointFloat") -> "PointFloat":
        return PointFloat(self.x - other.x, self.y - other.y)
