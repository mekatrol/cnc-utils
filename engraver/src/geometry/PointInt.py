from dataclasses import dataclass
from typing import Tuple
from geometry.VectorInt import VectorInt


@dataclass(frozen=True)
class PointInt:
    x: int
    y: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def __add__(self, other: "VectorInt") -> "PointInt":
        return PointInt(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "PointInt") -> "VectorInt":
        return VectorInt(self.x - other.x, self.y - other.y)
