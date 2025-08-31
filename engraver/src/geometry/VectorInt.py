from dataclasses import dataclass


@dataclass(frozen=True)
class VectorInt:
    x: int
    y: int

    def dot(self, other: "VectorInt") -> int:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "VectorInt") -> int:
        # 2D cross product (z-component)
        return self.x * other.y - self.y * other.x

    def scale(self, k: int) -> "VectorInt":
        return VectorInt(self.x * k, self.y * k)

    def __add__(self, other: "VectorInt") -> "VectorInt":
        return VectorInt(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "VectorInt") -> "VectorInt":
        return VectorInt(self.x - other.x, self.y - other.y)
