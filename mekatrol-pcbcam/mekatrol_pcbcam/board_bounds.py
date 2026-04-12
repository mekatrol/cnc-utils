from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class BoardBounds:
    x_min: float = math.inf
    x_max: float = -math.inf
    y_min: float = math.inf
    y_max: float = -math.inf

    @property
    def is_empty(self) -> bool:
        return not math.isfinite(self.x_min) or not math.isfinite(self.y_min)

    @property
    def width(self) -> float:
        if self.is_empty:
            return 0.0
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        if self.is_empty:
            return 0.0
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        if self.is_empty:
            return 0.0
        return (self.x_min + self.x_max) * 0.5

    @property
    def center_y(self) -> float:
        if self.is_empty:
            return 0.0
        return (self.y_min + self.y_max) * 0.5

    def include_point(self, x: float, y: float, margin: float = 0.0) -> None:
        self.x_min = min(self.x_min, x - margin)
        self.x_max = max(self.x_max, x + margin)
        self.y_min = min(self.y_min, y - margin)
        self.y_max = max(self.y_max, y + margin)

    def include_bounds(self, other: BoardBounds) -> None:
        if other.is_empty:
            return
        self.include_point(other.x_min, other.y_min)
        self.include_point(other.x_max, other.y_max)
