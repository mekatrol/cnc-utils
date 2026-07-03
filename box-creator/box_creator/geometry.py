from dataclasses import dataclass, field


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Segment:
    start: Point
    end: Point


@dataclass
class Panel:
    name: str
    width: float
    height: float
    origin_x: float
    origin_y: float
    outline: list[Point]
    relief_points: list[Point] = field(default_factory=list)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [point.x for point in self.outline]
        ys = [point.y for point in self.outline]
        return min(xs), max(xs), min(ys), max(ys)
