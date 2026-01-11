from dataclasses import dataclass, field
from typing import List, Tuple
from geometry.PointInt import PointInt
from geometry.PolylineInt import PolylineInt


@dataclass
class GeometryInt:
    polylines: List[PolylineInt]
    points: List[PointInt]
    scale: int = 1
    degenerate_polylines: List[PolylineInt] = field(default_factory=list)

    def bounds(self) -> Tuple[int, int, int, int]:
        # Get super set of all points
        points = [p for pl in self.polylines for p in pl.points] + self.points

        # No points then return zero bounds
        if not points:
            return (0, 0, 0, 0)

        # Get independent x set and y set values
        x_set = [p.x for p in points]
        y_set = [p.y for p in points]

        # Return min/max of X and Y
        return (min(x_set), min(y_set), max(x_set), max(y_set))

    def simplify(self) -> None:
        for polyline in self.polylines:
            polyline.simplify()
