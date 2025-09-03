from dataclasses import dataclass
from typing import List, Tuple
from geometry.PolylineInt import PolylineInt


@dataclass
class GeometryInt:
    polylines: List[PolylineInt]
    scale: int = 1

    def bounds(self) -> Tuple[int, int, int, int]:
        """(minx, miny, maxx, maxy) in integer space"""
        xs = [p.x for pl in self.polylines for p in pl.points]
        ys = [p.y for pl in self.polylines for p in pl.points]
        return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else (0, 0, 0, 0)

    def simplify(self) -> None:
        for polyline in self.polylines:
            polyline.simplify()
