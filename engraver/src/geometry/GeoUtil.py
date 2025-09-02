import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from geometry.GeometryInt import GeometryInt
from geometry.PointFloat import PointFloat
from geometry.PointInPolygonResult import PointInPolygonResult
from geometry.PointInt import PointInt


@dataclass(frozen=True)
class GeoUtil:
    @staticmethod
    def float_to_int(x: float, scale: int) -> int:
        """Convert floating point to scaled integer."""
        # Round to nearest integer (ties to nearest even via Python round)
        return int(round(x * scale))

    @staticmethod
    def int_float_to(x: int, scale: int) -> float:
        """Convert scaled integer to floating point."""
        return float(x) / float(scale)

    @staticmethod
    def safe_float(s: str, default: float = 0.0) -> float:
        """Safely convert string to floating point, return default value if it fails."""
        try:
            return float(s)
        except Exception:
            return default

    @staticmethod
    def point_line_dist(p: complex, a: complex, b: complex) -> float:
        """Distance from point p to line segment ab (using doubles)."""
        ax, ay = a.real, a.imag
        bx, by = b.real, b.imag
        px, py = p.real, p.imag
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 == 0.0:
            dx, dy = px - ax, py - ay
            return math.hypot(dx, dy)
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
        cx, cy = ax + t * abx, ay + t * aby
        return math.hypot(px - cx, py - cy)

    @staticmethod
    def flatten_segment(segment, tol: float, t0: float = 0.0, t1: float = 1.0, depth: int = 0, max_depth: int = 18) -> List[complex]:
        """Recursively approximate any segment with a polyline within tolerance.
        Returns a list of complex points from t0..t1 (including endpoints).
        """
        p0 = segment.point(t0)
        p2 = segment.point(t1)
        pm = segment.point(0.5 * (t0 + t1))

        # Error as distance of midpoint to chord
        err = GeoUtil.point_line_dist(pm, p0, p2)
        if err <= tol or depth >= max_depth:
            return [p0, p2]
        else:
            left = GeoUtil.flatten_segment(
                segment, tol, t0, 0.5 * (t0 + t1), depth + 1, max_depth)
            right = GeoUtil.flatten_segment(
                segment, tol, 0.5 * (t0 + t1), t1, depth + 1, max_depth)
            # Avoid duplicating the midpoint
            return left[:-1] + right

    @staticmethod
    def world_bounds(geom: Optional[GeometryInt]) -> Tuple[float, float, float, float]:
        if not geom or not geom.polylines:
            return 0.0, 0.0, 1.0, 1.0
        minx, miny, maxx, maxy = geom.bounds()
        s = geom.scale if geom.scale else 1
        return minx / s, miny / s, maxx / s, maxy / s

    """ Safely try and convert any type to a float, will return default value if cannot be converted """
    @staticmethod
    def safe_to_float(x: Any, default: float = 0.0) -> float:
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def safe_to_point(pt: Any) -> Optional[PointFloat]:
        if pt is None:
            return None
        try:
            x, y = pt
            return PointFloat(GeoUtil.safe_to_float(x), GeoUtil.safe_to_float(y))
        except Exception:
            return None

    @staticmethod
    def equal_with_tolerance(a: PointFloat, b: PointFloat, abs_tol: float) -> bool:
        if a is None or b is None:
            return False
        d = abs(a - b)
        m = max(abs(a), abs(b), 1.0)
        return d <= max(abs_tol, 1e-6 * m)

    @staticmethod
    def area(points: List["PointInt"]) -> int:
        n = len(points)
        area = 0

        # Calculate area (shoelace)
        for i1 in range(n):
            i2 = (i1 + 1) % n
            area += points[i1].x * points[i2].y - points[i1].y * points[i2].x

        # Divide by 2 to get true area
        return area >> 1

    @staticmethod
    def point_in_polygon(point: PointInt, poly_points: List[PointInt]) -> PointInPolygonResult:
        """
        Ray-casting with left/right counts
        Polygon may be open or closed; repeated first/last vertex is OK.
        """
        n = len(poly_points)

        # Empty polygon then point is outside of it
        if n == 0:
            return PointInPolygonResult.Outside

        # shift so `point` -> (0,0)
        poly = [PointInt(p.x - point.x, p.y - point.y) for p in poly_points]

        r_cross = 0  # crossings on +x ray
        l_cross = 0  # crossings on -x ray

        for i in range(n):
            # vertex hit
            if poly[i].x == 0 and poly[i].y == 0:
                return PointInPolygonResult.Vertex

            i1 = (i - 1) % n
            yi, yi1 = poly[i].y, poly[i1].y

            # straddles x-axis?
            if (yi > 0) != (yi1 > 0):
                # intersection x with +x ray
                # x = (xi*yi1 - xi1*yi) / (yi1 - yi)
                num = poly[i].x * yi1 - poly[i1].x * yi
                den = yi1 - yi  # nonzero because signs differ
                x = num / den
                if x > 0:
                    r_cross += 1

            if (yi < 0) != (yi1 < 0):
                # intersection x with -x ray
                num = poly[i].x * yi1 - poly[i1].x * yi
                den = yi1 - yi
                x = num / den
                if x < 0:
                    l_cross += 1

        # on-edge if parities differ
        if (r_cross % 2) != (l_cross % 2):
            return PointInPolygonResult.Edge

        # inside if odd right-cross count
        return PointInPolygonResult.Inside if (r_cross % 2) == 1 else PointInPolygonResult.Outside
