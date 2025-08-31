import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GeometryUtils:
    @staticmethod
    def float_to_int(x: float, scale: int) -> int:
        """Convert floating point to scaled integer."""
        # Round to nearest integer (ties to nearest even via Python round)
        return int(round(x * scale))

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
        err = GeometryUtils.point_line_dist(pm, p0, p2)
        if err <= tol or depth >= max_depth:
            return [p0, p2]
        else:
            left = GeometryUtils.flatten_segment(
                segment, tol, t0, 0.5 * (t0 + t1), depth + 1, max_depth)
            right = GeometryUtils.flatten_segment(
                segment, tol, 0.5 * (t0 + t1), t1, depth + 1, max_depth)
            # Avoid duplicating the midpoint
            return left[:-1] + right
