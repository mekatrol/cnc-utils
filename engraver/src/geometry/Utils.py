import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Utils:
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
