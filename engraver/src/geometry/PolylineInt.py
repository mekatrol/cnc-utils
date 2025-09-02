from dataclasses import dataclass, field
from typing import List
from geometry.PointInt import PointInt


@dataclass
class PolylineInt:
    points: List[PointInt] = field(default_factory=list)
    simplify_tolerance: int = 5

    def simplify(self) -> None:
        # There needs to be two or more points to simplyfy
        if len(self.points) < 2:
            return

        # The cross product returns twice the signed area of the traingle formed by the 3 points
        def cross(a: PointInt, b: PointInt, c: PointInt) -> int:
            # 2 * triangle area with sign
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

        tol = max(0, self.simplify_tolerance)
        out: List[PointInt] = []

        for p in self.points:
            # Step 1: skip consecutive duplicates
            if not out or p != out[-1]:
                out.append(p)

                # Step 2: drop middle if collinear within tolerance
                while len(out) >= 3:
                    a, b, c = out[-3], out[-2], out[-1]

                    if (abs(cross(a, b, c)) >> 1) < tol:
                        # Remove middle point
                        out.pop(-2)
                    else:
                        break

        # Update the point list
        self.points = out
