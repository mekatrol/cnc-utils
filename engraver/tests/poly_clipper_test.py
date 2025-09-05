import os
import sys

from geometry.GeometryInt import GeometryInt
from geometry.PolylineInt import PolylineInt
from geometry.VattiClipper import ClipOp, VattiClipper

# Make src importable
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from geometry.PointInt import PointInt


def test_convex_polygon_cases():
    # subjects and clips can be List[PointInt] or PolylineInt(points=[...])
    A = GeometryInt(
        polylines=[
            PolylineInt(
                points=[PointInt(0, 0), PointInt(8, 0), PointInt(8, 8), PointInt(0, 8)]
            )
        ],
        points=[],
    )
    B = GeometryInt(
        polylines=[
            PolylineInt(
                points=[PointInt(2, 2), PointInt(6, 2), PointInt(6, 6), PointInt(2, 6)]
            )
        ],
        points=[],
    )

    _u = VattiClipper.clip_polygons(A, B, ClipOp.UNION)  # A ∪ B
    _i = VattiClipper.clip_polygons(A, B, ClipOp.INTERSECTION)  # A ∩ B
    _d = VattiClipper.clip_polygons(A, B, ClipOp.DIFFERENCE)  # A \ B
    _x = VattiClipper.clip_polygons(A, B, ClipOp.XOR)  # symmetric diff

    # With hierarchy (holes marked)
    _u_tree = VattiClipper.clip_polygons_tree(A, B, ClipOp.UNION)
    # u_tree -> [{"points":[...], "is_hole": False/True}, ...]
