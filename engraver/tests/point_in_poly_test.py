import os
import sys

from geometry.PolylineInt import PolylineInt
# Make src importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geometry.PointInt import PointInt
from geometry.GeoUtil import GeoUtil
from geometry.PointInPolygonResult import PointInPolygonResult


def test_convex_polygon_cases():
    # Convex rectangle 0,0 -> 5,3
    convex = PolylineInt(points=[PointInt(0, 0), PointInt(5, 0), PointInt(5, 3), PointInt(0, 3)])

    # Inside
    assert GeoUtil.point_in_polygon(PointInt(2, 1), convex.points) == PointInPolygonResult.Inside

    # Edge (midpoint on bottom edge)
    assert GeoUtil.point_in_polygon(PointInt(3, 0), convex.points) == PointInPolygonResult.Edge

    # Vertex
    assert GeoUtil.point_in_polygon(PointInt(0, 0), convex.points) == PointInPolygonResult.Vertex

    # Outside
    assert GeoUtil.point_in_polygon(PointInt(6, 1), convex.points) == PointInPolygonResult.Outside


def test_concave_polygon_cases():
    # Concave "U" shape scaled to ensure integer interior points.
    # Convex hull is the 0..8 square, but the notch (2..6, y>2) is outside the polygon.
    concave = PolylineInt(points=[
        PointInt(0, 0), PointInt(8, 0), PointInt(8, 8),
        PointInt(6, 8), PointInt(6, 2), PointInt(2, 2),
        PointInt(2, 8), PointInt(0, 8), PointInt(0, 0)],
        simplify_tolerance=5)

    # Inside polygon (left arm interior)
    assert GeoUtil.point_in_polygon(PointInt(1, 4), concave.points) == PointInPolygonResult.Inside

    # On an edge (bottom of notch)
    assert GeoUtil.point_in_polygon(PointInt(4, 2), concave.points) == PointInPolygonResult.Edge

    # Vertex
    assert GeoUtil.point_in_polygon(PointInt(2, 2), concave.points) == PointInPolygonResult.Vertex

    # Outside but inside the convex hull (in the notch)
    assert GeoUtil.point_in_polygon(PointInt(4, 6), concave.points) == PointInPolygonResult.Outside

    # Outside hull
    assert GeoUtil.point_in_polygon(PointInt(9, 4), concave.points) == PointInPolygonResult.Outside

    # Outside hull
    assert GeoUtil.point_in_polygon(PointInt(-2, 1), concave.points) == PointInPolygonResult.Outside
