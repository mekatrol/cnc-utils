import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geometry.PointInt import PointInt
from geometry.PolylineInt import PolylineInt


def as_xy(points):
    return [(p.x, p.y) for p in points]


def test_len_lt_2_noop():
    pl = PolylineInt(points=[], simplify_tolerance=0)
    pl.simplify()
    assert as_xy(pl.points) == []

    pl = PolylineInt(points=[PointInt(1, 1)], simplify_tolerance=0)
    pl.simplify()
    assert as_xy(pl.points) == [(1, 1)]


def test_removes_consecutive_duplicates_only():
    # tol=0 so no collinear removal, only duplicate removal
    pts = [PointInt(0, 0), PointInt(0, 0),
           PointInt(1, 0), PointInt(1, 0), PointInt(2, 0)]
    pl = PolylineInt(points=pts, simplify_tolerance=0)
    pl.simplify()
    assert as_xy(pl.points) == [(0, 0), (1, 0), (2, 0)]


def test_removes_collinear_interior_points_and_preserves_endpoints():
    # perfectly collinear; tol=1 ensures removal (since (|cross|>>1)=0 < 1)
    pts = [PointInt(0, 0), PointInt(1, 0), PointInt(2, 0), PointInt(3, 0)]
    pl = PolylineInt(points=pts, simplify_tolerance=1)
    pl.simplify()
    assert as_xy(pl.points) == [(0, 0), (3, 0)]  # endpoints kept


def test_tolerance_zero_keeps_small_bend():
    # slight bend has tiny area; with tol=0 it must be kept
    pts = [PointInt(0, 0), PointInt(1, 0), PointInt(2, 1)]
    pl = PolylineInt(points=pts, simplify_tolerance=0)
    pl.simplify()
    assert as_xy(pl.points) == [(0, 0), (1, 0), (2, 1)]


def test_tolerance_one_removes_small_bend():
    # same bend; with tol=1 it is simplified
    pts = [PointInt(0, 0), PointInt(1, 0), PointInt(2, 1)]
    pl = PolylineInt(points=pts, simplify_tolerance=1)
    pl.simplify()
    assert as_xy(pl.points) == [(0, 0), (2, 1)]


def test_polyline_stays_open():
    # ensure not closed after simplify
    pts = [PointInt(0, 0), PointInt(1, 0), PointInt(2, 0), PointInt(3, 0)]
    pl = PolylineInt(points=pts, simplify_tolerance=1)
    pl.simplify()
    out = as_xy(pl.points)
    assert out[0] != out[-1]
