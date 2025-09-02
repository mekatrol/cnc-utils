import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from geometry.GeoUtil import GeoUtil, PointInt


def test_area_positive_triangle():
    pts = [PointInt(0, 0), PointInt(4, 0), PointInt(4, 3)]
    assert GeoUtil.area(pts) == 6


def test_area_negative_triangle():
    pts = [PointInt(0, 0), PointInt(4, 3), PointInt(4, 0)]
    assert GeoUtil.area(pts) == -6


def test_area_zero_line():
    pts = [PointInt(0, 0), PointInt(1, 1), PointInt(2, 2)]
    assert GeoUtil.area(pts) == 0


def test_area_zero_line_duplicate_point():
    pts = [PointInt(0, 0), PointInt(1, 1), PointInt(1, 1)]
    assert GeoUtil.area(pts) == 0


def test_area_zero_line_same_point():
    pts = [PointInt(1, 1), PointInt(1, 1), PointInt(1, 1)]
    assert GeoUtil.area(pts) == 0


def test_area_square():
    pts = [PointInt(0, 0), PointInt(1, 0), PointInt(1, 1), PointInt(0, 1)]
    assert GeoUtil.area(pts) == 1


def test_area_square_negative():
    pts = [PointInt(0, 0), PointInt(0, 1), PointInt(1, 1), PointInt(1, 0)]
    assert GeoUtil.area(pts) == -1
