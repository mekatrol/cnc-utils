import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from geometry.GeoUtil import GeoUtil, PointInt


@pytest.fixture
def point():
    return lambda x, y: PointInt(x, y)


@pytest.fixture
def geo():
    return GeoUtil


def test_area_positive_triangle(geo, point):
    pts = [point(0, 0), point(4, 0), point(4, 3)]
    assert geo.area(pts) == 6


def test_area_negative_triangle(geo, point):
    pts = [point(0, 0), point(4, 3), point(4, 0)]
    assert geo.area(pts) == -6


def test_area_zero_line(geo, point):
    pts = [point(0, 0), point(1, 1), point(2, 2)]
    assert geo.area(pts) == 0


def test_area_square(geo, point):
    pts = [point(0, 0), point(1, 0), point(1, 1), point(0, 1)]
    assert geo.area(pts) == 1


def test_area_square_negative(geo, point):
    pts = [point(0, 0), point(0, 1), point(1, 1), point(1, 0)]
    assert geo.area(pts) == -1
