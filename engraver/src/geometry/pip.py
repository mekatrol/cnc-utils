from enum import Enum, auto
from typing import List
from geometry.PointInt import PointInt


class PolygonRelation(Enum):
    DISJOINT = auto()  # no touching
    INTERSECT = auto()  # edges cross or touch (incl. tangency/overlap)
    A_INSIDE_B = auto()
    B_INSIDE_A = auto()


def bounding_box(polygon: List[PointInt]):
    x_values = [point.x for point in polygon]
    y_values = [point.y for point in polygon]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def orientation(point_a: PointInt, point_b: PointInt, point_c: PointInt) -> int:
    return (point_b.x - point_a.x) * (point_c.y - point_a.y) - (
        point_b.y - point_a.y
    ) * (point_c.x - point_a.x)


def is_point_on_segment(
    point_a: PointInt, point_b: PointInt, point_c: PointInt
) -> bool:
    return min(point_a.x, point_b.x) <= point_c.x <= max(point_a.x, point_b.x) and min(
        point_a.y, point_b.y
    ) <= point_c.y <= max(point_a.y, point_b.y)


def do_segments_intersect(
    point_a: PointInt, point_b: PointInt, point_c: PointInt, point_d: PointInt
) -> bool:
    orientation1 = orientation(point_a, point_b, point_c)
    orientation2 = orientation(point_a, point_b, point_d)
    orientation3 = orientation(point_c, point_d, point_a)
    orientation4 = orientation(point_c, point_d, point_b)

    if (orientation1 > 0) != (orientation2 > 0) and (orientation3 > 0) != (
        orientation4 > 0
    ):
        return True  # proper crossing

    # touching or collinear overlap
    if orientation1 == 0 and is_point_on_segment(point_a, point_b, point_c):
        return True
    if orientation2 == 0 and is_point_on_segment(point_a, point_b, point_d):
        return True
    if orientation3 == 0 and is_point_on_segment(point_c, point_d, point_a):
        return True
    if orientation4 == 0 and is_point_on_segment(point_c, point_d, point_b):
        return True
    return False


def polygon_edges(polygon: List[PointInt]):
    count = len(polygon)
    max_index = count - 1 if count >= 2 and polygon[0] == polygon[-1] else count
    for i in range(max_index):
        point_a = polygon[i]
        point_b = polygon[(i + 1) % max_index]
        if point_a != point_b:
            yield point_a, point_b


class PointInPolygonResult(Enum):
    OUTSIDE = 0
    INSIDE = 1
    ON_EDGE = 2
    ON_VERTEX = 3


def point_in_polygon(
    query_point: PointInt, polygon: List[PointInt]
) -> PointInPolygonResult:
    count = len(polygon)
    if count == 0:
        return PointInPolygonResult.OUTSIDE

    shifted_x = [point.x - query_point.x for point in polygon]
    shifted_y = [point.y - query_point.y for point in polygon]

    crossings_right = 0
    crossings_left = 0

    for i in range(count):
        if shifted_x[i] == 0 and shifted_y[i] == 0:
            return PointInPolygonResult.ON_VERTEX

        j = (i - 1) % count

        if (shifted_y[i] > 0) != (shifted_y[j] > 0):
            numerator = shifted_x[i] * shifted_y[j] - shifted_x[j] * shifted_y[i]
            denominator = shifted_y[j] - shifted_y[i]
            x_intersection = numerator / denominator
            if x_intersection > 0:
                crossings_right += 1

        if (shifted_y[i] < 0) != (shifted_y[j] < 0):
            numerator = shifted_x[i] * shifted_y[j] - shifted_x[j] * shifted_y[i]
            denominator = shifted_y[j] - shifted_y[i]
            x_intersection = numerator / denominator
            if x_intersection < 0:
                crossings_left += 1

    if (crossings_right & 1) != (crossings_left & 1):
        return PointInPolygonResult.ON_EDGE
    return (
        PointInPolygonResult.INSIDE
        if (crossings_right & 1)
        else PointInPolygonResult.OUTSIDE
    )


def classify_polygons(
    polygon_a: List[PointInt], polygon_b: List[PointInt]
) -> PolygonRelation:
    if len(polygon_a) < 3 or len(polygon_b) < 3:
        for edge_a_start, edge_a_end in polygon_edges(polygon_a):
            for edge_b_start, edge_b_end in polygon_edges(polygon_b):
                if do_segments_intersect(
                    edge_a_start, edge_a_end, edge_b_start, edge_b_end
                ):
                    return PolygonRelation.INTERSECT
        return PolygonRelation.DISJOINT

    # bounding box reject
    ax0, ay0, ax1, ay1 = bounding_box(polygon_a)
    bx0, by0, bx1, by1 = bounding_box(polygon_b)
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return PolygonRelation.DISJOINT

    # edge intersection
    for edge_a_start, edge_a_end in polygon_edges(polygon_a):
        for edge_b_start, edge_b_end in polygon_edges(polygon_b):
            if do_segments_intersect(
                edge_a_start, edge_a_end, edge_b_start, edge_b_end
            ):
                return PolygonRelation.INTERSECT

    # containment check
    pip_a_in_b = point_in_polygon(polygon_a[0], polygon_b)
    if pip_a_in_b == PointInPolygonResult.INSIDE:
        return PolygonRelation.A_INSIDE_B
    if pip_a_in_b in (PointInPolygonResult.ON_EDGE, PointInPolygonResult.ON_VERTEX):
        return PolygonRelation.INTERSECT

    pip_b_in_a = point_in_polygon(polygon_b[0], polygon_a)
    if pip_b_in_a == PointInPolygonResult.INSIDE:
        return PolygonRelation.B_INSIDE_A
    if pip_b_in_a in (PointInPolygonResult.ON_EDGE, PointInPolygonResult.ON_VERTEX):
        return PolygonRelation.INTERSECT

    return PolygonRelation.DISJOINT
