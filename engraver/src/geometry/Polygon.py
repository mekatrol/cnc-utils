from enum import Enum, auto
from typing import List
from geometry.PointInt import PointInt


class PolygonRelation(Enum):
    """
    High-level spatial relationship between two polygons.
    """

    DISJOINT = auto()  # Polygons do not touch or overlap in any way
    INTERSECT = (
        auto()
    )  # Polygons touch or overlap (edge crossing, touching, or collinear overlap)
    A_INSIDE_B = auto()  # Polygon A lies strictly inside polygon B
    B_INSIDE_A = auto()  # Polygon B lies strictly inside polygon A


def bounding_box(polygon: List[PointInt]):
    """
    Compute the axis-aligned bounding box (AABB) of a polygon.

    Returns:
        (min_x, min_y, max_x, max_y)

    Used as a fast rejection test before more expensive geometry checks.
    """
    x_values = [point.x for point in polygon]
    y_values = [point.y for point in polygon]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def orientation(point_a: PointInt, point_b: PointInt, point_c: PointInt) -> int:
    """
    Compute the signed area (cross product) of the triangle ABC.

    Interpretation of the return value:
        > 0 : C is to the left of the directed line AB (counter-clockwise turn)
        < 0 : C is to the right of the directed line AB (clockwise turn)
        = 0 : A, B, and C are collinear

    This is the fundamental primitive for segment intersection tests.
    """
    return (point_b.x - point_a.x) * (point_c.y - point_a.y) - (
        point_b.y - point_a.y
    ) * (point_c.x - point_a.x)


def is_point_on_segment(
    point_a: PointInt, point_b: PointInt, point_c: PointInt
) -> bool:
    """
    Check whether point C lies on the closed line segment AB.

    Assumes that A, B, and C are collinear.
    """
    return min(point_a.x, point_b.x) <= point_c.x <= max(point_a.x, point_b.x) and min(
        point_a.y, point_b.y
    ) <= point_c.y <= max(point_a.y, point_b.y)


def do_segments_intersect(
    point_a: PointInt, point_b: PointInt, point_c: PointInt, point_d: PointInt
) -> bool:
    """
    Determine whether line segments AB and CD intersect.

    Handles:
      - Proper crossings
      - Touching at endpoints
      - Collinear overlap
    """
    # Compute orientations of all endpoint combinations
    orientation1 = orientation(point_a, point_b, point_c)
    orientation2 = orientation(point_a, point_b, point_d)
    orientation3 = orientation(point_c, point_d, point_a)
    orientation4 = orientation(point_c, point_d, point_b)

    # Proper intersection: endpoints lie on opposite sides of the other segment
    if (orientation1 > 0) != (orientation2 > 0) and (orientation3 > 0) != (
        orientation4 > 0
    ):
        return True

    # Degenerate cases: touching or collinear overlap
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
    """
    Yield all non-degenerate edges of a polygon as (start, end) point pairs.

    If the polygon is explicitly closed (first point equals last point),
    the duplicate closing edge is ignored.
    """
    count = len(polygon)

    # If the polygon is closed, ignore the duplicated last point
    max_index = count - 1 if count >= 2 and polygon[0] == polygon[-1] else count

    for i in range(max_index):
        point_a = polygon[i]
        point_b = polygon[(i + 1) % max_index]

        # Skip zero-length edges
        if point_a != point_b:
            yield point_a, point_b


class PointInPolygonResult(Enum):
    """
    Result of a point-in-polygon query.
    """

    OUTSIDE = 0  # Point lies strictly outside the polygon
    INSIDE = 1  # Point lies strictly inside the polygon
    ON_EDGE = 2  # Point lies on an edge of the polygon
    ON_VERTEX = 3  # Point coincides exactly with a polygon vertex


def point_in_polygon(
    query_point: PointInt, polygon: List[PointInt]
) -> PointInPolygonResult:
    """
    Determine the position of a point relative to a polygon.

    Uses a symmetric ray-casting approach:
      - Counts ray crossings to the right and left
      - Detects edge and vertex hits explicitly

    This avoids ambiguity when the ray passes exactly through a vertex.
    """
    count = len(polygon)
    if count == 0:
        return PointInPolygonResult.OUTSIDE

    # Translate polygon so that query_point becomes the origin (0, 0)
    shifted_x = [point.x - query_point.x for point in polygon]
    shifted_y = [point.y - query_point.y for point in polygon]

    crossings_right = 0
    crossings_left = 0

    for i in range(count):
        # Exact vertex hit
        if shifted_x[i] == 0 and shifted_y[i] == 0:
            return PointInPolygonResult.ON_VERTEX

        j = (i - 1) % count

        # Check upward or downward crossings for a ray to the right
        if (shifted_y[i] > 0) != (shifted_y[j] > 0):
            numerator = shifted_x[i] * shifted_y[j] - shifted_x[j] * shifted_y[i]
            denominator = shifted_y[j] - shifted_y[i]
            x_intersection = numerator / denominator
            if x_intersection > 0:
                crossings_right += 1

        # Check upward or downward crossings for a ray to the left
        if (shifted_y[i] < 0) != (shifted_y[j] < 0):
            numerator = shifted_x[i] * shifted_y[j] - shifted_x[j] * shifted_y[i]
            denominator = shifted_y[j] - shifted_y[i]
            x_intersection = numerator / denominator
            if x_intersection < 0:
                crossings_left += 1

    # If parity differs, the point lies exactly on an edge
    if (crossings_right & 1) != (crossings_left & 1):
        return PointInPolygonResult.ON_EDGE

    # Odd number of crossings → inside, even → outside
    return (
        PointInPolygonResult.INSIDE
        if (crossings_right & 1)
        else PointInPolygonResult.OUTSIDE
    )


def classify_polygons(
    polygon_a: List[PointInt], polygon_b: List[PointInt]
) -> PolygonRelation:
    """
    Classify the spatial relationship between two polygons.

    The algorithm proceeds in increasing order of cost:
      1. Handle degenerate polygons (lines/points)
      2. Bounding box rejection
      3. Edge intersection test
      4. Containment test using point-in-polygon
    """
    # Degenerate case: one or both polygons are lines or points
    if len(polygon_a) < 3 or len(polygon_b) < 3:
        for edge_a_start, edge_a_end in polygon_edges(polygon_a):
            for edge_b_start, edge_b_end in polygon_edges(polygon_b):
                if do_segments_intersect(
                    edge_a_start, edge_a_end, edge_b_start, edge_b_end
                ):
                    return PolygonRelation.INTERSECT
        return PolygonRelation.DISJOINT

    # Fast axis-aligned bounding box rejection
    ax0, ay0, ax1, ay1 = bounding_box(polygon_a)
    bx0, by0, bx1, by1 = bounding_box(polygon_b)
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return PolygonRelation.DISJOINT

    # Check for any intersecting edges
    for edge_a_start, edge_a_end in polygon_edges(polygon_a):
        for edge_b_start, edge_b_end in polygon_edges(polygon_b):
            if do_segments_intersect(
                edge_a_start, edge_a_end, edge_b_start, edge_b_end
            ):
                return PolygonRelation.INTERSECT

    # Check if polygon A is contained in polygon B
    pip_a_in_b = point_in_polygon(polygon_a[0], polygon_b)
    if pip_a_in_b == PointInPolygonResult.INSIDE:
        return PolygonRelation.A_INSIDE_B
    if pip_a_in_b in (PointInPolygonResult.ON_EDGE, PointInPolygonResult.ON_VERTEX):
        return PolygonRelation.INTERSECT

    # Check if polygon B is contained in polygon A
    pip_b_in_a = point_in_polygon(polygon_b[0], polygon_a)
    if pip_b_in_a == PointInPolygonResult.INSIDE:
        return PolygonRelation.B_INSIDE_A
    if pip_b_in_a in (PointInPolygonResult.ON_EDGE, PointInPolygonResult.ON_VERTEX):
        return PolygonRelation.INTERSECT

    # No intersection and no containment
    return PolygonRelation.DISJOINT
