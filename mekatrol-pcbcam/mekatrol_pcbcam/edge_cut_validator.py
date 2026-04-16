from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]
LineSegment = tuple[Point, Point]
Polygon = list[Point]


@dataclass
class EdgeCutValidationResult:
    polygons: list[Polygon] = field(default_factory=list)
    error_segments: list[LineSegment] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return bool(self.polygons) and not self.issues

    @property
    def message(self) -> str:
        if not self.issues:
            return ""
        return " ".join(self.issues)


def validate_edge_segments(segments: list[LineSegment]) -> EdgeCutValidationResult:
    from shapely.geometry import LineString
    from shapely.ops import polygonize_full, unary_union

    filtered_segments = [segment for segment in segments if segment[0] != segment[1]]
    if not filtered_segments:
        return EdgeCutValidationResult(
            issues=["The selected edge cut layer does not contain any usable edge segments."]
        )

    issues: list[str] = []
    error_segments: list[LineSegment] = []

    intersecting_segments = _find_intersecting_segments(filtered_segments)
    if intersecting_segments:
        error_segments.extend(intersecting_segments)
        issues.append("Edge segments intersect. Polygons must not cross or overlap.")

    linework = unary_union([LineString([start, end]) for start, end in filtered_segments])
    polygons_geom, cuts_geom, dangles_geom, invalid_geom = polygonize_full(linework)

    polygons = _extract_polygon_boundaries(polygons_geom)
    if not polygons:
        issues.append("Edge segments do not form any closed polygons.")

    dangle_segments = _extract_segments(dangles_geom)
    if dangle_segments:
        error_segments.extend(dangle_segments)
        issues.append("Each polygon must have a continuous boundary with no dangling edges.")

    cut_segments = _extract_segments(cuts_geom)
    if cut_segments:
        error_segments.extend(cut_segments)
        issues.append("All edge segments must belong to the polygon boundaries.")

    invalid_segments = _extract_segments(invalid_geom)
    if invalid_segments:
        error_segments.extend(invalid_segments)
        issues.append("Some closed edge rings are invalid and cannot be used as board outlines.")

    polygon_overlap_segments = _find_polygon_overlap_segments(polygons_geom)
    if polygon_overlap_segments:
        error_segments.extend(polygon_overlap_segments)
        issues.append("Polygon boundaries must not intersect or touch each other.")

    return EdgeCutValidationResult(
        polygons=polygons,
        error_segments=_unique_segments(error_segments),
        issues=_unique_messages(issues),
    )


def _find_intersecting_segments(segments: list[LineSegment]) -> list[LineSegment]:
    intersecting_indices: set[int] = set()
    for index, first in enumerate(segments):
        for other_index in range(index + 1, len(segments)):
            if _segments_intersect(first, segments[other_index]):
                intersecting_indices.add(index)
                intersecting_indices.add(other_index)
    return [segments[index] for index in sorted(intersecting_indices)]


def _segments_intersect(first: LineSegment, second: LineSegment) -> bool:
    p1, q1 = first
    p2, q2 = second
    shared_points = {p1, q1}.intersection((p2, q2))
    if len(shared_points) == 1:
        return False

    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _point_on_segment(p2, first):
        return True
    if o2 == 0 and _point_on_segment(q2, first):
        return True
    if o3 == 0 and _point_on_segment(p1, second):
        return True
    if o4 == 0 and _point_on_segment(q1, second):
        return True
    return False


def _orientation(p: Point, q: Point, r: Point) -> int:
    value = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def _point_on_segment(point: Point, segment: LineSegment) -> bool:
    start, end = segment
    epsilon = 1e-9
    return (
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )


def _extract_polygon_boundaries(geometry) -> list[Polygon]:
    polygons: list[Polygon] = []
    for polygon in getattr(geometry, "geoms", []):
        polygons.append([(float(x), float(y)) for x, y in polygon.exterior.coords])
        polygons.extend(
            [[(float(x), float(y)) for x, y in interior.coords] for interior in polygon.interiors]
        )
    return polygons


def _extract_segments(geometry) -> list[LineSegment]:
    segments: list[LineSegment] = []
    if geometry.is_empty:
        return segments
    geom_type = geometry.geom_type
    if geom_type == "LineString":
        coords = [(float(x), float(y)) for x, y in geometry.coords]
        segments.extend((start, end) for start, end in zip(coords, coords[1:]))
        return segments
    if geom_type in {"MultiLineString", "GeometryCollection"}:
        for item in geometry.geoms:
            segments.extend(_extract_segments(item))
        return segments
    if geom_type == "LinearRing":
        coords = [(float(x), float(y)) for x, y in geometry.coords]
        segments.extend((start, end) for start, end in zip(coords, coords[1:]))
    return segments


def _find_polygon_overlap_segments(geometry) -> list[LineSegment]:
    polygons = list(getattr(geometry, "geoms", []))
    segments: list[LineSegment] = []
    for index, first in enumerate(polygons):
        for second in polygons[index + 1 :]:
            if first.boundary.disjoint(second.boundary):
                continue
            segments.extend(_extract_segments(first.boundary))
            segments.extend(_extract_segments(second.boundary))
    return segments


def _unique_segments(segments: list[LineSegment]) -> list[LineSegment]:
    ordered: list[LineSegment] = []
    seen: set[tuple[Point, Point]] = set()
    for start, end in segments:
        key = (start, end) if start <= end else (end, start)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((start, end))
    return ordered


def _unique_messages(messages: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        ordered.append(message)
    return ordered
