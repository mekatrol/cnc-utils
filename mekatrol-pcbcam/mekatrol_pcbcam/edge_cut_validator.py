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
    filtered_segments = [segment for segment in segments if segment[0] != segment[1]]
    filtered_segments = _unique_segments(filtered_segments)
    filtered_segments = _collapse_colinear_segments(filtered_segments)
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

    polygons, continuity_error_segments = _build_closed_loops(filtered_segments)
    if continuity_error_segments:
        error_segments.extend(continuity_error_segments)
        issues.append("Each polygon must have a continuous boundary with no dangling edges.")
    if not polygons:
        issues.append("Edge segments do not form any closed polygons.")

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


def _build_closed_loops(segments: list[LineSegment]) -> tuple[list[Polygon], list[LineSegment]]:
    adjacency: dict[Point, list[tuple[Point, int]]] = {}
    for index, (start, end) in enumerate(segments):
        adjacency.setdefault(start, []).append((end, index))
        adjacency.setdefault(end, []).append((start, index))

    error_segment_indices: set[int] = set()
    for point, neighbors in adjacency.items():
        if len(neighbors) != 2:
            for _, segment_index in neighbors:
                error_segment_indices.add(segment_index)

    if error_segment_indices:
        return [], [segments[index] for index in sorted(error_segment_indices)]

    polygons: list[Polygon] = []
    visited_segment_indices: set[int] = set()

    for start_index, (start, end) in enumerate(segments):
        if start_index in visited_segment_indices:
            continue
        loop = [start]
        visited_segment_indices.add(start_index)
        previous = start
        current = end
        loop.append(current)

        while current != start:
            next_options = adjacency[current]
            next_point = None
            next_segment_index = None
            for candidate_point, candidate_segment_index in next_options:
                if candidate_point == previous:
                    continue
                next_point = candidate_point
                next_segment_index = candidate_segment_index
                break
            if next_point is None or next_segment_index is None:
                error_segment_indices.add(start_index)
                break
            if next_segment_index in visited_segment_indices and next_point != start:
                error_segment_indices.add(next_segment_index)
                break
            visited_segment_indices.add(next_segment_index)
            previous, current = current, next_point
            loop.append(current)
            if len(loop) > len(segments) + 1:
                error_segment_indices.add(start_index)
                break

        if loop[0] == loop[-1] and len(loop) >= 4:
            polygons.append(loop)

    if len(visited_segment_indices) != len(segments):
        for index in range(len(segments)):
            if index not in visited_segment_indices:
                error_segment_indices.add(index)

    if error_segment_indices:
        return [], [segments[index] for index in sorted(error_segment_indices)]
    return polygons, []


def _collapse_colinear_segments(segments: list[LineSegment]) -> list[LineSegment]:
    collapsed = list(segments)
    changed = True
    while changed:
        changed = False
        adjacency = _build_adjacency(collapsed)
        for point, neighbors in adjacency.items():
            if len(neighbors) != 2:
                continue
            first_neighbor, first_index = neighbors[0]
            second_neighbor, second_index = neighbors[1]
            if first_index == second_index:
                continue
            if _orientation(first_neighbor, point, second_neighbor) != 0:
                continue
            merged_segment = (first_neighbor, second_neighbor)
            next_segments = [
                segment
                for index, segment in enumerate(collapsed)
                if index not in {first_index, second_index}
            ]
            if merged_segment[0] != merged_segment[1]:
                next_segments.append(merged_segment)
            collapsed = _unique_segments(next_segments)
            changed = True
            break
    return collapsed


def _build_adjacency(segments: list[LineSegment]) -> dict[Point, list[tuple[Point, int]]]:
    adjacency: dict[Point, list[tuple[Point, int]]] = {}
    for index, (start, end) in enumerate(segments):
        adjacency.setdefault(start, []).append((end, index))
        adjacency.setdefault(end, []).append((start, index))
    return adjacency


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
