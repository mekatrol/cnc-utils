from __future__ import annotations

import math
from pathlib import Path
import re

from .board_bounds import BoardBounds
from .imported_gerber_file import ImportedGerberFile

Point = tuple[float, float]
Segment = tuple[Point, Point]


class GerberFileParser:
    def parse_file(self, path: str | Path) -> ImportedGerberFile:
        file_path = Path(path).resolve()
        self._apertures: dict[int, dict[str, float | str]] = {}
        self._current_aperture = -1
        self._current_operation = 2
        self._interpolation_mode = "linear"
        self._current_x: float | None = None
        self._current_y: float | None = None
        self._unit_mult = 1.0
        self._in_region = False
        self._current_region_points: list[Point] = []
        self._traces: list = []
        self._pads: list = []
        self._regions: list = []
        self._segments: list[Segment] = []
        self._arc_centers: list[Point] = []
        self._bounds = BoardBounds()

        with file_path.open("r", encoding="utf-8") as gerber_file:
            for raw_line in gerber_file:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("%"):
                    self._process_extended_command(line)
                    continue
                self._process_command(line)

        outline = self._simplify_outline(self._build_outline())
        return ImportedGerberFile(
            path=file_path,
            display_name=file_path.name,
            traces=self._traces,
            segments=list(self._segments),
            arc_centers=list(self._arc_centers),
            pads=self._pads,
            regions=self._regions,
            outline=outline,
            bounds=self._bounds,
        )

    def _process_extended_command(self, line: str) -> None:
        aperture_match = re.match(r"%ADD(\d+)([^,]+),([^*]+)\*%", line)
        if aperture_match:
            aperture_num = int(aperture_match.group(1))
            aperture_type = aperture_match.group(2)
            params = aperture_match.group(3).split("X")
            if aperture_type == "C":
                self._apertures[aperture_num] = {
                    "type": "circle",
                    "diameter": float(params[0]),
                }
            elif aperture_type == "R":
                width = float(params[0])
                height = float(params[1]) if len(params) > 1 else width
                self._apertures[aperture_num] = {
                    "type": "rectangle",
                    "width": width,
                    "height": height,
                }
            elif aperture_type == "RoundRect":
                corner_radius = float(params[0])
                x1 = float(params[1])
                y1 = float(params[2])
                x2 = float(params[3])
                y2 = float(params[4])
                self._apertures[aperture_num] = {
                    "type": "rectangle",
                    "width": abs(x2) + abs(x1) + corner_radius,
                    "height": abs(y2) + abs(y1) + corner_radius,
                }

        if "MOMM*%" in line:
            self._unit_mult = 1.0
        elif "MOIN*%" in line:
            self._unit_mult = 25.4

    def _process_command(self, line: str) -> None:
        line = line.rstrip("*")
        line = self._process_interpolation_mode(line)
        if line == "G36":
            self._in_region = True
            self._current_region_points = []
            return
        if line == "G37":
            self._finish_region()
            return

        aperture_match = re.match(r"(?:G54)?D(\d+)", line)
        if aperture_match:
            aperture_num = int(aperture_match.group(1))
            if aperture_num >= 10:
                self._current_aperture = aperture_num
            elif aperture_num <= 3:
                self._current_operation = aperture_num
            return

        coord_match = re.match(
            r"^(?:X(-?[0-9.]+))?"
            r"(?:Y(-?[0-9.]+))?"
            r"(?:I(-?[0-9.]+))?"
            r"(?:J(-?[0-9.]+))?"
            r"(?:D0([123]))?$",
            line,
        )
        if not coord_match:
            return
        if coord_match.group(1) is None and coord_match.group(2) is None:
            return

        x = (
            float(coord_match.group(1)) * 0.000001 * self._unit_mult
            if coord_match.group(1) is not None
            else self._current_x
        )
        y = (
            float(coord_match.group(2)) * 0.000001 * self._unit_mult
            if coord_match.group(2) is not None
            else self._current_y
        )
        i_offset = (
            float(coord_match.group(3)) * 0.000001 * self._unit_mult
            if coord_match.group(3) is not None
            else None
        )
        j_offset = (
            float(coord_match.group(4)) * 0.000001 * self._unit_mult
            if coord_match.group(4) is not None
            else None
        )
        if x is None or y is None:
            return

        if coord_match.group(5) is not None:
            self._current_operation = int(coord_match.group(5))
        operation = self._current_operation

        aperture = self._apertures.get(self._current_aperture)
        margin = 0.2
        if aperture:
            margin = max(
                float(aperture.get("diameter", 0.0)) * 0.5,
                float(aperture.get("width", 0.0)) * 0.5,
                float(aperture.get("height", 0.0)) * 0.5,
                margin,
            )
        self._bounds.include_point(x, y, margin)

        point = (x, y)
        if self._in_region:
            if (
                operation == 1
                and self._interpolation_mode in {"clockwise", "counterclockwise"}
                and self._current_x is not None
                and self._current_y is not None
            ):
                self._process_region_arc(point, i_offset, j_offset)
            else:
                self._process_region_point(point, operation)
            self._current_x = x
            self._current_y = y
            return

        if operation == 1:
            if self._current_x is not None and self._current_y is not None:
                start = (self._current_x, self._current_y)
                if self._interpolation_mode == "linear":
                    if start != point:
                        self._segments.append((start, point))
                    if aperture:
                        width = float(aperture.get("diameter", aperture.get("width", 0.1)))
                        self._traces.append((start, point, width))
                else:
                    self._append_arc_segments(
                        start,
                        point,
                        i_offset,
                        j_offset,
                        aperture=aperture,
                    )
        elif operation == 3 and aperture:
            self._pads.append((point, aperture))

        self._current_x = x
        self._current_y = y

    def _process_interpolation_mode(self, line: str) -> str:
        if line.startswith("G01"):
            self._interpolation_mode = "linear"
            return line[3:]
        if line.startswith("G02"):
            self._interpolation_mode = "clockwise"
            return line[3:]
        if line.startswith("G03"):
            self._interpolation_mode = "counterclockwise"
            return line[3:]
        return line

    def _process_region_point(self, point: Point, operation: int) -> None:
        if operation == 2 or not self._current_region_points:
            self._current_region_points = [point]
            return
        if operation == 1 and self._current_region_points[-1] != point:
            self._current_region_points.append(point)

    def _process_region_arc(
        self,
        point: Point,
        i_offset: float | None,
        j_offset: float | None,
    ) -> None:
        if self._current_x is None or self._current_y is None:
            self._process_region_point(point, 1)
            return
        start = (self._current_x, self._current_y)
        arc_points = self._approximate_arc_points(start, point, i_offset, j_offset)
        if not arc_points:
            self._process_region_point(point, 1)
            return
        if not self._current_region_points:
            self._current_region_points = [start]
        for arc_point in arc_points[1:]:
            if self._current_region_points[-1] != arc_point:
                self._current_region_points.append(arc_point)

    def _append_arc_segments(
        self,
        start: Point,
        end: Point,
        i_offset: float | None,
        j_offset: float | None,
        *,
        aperture: dict[str, float | str] | None,
    ) -> None:
        arc_points = self._approximate_arc_points(start, end, i_offset, j_offset)
        if not arc_points:
            if start != end:
                self._segments.append((start, end))
                if aperture:
                    width = float(aperture.get("diameter", aperture.get("width", 0.1)))
                    self._traces.append((start, end, width))
            return
        width = (
            float(aperture.get("diameter", aperture.get("width", 0.1)))
            if aperture
            else None
        )
        center = (
            None
            if i_offset is None or j_offset is None
            else (start[0] + i_offset, start[1] + j_offset)
        )
        for segment_start, segment_end in zip(arc_points, arc_points[1:]):
            if segment_start == segment_end:
                continue
            self._segments.append((segment_start, segment_end))
            if width is not None:
                self._traces.append((segment_start, segment_end, width))
            self._bounds.include_point(segment_start[0], segment_start[1], 0.2)
            self._bounds.include_point(segment_end[0], segment_end[1], 0.2)

    def _approximate_arc_points(
        self,
        start: Point,
        end: Point,
        i_offset: float | None,
        j_offset: float | None,
    ) -> list[Point]:
        if i_offset is None or j_offset is None:
            return []
        center_x = start[0] + i_offset
        center_y = start[1] + j_offset
        radius = math.hypot(start[0] - center_x, start[1] - center_y)
        if radius < 1e-9:
            return []
        center = (center_x, center_y)
        if center not in self._arc_centers:
            self._arc_centers.append(center)

        start_angle = math.atan2(start[1] - center_y, start[0] - center_x)
        end_angle = math.atan2(end[1] - center_y, end[0] - center_x)
        sweep = self._arc_sweep(start_angle, end_angle)
        if abs(sweep) < 1e-9 and start != end:
            sweep = (-2.0 * math.pi) if self._interpolation_mode == "clockwise" else (2.0 * math.pi)
        elif abs(sweep) < 1e-9 and start == end:
            sweep = (-2.0 * math.pi) if self._interpolation_mode == "clockwise" else (2.0 * math.pi)

        max_angle_step = math.radians(10.0)
        max_chord = 0.5
        chord_step = max_chord / radius if radius > 1e-9 else max_angle_step
        angle_step = min(max_angle_step, max(chord_step, math.radians(2.0)))
        segment_count = max(8, int(math.ceil(abs(sweep) / angle_step)))

        points = [start]
        for step in range(1, segment_count):
            angle = start_angle + (sweep * (step / segment_count))
            points.append(
                (
                    center_x + (math.cos(angle) * radius),
                    center_y + (math.sin(angle) * radius),
                )
            )
        points.append(end)
        return points

    def _arc_sweep(self, start_angle: float, end_angle: float) -> float:
        if self._interpolation_mode == "clockwise":
            sweep = end_angle - start_angle
            if sweep >= 0.0:
                sweep -= 2.0 * math.pi
            return sweep
        sweep = end_angle - start_angle
        if sweep <= 0.0:
            sweep += 2.0 * math.pi
        return sweep

    def _finish_region(self) -> None:
        self._in_region = False
        if len(self._current_region_points) < 3:
            self._current_region_points = []
            return
        region = self._current_region_points
        if region[0] == region[-1]:
            region = region[:-1]
        if len(region) >= 3:
            self._regions.append(region)
            for x, y in region:
                self._bounds.include_point(x, y)
        self._current_region_points = []

    def _build_outline(self) -> list[Point]:
        if not self._segments or self._has_invalid_intersections():
            return []

        adjacency: dict[Point, list[Point]] = {}
        for start, end in self._segments:
            adjacency.setdefault(start, []).append(end)
            adjacency.setdefault(end, []).append(start)
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return []

        start = self._segments[0][0]
        outline = [start]
        previous = None
        current = start
        while True:
            neighbors = adjacency[current]
            next_point = neighbors[0] if neighbors[0] != previous else neighbors[1]
            outline.append(next_point)
            if next_point == start:
                break
            previous, current = current, next_point
            if len(outline) > len(self._segments) + 1:
                return []
        if len(outline) != len(self._segments) + 1:
            return []
        return outline

    def _simplify_outline(self, outline: list[Point]) -> list[Point]:
        if len(outline) <= 3:
            return outline
        simplified = outline[:-1]
        changed = True
        while changed and len(simplified) > 2:
            changed = False
            next_outline: list[Point] = []
            count = len(simplified)
            for index, point in enumerate(simplified):
                previous = simplified[index - 1]
                following = simplified[(index + 1) % count]
                if self._orientation(previous, point, following) == 0:
                    changed = True
                    continue
                next_outline.append(point)
            if len(next_outline) == len(simplified):
                break
            simplified = next_outline
        if len(simplified) < 3:
            return []
        simplified.append(simplified[0])
        return simplified

    def _has_invalid_intersections(self) -> bool:
        for index, first in enumerate(self._segments):
            for second in self._segments[index + 1 :]:
                if self._segments_intersect(first, second):
                    return True
        return False

    def _segments_intersect(self, first: Segment, second: Segment) -> bool:
        p1, q1 = first
        p2, q2 = second
        shared_points = {p1, q1}.intersection((p2, q2))
        if len(shared_points) == 1:
            return False
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)
        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and self._point_on_segment(p2, first):
            return True
        if o2 == 0 and self._point_on_segment(q2, first):
            return True
        if o3 == 0 and self._point_on_segment(p1, second):
            return True
        if o4 == 0 and self._point_on_segment(q1, second):
            return True
        return False

    def _orientation(self, p: Point, q: Point, r: Point) -> int:
        value = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
        if abs(value) < 1e-9:
            return 0
        return 1 if value > 0 else 2

    def _point_on_segment(self, point: Point, segment: Segment) -> bool:
        start, end = segment
        epsilon = 1e-9
        return (
            min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
            and min(start[1], end[1]) - epsilon
            <= point[1]
            <= max(start[1], end[1]) + epsilon
        )
