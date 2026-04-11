import re

from gerber2nc_lib.board_context import BoardContext

Point = tuple[float, float]
Segment = tuple[Point, Point]


class GerberEdgeCutsParser:
    def __init__(self, filename: str, context: BoardContext):
        # The edge-cuts parser shares the same board context object as the rest
        # of the import pipeline so that the final board bounds include the
        # mechanical outline as well as any copper or drill features.
        self.context = context
        # The final outline is stored as an ordered loop of points. Downstream
        # code uses this for visualization and for the edge-marking G-code.
        self.outline: list[tuple[float, float]] = []
        # Gerber profile files are effectively a list of plotting moves and
        # draws. During parsing we collect the individual drawn line segments
        # first, then reconstruct a clean closed polygon from those segments.
        self._segments: list[Segment] = []
        # As with other Gerber layers, coordinate numbers are just digits until
        # the file declares whether they are in millimetres or inches.
        self.unit_mult: float = 1.0
        # Gerber drawing is stateful: a D01 draw command uses the previously
        # visited point as its start, so we keep track of the current point.
        self.current_point: Point | None = None

        try:
            with open(filename, "r", encoding="utf-8") as edge_cuts_file:
                for line in edge_cuts_file:
                    self._process_line(line.strip())
        except FileNotFoundError:
            print("No edge cuts defined, thats OK")
            return

        self.outline = self._simplify_outline(self._build_outline())

        if self.outline and self.outline[0] != self.outline[-1]:
            print("Error: non closed outline")

    def _process_line(self, line: str) -> None:
        # Edge-cuts Gerber files contain the same general command vocabulary as
        # copper layers, but here we only care about unit declarations and
        # straight coordinate moves/draws that define the PCB profile.
        if "MOMM*%" in line:
            self.unit_mult = 1.0
        elif "MOIN*%" in line:
            self.unit_mult = 25.4

        # Example:
        #   X120000000Y-80000000D02*
        #   X180000000Y-80000000D01*
        #
        # KiCad edge-cuts are typically emitted as absolute XY coordinates with
        # D02 meaning "move to this point without drawing" and D01 meaning
        # "draw from the previous point to this point". For the profile layer
        # that drawn stroke is the board outline segment we need to recover.
        coord_match = re.match(r"X(-?[0-9.]+)Y(-?[0-9.]+)D0([0123])?", line)
        if coord_match:
            # The code assumes six decimal places of coordinate precision. The
            # raw Gerber integers are therefore scaled by 1e-6 and then
            # converted to millimetres using the active unit multiplier.
            x = float(coord_match.group(1)) * 0.000001 * self.unit_mult
            y = float(coord_match.group(2)) * 0.000001 * self.unit_mult
            operation = coord_match.group(3)
            point = (x, y)

            # Even though edge cuts are thin lines, they still define the board
            # extents, so a small margin keeps the overall bounds conservative.
            self.context.update_bounds(x, y, 0.2)

            if operation == "1":
                # D01 means "draw". In an outline layer this becomes a segment
                # from the previous point to the new point. Zero-length segments
                # are ignored because they do not contribute any usable profile.
                if self.current_point and self.current_point != point:
                    self._segments.append((self.current_point, point))
                self.current_point = point
            else:
                # D02 and any other non-drawing cases simply reposition the
                # current point for the next segment start.
                self.current_point = point

    def _build_outline(self) -> list[tuple[float, float]]:
        # If no drawn segments were discovered, there is no outline to build.
        if not self._segments:
            return []

        # The downstream machining logic expects one clean, non-self-
        # intersecting loop. Reject profiles that cross over themselves away
        # from legitimate segment joins because those are ambiguous to follow.
        if self._has_invalid_intersections():
            print("Error: edge cuts intersect away from segment joins")
            return []

        # Build an undirected graph where each unique point knows which other
        # points it is connected to by a drawn edge-cut segment.
        adjacency: dict[Point, list[Point]] = {}
        for start, end in self._segments:
            adjacency.setdefault(start, []).append(end)
            adjacency.setdefault(end, []).append(start)

        # A simple closed outline should have degree 2 at every vertex: one
        # segment arriving and one segment leaving. If any point has a
        # different number of neighbors, the profile is branched or open.
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return []

        # Reconstruct the ordered loop by walking from one segment endpoint and
        # always taking the next neighbor that is not the point we just came
        # from. Because every vertex has degree 2, this yields the unique cycle.
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
        # KiCad edge cuts often contain runs of collinear segments. For
        # visualization and G-code output it is cleaner to collapse intermediate
        # points that lie on a straight line and preserve only true corners.
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
        # Compare every segment pair to detect crossings that occur away from a
        # shared endpoint. This is O(n^2), but profile layers are usually small
        # enough that the simplicity is worth it here.
        for i, first in enumerate(self._segments):
            for second in self._segments[i + 1 :]:
                if self._segments_intersect(first, second):
                    return True

        return False

    def _segments_intersect(self, first: Segment, second: Segment) -> bool:
        p1, q1 = first
        p2, q2 = second

        # Two neighboring outline segments are allowed to meet at exactly one
        # shared endpoint. That is not considered an invalid self-intersection.
        shared_points = {p1, q1}.intersection((p2, q2))
        if len(shared_points) == 1:
            shared_point = next(iter(shared_points))
            if self._point_on_segment(shared_point, first) and self._point_on_segment(
                shared_point, second
            ):
                return False

        # The orientation test is the standard computational-geometry approach
        # for segment intersection. It determines whether each segment straddles
        # the infinite line extended by the other segment.
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
        # This computes the signed area / cross product for the turn formed by
        # p -> q -> r:
        #   0 = collinear
        #   1 = clockwise
        #   2 = counter-clockwise
        #
        # A tiny epsilon avoids classifying nearly straight runs as corners due
        # to floating-point noise after the Gerber coordinates are scaled.
        value = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
        epsilon = 1e-9
        if abs(value) < epsilon:
            return 0
        return 1 if value > 0 else 2

    def _point_on_segment(self, point: Point, segment: Segment) -> bool:
        # Bounding-box containment is enough here because callers only use this
        # after a collinearity check or when testing a shared endpoint case.
        start, end = segment
        epsilon = 1e-9
        return (
            min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
            and min(start[1], end[1]) - epsilon
            <= point[1]
            <= max(start[1], end[1]) + epsilon
        )

    def shift(self, x_base: float, y_base: float) -> None:
        # Once the shared board bounds are known, the outline is shifted into a
        # local origin so preview rendering and generated G-code all work in the
        # same rebased coordinate system.
        for i, point in enumerate(self.outline):
            self.outline[i] = (point[0] - x_base, point[1] - y_base)
