from typing import List, Optional
import math
import pyclipper
from geometry.PointInt import PointInt
from geometry.PolylineInt import PolylineInt


class PolyProcessor:
    @staticmethod
    def split_self_intersections(
        polylines: List[PolylineInt],
    ) -> List[PolylineInt]:
        out: List[PolylineInt] = []
        for poly in polylines:
            pts = poly.points
            if len(pts) < 4:
                out.append(poly)
                continue

            if pts[0] != pts[-1]:
                out.append(poly)
                continue

            path = [(p.x, p.y) for p in pts[:-1]]
            if len(path) < 3:
                out.append(poly)
                continue

            try:
                simple_paths = pyclipper.SimplifyPolygon(
                    path, fill_type=pyclipper.PFT_EVENODD
                )
            except Exception:
                out.append(poly)
                continue

            if not simple_paths:
                continue

            for sp in simple_paths:
                if len(sp) < 3:
                    continue
                points = [PointInt(int(x), int(y)) for (x, y) in sp]
                if points[0] != points[-1]:
                    points.append(points[0])
                out.append(
                    PolylineInt(
                        points=points,
                        simplify_tolerance=poly.simplify_tolerance,
                    )
                )

        return out

    @staticmethod
    def close_open_polylines(polylines: List[PolylineInt]) -> List[PolylineInt]:
        out: List[PolylineInt] = []
        for poly in polylines:
            pts = poly.points
            if len(pts) < 3:
                out.append(poly)
                continue
            if pts[0] == pts[-1]:
                out.append(poly)
                continue
            path = [(p.x, p.y) for p in pts] + [(pts[0].x, pts[0].y)]
            if abs(pyclipper.Area(path)) == 0:
                out.append(poly)
                continue
            closed = pts + [pts[0]]
            out.append(
                PolylineInt(points=closed, simplify_tolerance=poly.simplify_tolerance)
            )
        return out

    @staticmethod
    def split_intersections_between_polygons(
        polylines: List[PolylineInt],
        scale: int,
    ) -> List[PolylineInt]:
        closed: List[PolylineInt] = []
        open_polylines: List[PolylineInt] = []
        for poly in polylines:
            pts = poly.points
            if len(pts) >= 4 and pts[0] == pts[-1]:
                closed.append(poly)
            else:
                open_polylines.append(poly)

        cut_delta = max(2, int(round(scale * 1e-3)))

        if len(closed) < 2 and not open_polylines:
            return polylines

        disjoint_paths: List[List[tuple[int, int]]] = []

        def clip_paths(
            subject: List[List[tuple[int, int]]],
            clip: List[List[tuple[int, int]]],
            op: int,
        ) -> List[List[tuple[int, int]]]:
            pc = pyclipper.Pyclipper()

            if subject:
                pc.AddPaths(subject, pyclipper.PT_SUBJECT, True)  # type: ignore
            if clip:
                pc.AddPaths(clip, pyclipper.PT_CLIP, True)  # type: ignore

            tree = pc.Execute2(
                op, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD
            )
            out: List[List[tuple[int, int]]] = []

            def walk(node) -> None:
                for child in node.Childs:
                    if not child.IsHole and child.Contour:
                        out.append(child.Contour)
                    walk(child)

            walk(tree)
            return out

        def bbox(path: List[tuple[int, int]]) -> tuple[int, int, int, int]:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            return min(xs), min(ys), max(xs), max(ys)

        def bboxes_overlap(
            a: tuple[int, int, int, int], b: tuple[int, int, int, int]
        ) -> bool:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

        paths: List[List[tuple[int, int]]] = []
        for poly in closed:
            pts = poly.points
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) < 3:
                continue
            paths.append([(p.x, p.y) for p in pts])

        tol = max(1, cut_delta)

        def point_on_segment(
            p: tuple[int, int],
            a: tuple[int, int],
            b: tuple[int, int],
            tol_dist: int,
        ) -> bool:
            ax, ay = a
            bx, by = b
            px, py = p
            dx = bx - ax
            dy = by - ay
            if dx == 0 and dy == 0:
                return px == ax and py == ay
            cross = (px - ax) * dy - (py - ay) * dx
            if abs(cross) > tol_dist * max(abs(dx), abs(dy), 1):
                return False
            minx = min(ax, bx) - tol_dist
            maxx = max(ax, bx) + tol_dist
            miny = min(ay, by) - tol_dist
            maxy = max(ay, by) + tol_dist
            return minx <= px <= maxx and miny <= py <= maxy

        def insert_point(
            path: List[tuple[int, int]],
            p: tuple[int, int],
            tol_dist: int,
        ) -> tuple[Optional[List[tuple[int, int]]], Optional[int]]:
            for i, pt in enumerate(path):
                if pt == p:
                    return path, i
            n = len(path)
            for i in range(n):
                a = path[i]
                b = path[(i + 1) % n]
                if point_on_segment(p, a, b, tol_dist):
                    new_path = path[: i + 1] + [p] + path[i + 1 :]
                    return new_path, i + 1
            return None, None

        def split_path_with_polyline(
            path: List[tuple[int, int]],
            polyline: List[tuple[int, int]],
            tol_dist: int,
        ) -> Optional[List[List[tuple[int, int]]]]:
            if len(polyline) < 2:
                return None
            p0 = polyline[0]
            p1 = polyline[-1]
            if p0 == p1:
                return None

            updated, i0 = insert_point(path, p0, tol_dist)
            if updated is None or i0 is None:
                return None
            updated, i1 = insert_point(updated, p1, tol_dist)
            if updated is None or i1 is None:
                return None
            if i0 == i1:
                return None

            if i0 < i1:
                seg1 = updated[i0 : i1 + 1]
                seg2 = updated[i1:] + updated[: i0 + 1]
            else:
                seg1 = updated[i0:] + updated[: i1 + 1]
                seg2 = updated[i1 : i0 + 1]

            rev_line = list(reversed(polyline))
            poly1 = seg1 + rev_line[1:]
            if poly1 and poly1[0] == poly1[-1]:
                poly1 = poly1[:-1]

            poly2 = seg2 + polyline[1:]
            if poly2 and poly2[0] == poly2[-1]:
                poly2 = poly2[:-1]

            if len(poly1) < 3 or len(poly2) < 3:
                return None

            return [poly1, poly2]

        if open_polylines and paths:
            remaining_open: List[PolylineInt] = []
            for poly in open_polylines:
                pts = poly.points
                if len(pts) < 2:
                    continue
                polyline = [(p.x, p.y) for p in pts]
                split_done = False
                for i in range(len(paths)):
                    split_paths = split_path_with_polyline(paths[i], polyline, tol)
                    if split_paths:
                        paths.pop(i)
                        paths.extend(split_paths)
                        split_done = True
                        break
                if not split_done:
                    remaining_open.append(poly)
            open_polylines = remaining_open

        if not paths:
            return polylines

        areas = [abs(pyclipper.Area(p)) for p in paths]

        def intersection_area(
            a: List[tuple[int, int]], b: List[tuple[int, int]]
        ) -> int:
            if not bboxes_overlap(bbox(a), bbox(b)):
                return 0
            pc = pyclipper.Pyclipper()
            pc.AddPath(a, pyclipper.PT_SUBJECT, True)  # type: ignore
            pc.AddPath(b, pyclipper.PT_CLIP, True)  # type: ignore
            inter = pc.Execute(
                pyclipper.CT_INTERSECTION,
                pyclipper.PFT_EVENODD,
                pyclipper.PFT_EVENODD,
            )
            return int(sum(abs(pyclipper.Area(p)) for p in inter))

        depths = [0 for _ in paths]
        for i, path in enumerate(paths):
            area_i = areas[i]
            if area_i == 0:
                continue
            for j, other in enumerate(paths):
                if i == j:
                    continue
                area_j = areas[j]
                if area_j == 0:
                    continue
                ia = intersection_area(path, other)
                if ia <= 0:
                    continue
                if ia >= area_i:
                    if ia == area_i and area_i == area_j:
                        continue
                    depths[i] += 1

        order = sorted(
            range(len(paths)),
            key=lambda i: (-depths[i], areas[i]),
        )

        for idx in order:
            path = paths[idx]
            if not disjoint_paths:
                disjoint_paths.append(path)
                continue
            remaining_new: List[List[tuple[int, int]]] = [path]
            next_disjoint: List[List[tuple[int, int]]] = []

            for existing in disjoint_paths:
                if not remaining_new:
                    next_disjoint.append(existing)
                    continue
                if not bboxes_overlap(bbox(existing), bbox(path)):
                    next_disjoint.append(existing)
                    continue

                intersections = clip_paths(
                    remaining_new, [existing], pyclipper.CT_INTERSECTION
                )
                if not intersections:
                    next_disjoint.append(existing)
                    continue

                existing_minus_new = clip_paths(
                    [existing], remaining_new, pyclipper.CT_DIFFERENCE
                )
                if existing_minus_new:
                    next_disjoint.extend(existing_minus_new)
                next_disjoint.extend(intersections)

                remaining_new = clip_paths(
                    remaining_new, [existing], pyclipper.CT_DIFFERENCE
                )

            if remaining_new:
                next_disjoint.extend(remaining_new)

            disjoint_paths = next_disjoint

        if open_polylines and disjoint_paths:
            cutter_paths: List[List[tuple[int, int]]] = []

            def segment_strip(
                p0: PointInt, p1: PointInt, delta: int
            ) -> Optional[List[tuple[int, int]]]:
                dx = p1.x - p0.x
                dy = p1.y - p0.y
                if dx == 0 and dy == 0:
                    return None
                length = math.hypot(dx, dy)
                if length == 0.0:
                    return None
                nx = -dy / length
                ny = dx / length
                ox = int(round(nx * delta))
                oy = int(round(ny * delta))
                if ox == 0 and oy == 0:
                    ox = 0
                    oy = 1 if delta > 0 else -1
                return [
                    (p0.x + ox, p0.y + oy),
                    (p0.x - ox, p0.y - oy),
                    (p1.x - ox, p1.y - oy),
                    (p1.x + ox, p1.y + oy),
                ]

            for poly in open_polylines:
                pts = poly.points
                if len(pts) < 2:
                    continue
                for i in range(len(pts) - 1):
                    strip = segment_strip(pts[i], pts[i + 1], cut_delta)
                    if strip:
                        cutter_paths.append(strip)

            if cutter_paths:
                disjoint_paths = clip_paths(
                    disjoint_paths, cutter_paths, pyclipper.CT_DIFFERENCE
                )

        out: List[PolylineInt] = []
        out.extend(open_polylines)
        for path in disjoint_paths:
            if len(path) < 3:
                continue
            points = [PointInt(int(x), int(y)) for (x, y) in path]
            if points[0] != points[-1]:
                points.append(points[0])
            out.append(PolylineInt(points=points))

        return out
