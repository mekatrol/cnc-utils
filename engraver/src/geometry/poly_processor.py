from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple
import math
import pyclipper
from geometry.PointInt import PointInt
from geometry.PolylineInt import PolylineInt
from geometry.Polygon import do_segments_intersect, is_point_on_segment, orientation


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

    @staticmethod
    def extract_faces_from_polylines(
        polylines: List[PolylineInt],
        snap_tol: int = 0,
    ) -> List[PolylineInt]:
        segments: List[Tuple[PointInt, PointInt]] = []
        for poly in polylines:
            pts = poly.points
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                a = pts[i]
                b = pts[i + 1]
                if a != b:
                    segments.append((a, b))

        if len(segments) < 3:
            return []

        def point_key(p: PointInt) -> Tuple[int, int]:
            return (p.x, p.y)

        def round_fraction(fr: Fraction) -> int:
            num = fr.numerator
            den = fr.denominator
            if den == 0:
                return 0
            if num >= 0:
                return (num + den // 2) // den
            return -((-num + den // 2) // den)

        def point_on_segment_tol(
            p: PointInt, a: PointInt, b: PointInt, tol: int
        ) -> bool:
            ax, ay = a.x, a.y
            bx, by = b.x, b.y
            px, py = p.x, p.y
            dx = bx - ax
            dy = by - ay
            if dx == 0 and dy == 0:
                return px == ax and py == ay
            cross = (px - ax) * dy - (py - ay) * dx
            if abs(cross) > tol * max(abs(dx), abs(dy), 1):
                return False
            minx = min(ax, bx) - tol
            maxx = max(ax, bx) + tol
            miny = min(ay, by) - tol
            maxy = max(ay, by) + tol
            return minx <= px <= maxx and miny <= py <= maxy

        def line_intersection(
            a: PointInt, b: PointInt, c: PointInt, d: PointInt
        ) -> Optional[PointInt]:
            ax, ay = a.x, a.y
            bx, by = b.x, b.y
            cx, cy = c.x, c.y
            dx, dy = d.x, d.y
            denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
            if denom == 0:
                return None
            det_ab = ax * by - ay * bx
            det_cd = cx * dy - cy * dx
            x = Fraction(det_ab * (cx - dx) - (ax - bx) * det_cd, denom)
            y = Fraction(det_ab * (cy - dy) - (ay - by) * det_cd, denom)
            return PointInt(round_fraction(x), round_fraction(y))

        def intersection_points(
            a: PointInt, b: PointInt, c: PointInt, d: PointInt
        ) -> List[PointInt]:
            if not do_segments_intersect(a, b, c, d):
                return []
            o1 = orientation(a, b, c)
            o2 = orientation(a, b, d)
            o3 = orientation(c, d, a)
            o4 = orientation(c, d, b)
            if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
                hits: List[PointInt] = []
                for p in (a, b, c, d):
                    if is_point_on_segment(a, b, p) and is_point_on_segment(c, d, p):
                        hits.append(p)
                return hits
            hit = line_intersection(a, b, c, d)
            if not hit:
                return []
            tol = max(0, snap_tol)
            if point_on_segment_tol(hit, a, b, tol) and point_on_segment_tol(
                hit, c, d, tol
            ):
                return [hit]
            return []

        split_points: List[Set[Tuple[int, int]]] = [
            {point_key(seg[0]), point_key(seg[1])} for seg in segments
        ]

        seg_bboxes: List[Tuple[int, int, int, int]] = []
        seg_lengths: List[float] = []
        for a, b in segments:
            seg_bboxes.append(
                (min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y))
            )
            seg_lengths.append(math.hypot(b.x - a.x, b.y - a.y))

        avg_len = sum(seg_lengths) / max(1, len(seg_lengths))
        grid_size = max(1, int(round(avg_len * 0.5)))
        if snap_tol > 0:
            grid_size = max(grid_size, snap_tol * 8)

        def grid_index(value: int) -> int:
            return math.floor(value / grid_size)

        grid: Dict[Tuple[int, int], List[int]] = {}
        for idx, (minx, miny, maxx, maxy) in enumerate(seg_bboxes):
            gx0 = grid_index(minx)
            gx1 = grid_index(maxx)
            gy0 = grid_index(miny)
            gy1 = grid_index(maxy)
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    grid.setdefault((gx, gy), []).append(idx)

        seen_pairs: Set[Tuple[int, int]] = set()
        for cell_indices in grid.values():
            count = len(cell_indices)
            if count < 2:
                continue
            for ii in range(count):
                for jj in range(ii + 1, count):
                    i = cell_indices[ii]
                    j = cell_indices[jj]
                    key = (i, j) if i < j else (j, i)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    a, b = segments[i]
                    c, d = segments[j]
                    hits = intersection_points(a, b, c, d)
                    if not hits:
                        continue
                    for p in hits:
                        key_pt = point_key(p)
                        split_points[i].add(key_pt)
                        split_points[j].add(key_pt)

        def sort_points_on_segment(
            a: PointInt, b: PointInt, keys: Set[Tuple[int, int]]
        ) -> List[PointInt]:
            dx = b.x - a.x
            dy = b.y - a.y
            if dx == 0 and dy == 0:
                return [a]
            if abs(dx) >= abs(dy):
                denom = dx if dx != 0 else 1
                items = sorted(
                    keys, key=lambda p: (p[0] - a.x) / denom
                )
            else:
                denom = dy if dy != 0 else 1
                items = sorted(
                    keys, key=lambda p: (p[1] - a.y) / denom
                )
            return [PointInt(x, y) for (x, y) in items]

        unique_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        adjacency: Dict[PointInt, Set[PointInt]] = {}

        for (a, b), keys in zip(segments, split_points):
            pts = sort_points_on_segment(a, b, keys)
            if len(pts) < 2:
                continue
            for i in range(len(pts) - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
                if p0 == p1:
                    continue
                k0 = point_key(p0)
                k1 = point_key(p1)
                edge_key = (k0, k1) if k0 <= k1 else (k1, k0)
                if edge_key in unique_edges:
                    continue
                unique_edges.add(edge_key)
                adjacency.setdefault(p0, set()).add(p1)
                adjacency.setdefault(p1, set()).add(p0)

        if not adjacency:
            return []

        def angle(from_pt: PointInt, to_pt: PointInt) -> float:
            return math.atan2(to_pt.y - from_pt.y, to_pt.x - from_pt.x)

        neighbors: Dict[PointInt, List[PointInt]] = {}
        neighbor_index: Dict[Tuple[PointInt, PointInt], int] = {}
        for node, nbrs in adjacency.items():
            ordered = sorted(nbrs, key=lambda p: angle(node, p))
            neighbors[node] = ordered
            for idx, nbr in enumerate(ordered):
                neighbor_index[(node, nbr)] = idx

        visited: Set[Tuple[PointInt, PointInt]] = set()
        faces: List[List[PointInt]] = []
        max_steps = max(1, len(unique_edges) * 2 + 1)

        def polygon_area(points: List[PointInt]) -> int:
            area2 = 0
            n = len(points)
            for i in range(n):
                p0 = points[i]
                p1 = points[(i + 1) % n]
                area2 += p0.x * p1.y - p1.x * p0.y
            return area2

        for u, nbrs in neighbors.items():
            for v in nbrs:
                if (u, v) in visited:
                    continue
                face: List[PointInt] = []
                curr_u, curr_v = u, v
                steps = 0
                while True:
                    visited.add((curr_u, curr_v))
                    face.append(curr_u)
                    next_candidates = neighbors.get(curr_v)
                    if not next_candidates:
                        break
                    idx = neighbor_index.get((curr_v, curr_u))
                    if idx is None:
                        break
                    next_idx = (idx + 1) % len(next_candidates)
                    next_v = next_candidates[next_idx]
                    curr_u, curr_v = curr_v, next_v
                    steps += 1
                    if curr_u == u and curr_v == v:
                        break
                    if steps > max_steps:
                        break
                if len(face) < 3:
                    continue
                if polygon_area(face) == 0:
                    continue
                faces.append(face)

        out: List[PolylineInt] = []
        for face in faces:
            if face[0] != face[-1]:
                face = face + [face[0]]
            out.append(PolylineInt(points=face))

        return out
