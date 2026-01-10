import math
from typing import Any, Iterable, List, Optional, cast
import pyclipper
from svgelements import (
    SVG,
    Path,
    Line,
    SimpleLine,
    Rect,
    Circle,
    Ellipse,
    Polyline,
    Polygon,
    Matrix,
    Close,
)
from geometry.GeoUtil import GeoUtil
from geometry.GeometryInt import GeometryInt
from geometry.PolylineInt import PolylineInt
from geometry.PointInt import PointInt
from geometry.PointFloat import PointFloat


class SvgConverter:
    """SVG -> integer-scaled polylines (GeometryInt)."""

    @staticmethod
    def _split_self_intersections(
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
    def _split_intersections_between_polygons(
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

            n = len(updated)
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
    def _walk_with_matrix(node: Any, parent: Optional[Matrix] = None):
        """Yield (leaf, parent_matrix_without_leaf)."""
        parent_matrix = Matrix() if parent is None else parent
        node_matrix = getattr(node, "transform", None)
        node_matrix = node_matrix if isinstance(node_matrix, Matrix) else Matrix()

        is_leaf = isinstance(
            node, (Path, Line, SimpleLine, Rect, Circle, Ellipse, Polyline, Polygon)
        )
        if is_leaf:
            # important: do NOT fold the leaf's own transform here
            yield node, parent_matrix
            return

        if hasattr(node, "__iter__"):
            for ch in node:
                yield from SvgConverter._walk_with_matrix(ch, parent_matrix)

    @staticmethod
    def _apply_matrix_to_points(pts: List[PointFloat], M: Matrix) -> List[PointFloat]:
        # SVG matrix: [a c e; b d f; 0 0 1], column-vector on the right.
        a = getattr(M, "a", 1.0)
        b = getattr(M, "b", 0.0)
        c = getattr(M, "c", 0.0)
        d = getattr(M, "d", 1.0)
        e = getattr(M, "e", 0.0)
        f = getattr(M, "f", 0.0)
        out: List[PointFloat] = []
        for p in pts:
            x, y = p.x, p.y
            out.append(PointFloat(a * x + c * y + e, b * x + d * y + f))
        return out

    @staticmethod
    def _path_to_polylines(path: Path, chord_tol: float) -> List[List[PointFloat]]:
        polylines: List[List[PointFloat]] = []
        current: List[PointFloat] = []
        last_end: Optional[PointFloat] = None
        abs_tol = max(1e-6, chord_tol)

        for seg in path:
            if isinstance(seg, Close):
                pts = [seg.start, seg.end]
            else:
                L = max(seg.length(error=1e-4), 0.0)
                n = max(2, int(math.ceil(L / max(chord_tol, 1e-9))))
                pts = [seg.point(i / (n - 1)) for i in range(n)]
            if len(pts) < 2:
                continue
            if current and GeoUtil.equal_with_tolerance(last_end, pts[0], abs_tol):
                current.extend(pts[1:])
            else:
                if len(current) >= 2:
                    polylines.append(current)
                current = pts
            last_end = pts[-1]

        if len(current) >= 2:
            polylines.append(current)
        return polylines

    @staticmethod
    def _circle_poly(cx: float, cy: float, r: float, tol: float) -> List[PointFloat]:
        if r <= 0:
            return []
        dtheta = 2 * math.asin(min(1.0, max(tol, 1e-9) / (2 * r)))
        n = max(8, int(math.ceil(2 * math.pi / max(dtheta, 1e-6))))
        return [
            PointFloat(
                cx + r * math.cos(2 * math.pi * i / n),
                cy + r * math.sin(2 * math.pi * i / n),
            )
            for i in range(n + 1)
        ]

    @staticmethod
    def _ellipse_poly(
        cx: float, cy: float, rx: float, ry: float, tol: float
    ) -> List[PointFloat]:
        if rx <= 0 or ry <= 0:
            return []
        r = max(rx, ry)
        dtheta = 2 * math.asin(min(1.0, max(tol, 1e-9) / (2 * r)))
        n = max(12, int(math.ceil(2 * math.pi / max(dtheta, 1e-6))))
        return [
            PointFloat(
                cx + rx * math.cos(2 * math.pi * i / n),
                cy + ry * math.sin(2 * math.pi * i / n),
            )
            for i in range(n + 1)
        ]

    @staticmethod
    def _rect_poly(x: float, y: float, w: float, h: float) -> List[PointFloat]:
        if w <= 0 or h <= 0:
            return []
        return [
            PointFloat(x, y),
            PointFloat(x + w, y),
            PointFloat(x + w, y + h),
            PointFloat(x, y + h),
            PointFloat(x, y),
        ]

    @staticmethod
    def _rounded_rect_poly(
        x: float, y: float, w: float, h: float, rx: float, ry: float, tol: float
    ) -> List[PointFloat]:
        if w <= 0 or h <= 0:
            return []
        if rx <= 0 and ry <= 0:
            return SvgConverter._rect_poly(x, y, w, h)
        rx = min(rx, w / 2.0)
        ry = min(ry, h / 2.0)

        def arc(cx, cy, rx, ry, a0, a1):
            rmax = max(rx, ry)
            dtheta = 2 * math.asin(min(1.0, max(tol, 1e-9) / (2 * rmax)))
            n = max(3, int(math.ceil(abs(a1 - a0) / max(dtheta, 1e-6))))
            return [
                PointFloat(
                    cx + rx * math.cos(a0 + (a1 - a0) * i / n),
                    cy + ry * math.sin(a0 + (a1 - a0) * i / n),
                )
                for i in range(n + 1)
            ]

        pts: List[PointFloat] = []
        pts.append(PointFloat(x + rx, y))
        pts.append(PointFloat(x + w - rx, y))
        pts.extend(arc(x + w - rx, y + ry, rx, ry, -math.pi / 2, 0)[1:])
        pts.append(PointFloat(x + w, y + h - ry))
        pts.extend(arc(x + w - rx, y + h - ry, rx, ry, 0, math.pi / 2)[1:])
        pts.append(PointFloat(x + rx, y + h))
        pts.extend(arc(x + rx, y + h - ry, rx, ry, math.pi / 2, math.pi)[1:])
        pts.append(PointFloat(x, y + ry))
        pts.extend(arc(x + rx, y + ry, rx, ry, math.pi, 3 * math.pi / 2)[1:])
        pts.append(pts[0])
        return pts

    @staticmethod
    def _get_attr(o, *names):
        for n in names:
            v = getattr(o, n, None)
            if v is not None:
                return v
        return None

    @staticmethod
    def _parse_viewbox(vb: Any) -> Optional[List[float]]:
        if vb is None:
            return None
        if isinstance(vb, (list, tuple)) and len(vb) == 4:
            return [GeoUtil.safe_to_float(v, None) for v in vb]
        if all(hasattr(vb, k) for k in ("x", "y", "width", "height")):
            return [
                GeoUtil.safe_to_float(getattr(vb, "x"), None),
                GeoUtil.safe_to_float(getattr(vb, "y"), None),
                GeoUtil.safe_to_float(getattr(vb, "width"), None),
                GeoUtil.safe_to_float(getattr(vb, "height"), None),
            ]
        if isinstance(vb, str):
            parts = [p for p in vb.replace(",", " ").split() if p]
            if len(parts) == 4:
                return [GeoUtil.safe_to_float(p, None) for p in parts]
        return None

    @staticmethod
    def _viewbox_matrix(doc: SVG) -> Matrix:
        vb = getattr(doc, "viewbox", None) or getattr(doc, "viewBox", None)
        vb_vals = SvgConverter._parse_viewbox(vb)
        if not vb_vals or any(v is None for v in vb_vals):
            return Matrix()
        min_x, min_y, vb_w, vb_h = vb_vals
        if not vb_w or not vb_h:
            return Matrix()

        width = GeoUtil.safe_to_float(getattr(doc, "width", None), None)
        height = GeoUtil.safe_to_float(getattr(doc, "height", None), None)
        sx = (width / vb_w) if width not in (None, 0) else 1.0
        sy = (height / vb_h) if height not in (None, 0) else 1.0

        # Map viewBox coords into viewport: (x - min_x) * sx, (y - min_y) * sy
        return Matrix(sx, 0, 0, sy, -min_x * sx, -min_y * sy)

    @staticmethod
    def svg_to_geometry_int(
        svg_path: str, scale: int = 10000, tol: float = 0.25
    ) -> GeometryInt:
        doc = SVG.parse(svg_path)
        viewbox_matrix = SvgConverter._viewbox_matrix(doc)

        polylines_float: List[List[PointFloat]] = []

        for elem, M in SvgConverter._walk_with_matrix(doc, viewbox_matrix):
            t = (
                getattr(elem, "transform", Matrix())
                if isinstance(getattr(elem, "transform", None), Matrix)
                else Matrix()
            )
            M = M * t

            if isinstance(elem, Path):
                # Discretize in local coords, then transform sampled points.
                for poly in SvgConverter._path_to_polylines(elem, chord_tol=tol):
                    polylines_float.append(
                        SvgConverter._apply_matrix_to_points(poly, M)
                    )

            elif isinstance(elem, Line):
                a = GeoUtil.safe_to_point(getattr(elem, "start", None))
                b = GeoUtil.safe_to_point(getattr(elem, "end", None))
                if a and b:
                    polylines_float.append(
                        SvgConverter._apply_matrix_to_points([a, b], M)
                    )

            elif isinstance(elem, SimpleLine):
                x1 = GeoUtil.safe_to_float(getattr(elem, "x1", None))
                y1 = GeoUtil.safe_to_float(getattr(elem, "y1", None))
                x2 = GeoUtil.safe_to_float(getattr(elem, "x2", None))
                y2 = GeoUtil.safe_to_float(getattr(elem, "y2", None))
                if None not in (x1, y1, x2, y2):
                    polylines_float.append(
                        SvgConverter._apply_matrix_to_points(
                            [PointFloat(x1, y1), PointFloat(x2, y2)], M
                        )
                    )

            elif isinstance(elem, Rect):
                x = GeoUtil.safe_to_float(getattr(elem, "x", None))
                y = GeoUtil.safe_to_float(getattr(elem, "y", None))
                w = GeoUtil.safe_to_float(getattr(elem, "width", None))
                h = GeoUtil.safe_to_float(getattr(elem, "height", None))
                rx = GeoUtil.safe_to_float(getattr(elem, "rx", None)) or 0.0
                ry = GeoUtil.safe_to_float(getattr(elem, "ry", None)) or 0.0
                if None not in (x, y, w, h) and w > 0 and h > 0:
                    pts = (
                        SvgConverter._rounded_rect_poly(x, y, w, h, rx, ry, tol)
                        if (rx > 0 or ry > 0)
                        else SvgConverter._rect_poly(x, y, w, h)
                    )
                    polylines_float.append(SvgConverter._apply_matrix_to_points(pts, M))

            elif isinstance(elem, Circle):
                cx = GeoUtil.safe_to_float(
                    SvgConverter._get_attr(elem, "cx", "center_x")
                )
                cy = GeoUtil.safe_to_float(
                    SvgConverter._get_attr(elem, "cy", "center_y")
                )
                r = GeoUtil.safe_to_float(
                    SvgConverter._get_attr(elem, "r", "radius", "rx", "ry")
                )
                if None not in (cx, cy, r) and r > 0:
                    pts = SvgConverter._circle_poly(cx, cy, r, tol)
                    polylines_float.append(SvgConverter._apply_matrix_to_points(pts, M))

            elif isinstance(elem, Ellipse):
                cx = GeoUtil.safe_to_float(
                    SvgConverter._get_attr(elem, "cx", "center_x")
                )
                cy = GeoUtil.safe_to_float(
                    SvgConverter._get_attr(elem, "cy", "center_y")
                )
                rx = GeoUtil.safe_to_float(
                    SvgConverter._get_attr(elem, "rx", "radius_x")
                )
                ry = GeoUtil.safe_to_float(
                    SvgConverter._get_attr(elem, "ry", "radius_y")
                )
                if rx is None or ry is None:
                    r = GeoUtil.safe_to_float(
                        SvgConverter._get_attr(elem, "r", "radius")
                    )
                    if r is not None:
                        rx = ry = r
                if None not in (cx, cy, rx, ry) and rx > 0 and ry > 0:
                    pts = SvgConverter._ellipse_poly(cx, cy, rx, ry, tol)
                    polylines_float.append(SvgConverter._apply_matrix_to_points(pts, M))

            elif isinstance(elem, Polygon):
                raw = cast(Iterable[Any], getattr(elem, "points", ()))
                pts = [p for p in (GeoUtil.safe_to_point(p) for p in raw) if p]
                if len(pts) >= 2:
                    if pts[0] != pts[-1]:
                        pts.append(pts[0])
                    polylines_float.append(SvgConverter._apply_matrix_to_points(pts, M))

        # Integerize (and flip Y)
        polylines_int: List[PolylineInt] = []
        close_tol = max(1e-6, tol)
        for poly in polylines_float:
            is_closed = (
                len(poly) >= 3
                and GeoUtil.equal_with_tolerance(poly[0], poly[-1], close_tol)
            )
            pts_i = [
                PointInt(
                    GeoUtil.float_to_int(pt.x, scale),
                    -GeoUtil.float_to_int(pt.y, scale),
                )
                for pt in poly
            ]
            if len(pts_i) >= 2:
                if is_closed:
                    pts_i[-1] = pts_i[0]
                polylines_int.append(PolylineInt(points=pts_i))

        polylines_int = SvgConverter._split_self_intersections(polylines_int)
        polylines_int = SvgConverter._split_intersections_between_polygons(
            polylines_int, scale
        )

        return GeometryInt(polylines=polylines_int, points=[], scale=scale)
