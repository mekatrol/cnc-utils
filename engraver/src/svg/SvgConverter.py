import math
from typing import Any, Iterable, List, Optional, cast
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

        return GeometryInt(polylines=polylines_int, points=[], scale=scale)
