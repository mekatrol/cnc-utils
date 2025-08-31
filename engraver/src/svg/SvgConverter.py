import math
from typing import Any, Iterable, List, Optional, Tuple, cast
from geometry.GeometryUtils import GeometryUtils
from geometry.GeometryInt import GeometryInt
from geometry.PolylineInt import PolylineInt
from geometry.PointInt import PointInt
from svgelements import SVG, Path, Line, Rect, Circle, Ellipse, Polyline, Polygon, Group, Matrix, Close


class SvgConverter:
    """Parse an SVG and return integer-scaled polylines suitable for integer geometry algorithms.

    Args:
        svg_path: Path to the SVG file.
        scale: Scaling factor applied to all coordinates before rounding.
        tol: Geometric tolerance in the same units as the SVG (before scaling).
             Smaller = more segments when flattening curves.
    """

    @staticmethod
    def _svf(x: Any, default: float = 0.0) -> float:
        """Safe value->float for svgelements Length/None/Unknown."""
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            try:
                return float(getattr(x, "value", default))
            except Exception:
                return default

    @staticmethod
    def _scomplex(pt: Any) -> Optional[complex]:
        """Safe any->complex for svgelements points/tuples/None."""
        if pt is None:
            return None
        try:
            return complex(pt)
        except Exception:
            try:
                x, y = pt  # tuple-like
                return complex(SvgConverter._svf(x), SvgConverter._svf(y))
            except Exception:
                return None

    @staticmethod
    def _walk_parents(node: Any, parent: Optional[Matrix] = None) -> Iterable[Tuple[Any, Matrix, Matrix]]:
        """Yield (leaf_element, parent_matrix_without_leaf, leaf_transform_matrix)."""
        P = Matrix() if parent is None else parent
        t = getattr(node, "transform", None)
        t = t if isinstance(t, Matrix) else Matrix()

        is_leaf = isinstance(node, (Path, Line, Rect, Circle, Ellipse, Polyline, Polygon))
        if is_leaf:
            # do not fold leaf's own transform into P
            yield node, P, t
            return
        # containers: fold their transform into P only
        P2 = P * t  # NOTE: order matters; this matches svgelements (parent then child)
        if hasattr(node, "__iter__"):
            for ch in node:
                yield from SvgConverter._walk_parents(ch, P2)

    @staticmethod
    def _walk(node: Any, M: Optional[Matrix] = None) -> Iterable[Path]:
        mat = Matrix() if M is None else M
        t = getattr(node, "transform", None)
        if not isinstance(t, Matrix):
            t = Matrix()
        mat = mat * t

        # Recurse into any container: SVG, Group, Symbol, etc. (they're iterable)
        if hasattr(node, "__iter__"):
            for e in node:
                yield from SvgConverter._walk(e, mat)

        # Convert any shape-like to a Path
        if isinstance(node, Path):
            yield node * mat
        elif hasattr(node, "as_path"):
            p = node.as_path()
            if p is not None:
                yield p * mat

    @staticmethod
    def _cont(a: complex, b: complex, abs_tol: float) -> bool:
        if a is None or b is None:
            return False
        d = abs(a - b)
        m = max(abs(a), abs(b), 1.0)
        return d <= max(abs_tol, 1e-6 * m)

    @staticmethod
    def _path_to_polylines_se(path: Path, chord_tol: float) -> List[List[complex]]:
        polylines: List[List[complex]] = []
        current: List[complex] = []
        last_end: Optional[complex] = None
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
            if current and SvgConverter._cont(last_end, pts[0], abs_tol):
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
    def _circle_poly(cx: float, cy: float, r: float, tol: float) -> List[complex]:
        if r <= 0:
            return []
        # ensure chord <= tol
        dtheta = 2 * math.asin(min(1.0, max(tol, 1e-9) / (2 * r)))
        n = max(8, int(math.ceil(2 * math.pi / max(dtheta, 1e-6))))
        return [complex(cx + r * math.cos(2 * math.pi * i / n),
                        cy + r * math.sin(2 * math.pi * i / n)) for i in range(n+1)]

    @staticmethod
    def _ellipse_poly(cx: float, cy: float, rx: float, ry: float, tol: float) -> List[complex]:
        if rx <= 0 or ry <= 0:
            return []
        # use max radius for conservative sampling
        r = max(rx, ry)
        dtheta = 2 * math.asin(min(1.0, max(tol, 1e-9) / (2 * r)))
        n = max(12, int(math.ceil(2 * math.pi / max(dtheta, 1e-6))))
        return [complex(cx + rx * math.cos(2 * math.pi * i / n),
                        cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n+1)]

    @staticmethod
    def _rect_poly(x: float, y: float, w: float, h: float) -> List[complex]:
        if w <= 0 or h <= 0:
            return []
        return [complex(x, y), complex(x + w, y), complex(x + w, y + h),
                complex(x, y + h), complex(x, y)]

    @staticmethod
    def _rounded_rect_poly(x: float, y: float, w: float, h: float, rx: float, ry: float, tol: float) -> List[complex]:
        # quarter arcs with rx, ry
        if rx <= 0 and ry <= 0:
            return SvgConverter._rect_poly(x, y, w, h)
        rx = min(rx, w / 2.0)
        ry = min(ry, h / 2.0)
        # sample each corner arc

        def arc(cx, cy, rx, ry, a0, a1):
            # conservative sampling by max radius
            rmax = max(rx, ry)
            dtheta = 2 * math.asin(min(1.0, max(tol, 1e-9) / (2 * rmax)))
            n = max(3, int(math.ceil(abs(a1 - a0) / max(dtheta, 1e-6))))
            return [complex(cx + rx * math.cos(a0 + (a1 - a0) * i / n),
                            cy + ry * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n+1)]
        pts: List[complex] = []
        # start at top-left corner after rounding
        pts.append(complex(x + rx, y))
        pts.append(complex(x + w - rx, y))
        pts.extend(arc(x + w - rx, y + ry, rx, ry, -math.pi/2, 0)[1:])
        pts.append(complex(x + w, y + h - ry))
        pts.extend(arc(x + w - rx, y + h - ry, rx, ry, 0, math.pi/2)[1:])
        pts.append(complex(x + rx, y + h))
        pts.extend(arc(x + rx, y + h - ry, rx, ry, math.pi/2, math.pi)[1:])
        pts.append(complex(x, y + ry))
        pts.extend(arc(x + rx, y + ry, rx, ry, math.pi, 3*math.pi/2)[1:])
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
    def _walk_with_matrix(node: Any, M: Optional[Matrix] = None) -> Iterable[Tuple[Any, Matrix]]:
        mat = Matrix() if M is None else M
        t = getattr(node, "transform", None)
        if isinstance(t, Matrix):
            mat = mat * t
        # Recurse into containers (SVG/Group/etc.), not into leaf shapes
        if hasattr(node, "__iter__") and not isinstance(
            node, (Path, Line, Rect, Circle, Ellipse, Polyline, Polygon)
        ):
            for ch in node:
                yield from SvgConverter._walk_with_matrix(ch, mat)
        else:
            yield node, mat

    @staticmethod
    def _apply_matrix_to_points(pts: List[complex], M: Matrix) -> List[complex]:
        a = getattr(M, "a", 1.0)
        b = getattr(M, "b", 0.0)
        c = getattr(M, "c", 0.0)
        d = getattr(M, "d", 1.0)
        e = getattr(M, "e", 0.0)
        f = getattr(M, "f", 0.0)
        out: List[complex] = []
        for p in pts:
            x, y = p.real, p.imag
            out.append(complex(a * x + c * y + e, b * x + d * y + f))
        return out

    @staticmethod
    def svg_to_geometry_int(svg_path: str, scale: int = 10000, tol: float = 0.25) -> GeometryInt:
        doc = SVG.parse(svg_path)
        polylines_float: List[List[complex]] = []

        for elem, P, Tleaf in SvgConverter._walk_parents(doc):
            M = P * Tleaf  # final matrix for this leaf only

            if isinstance(elem, Path):
                # transform geometry once
                polylines_float.extend(SvgConverter._path_to_polylines_se(elem * M, chord_tol=tol))
                continue

            # sample primitives, then apply M once
            if isinstance(elem, Line):
                a = SvgConverter._scomplex(getattr(elem, "start", None))
                b = SvgConverter._scomplex(getattr(elem, "end", None))
                if a is not None and b is not None:
                    pts = SvgConverter._apply_matrix_to_points([a, b], M)
                    polylines_float.append(pts)

            elif isinstance(elem, Rect):
                x = SvgConverter._svf(getattr(elem, "x", None))
                y = SvgConverter._svf(getattr(elem, "y", None))
                w = SvgConverter._svf(getattr(elem, "width", None))
                h = SvgConverter._svf(getattr(elem, "height", None))
                rx = SvgConverter._svf(getattr(elem, "rx", None))
                ry = SvgConverter._svf(getattr(elem, "ry", None))
                pts = (SvgConverter._rounded_rect_poly(x, y, w, h, rx, ry, tol)
                       if (rx or ry) else SvgConverter._rect_poly(x, y, w, h))
                if pts:
                    polylines_float.append(SvgConverter._apply_matrix_to_points(pts, M))

            elif isinstance(elem, Circle):
                cx = SvgConverter._svf(SvgConverter._get_attr(elem, "cx", "center_x"))
                cy = SvgConverter._svf(SvgConverter._get_attr(elem, "cy", "center_y"))
                r = SvgConverter._svf(SvgConverter._get_attr(elem, "r", "radius", "rx", "ry"))
                if r > 0:
                    pts = SvgConverter._circle_poly(cx, cy, r, tol)
                    polylines_float.append(SvgConverter._apply_matrix_to_points(pts, M))

            elif isinstance(elem, Ellipse):
                cx = SvgConverter._svf(SvgConverter._get_attr(elem, "cx", "center_x"))
                cy = SvgConverter._svf(SvgConverter._get_attr(elem, "cy", "center_y"))
                rx = SvgConverter._svf(SvgConverter._get_attr(elem, "rx", "radius_x"))
                ry = SvgConverter._svf(SvgConverter._get_attr(elem, "ry", "radius_y"))
                if rx <= 0 and ry <= 0:
                    r = SvgConverter._svf(SvgConverter._get_attr(elem, "r", "radius"))
                    rx = ry = r
                if rx > 0 and ry > 0:
                    pts = SvgConverter._ellipse_poly(cx, cy, rx, ry, tol)
                    polylines_float.append(SvgConverter._apply_matrix_to_points(pts, M))

            elif isinstance(elem, Polyline):
                raw_pts = cast(Iterable[Any], getattr(elem, "points", ()))
                pts = [p for p in (SvgConverter._scomplex(p) for p in raw_pts) if p is not None]
                if len(pts) >= 2:
                    polylines_float.append(SvgConverter._apply_matrix_to_points(pts, M))

            elif isinstance(elem, Polygon):
                raw_pts = cast(Iterable[Any], getattr(elem, "points", ()))
                pts = [p for p in (SvgConverter._scomplex(p) for p in raw_pts) if p is not None]
                if len(pts) >= 2 and pts[0] != pts[-1]:
                    pts.append(pts[0])
                if len(pts) >= 2:
                    polylines_float.append(SvgConverter._apply_matrix_to_points(pts, M))

        # Integerize
        polylines_int: List[PolylineInt] = []
        for poly in polylines_float:
            pts_i = [PointInt(GeometryUtils.float_to_int(pt.real, scale),
                              -GeometryUtils.float_to_int(pt.imag, scale))
                     for pt in poly]
            if len(pts_i) >= 2:
                polylines_int.append(PolylineInt(pts=pts_i))

        return GeometryInt(polylines=polylines_int, scale=scale)
