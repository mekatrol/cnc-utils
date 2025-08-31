import math
from typing import List, Optional
from geometry.GeometryUtils import GeometryUtils
from geometry.GeometryInt import GeometryInt
from geometry.PolylineInt import PolylineInt
from geometry.PointInt import PointInt
import xml.etree.ElementTree as ET
from svgelements import SVG, Path as SEPath, Group, Matrix, Close


# add near imports
from dataclasses import dataclass, field
from copy import deepcopy


class SvgConverter:
    """Parse an SVG and return integer-scaled polylines suitable for integer geometry algorithms.

    Args:
        svg_path: Path to the SVG file.
        scale: Scaling factor applied to all coordinates before rounding.
        tol: Geometric tolerance in the same units as the SVG (before scaling).
             Smaller = more segments when flattening curves.
    """
    @staticmethod
    def _walk(node, M: Optional[Matrix] = None):
        if M is None:
            M = Matrix()
        M = M * getattr(node, "transform", Matrix())
        # Descend groups
        if isinstance(node, Group):
            for e in node:
                yield from SvgConverter._walk(e, M)
            return
        # Anything path-like
        path = None
        if isinstance(node, SEPath):
            path = node
        elif hasattr(node, "as_path"):
            path = node.as_path()
        if path is not None:
            # apply transform
            path = path * M
            yield path

    @staticmethod
    def _cont(a: complex, b: complex, abs_tol: float) -> bool:
        if a is None or b is None:
            return False
        d = abs(a - b)
        m = max(abs(a), abs(b), 1.0)
        return d <= max(abs_tol, 1e-9 * m)

    @staticmethod
    def _path_to_polylines_se(path: SEPath, chord_tol: float) -> List[List[complex]]:
        polylines: List[List[complex]] = []
        current: List[complex] = []
        last_end: Optional[complex] = None
        abs_tol = max(1e-6, chord_tol)

        for seg in path:
            # handle Close explicitly
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
    def svg_to_geometry_int(svg_path: str, scale: int = 10000, tol: float = 0.25) -> GeometryInt:
        """tol: max chord length in SVG units."""
        svg = SVG.parse(svg_path)
        polys_float: List[List[complex]] = []
        for path in SvgConverter._walk(svg):
            polys_float.extend(SvgConverter._path_to_polylines_se(path, chord_tol=tol))

        polylines_int: List[PolylineInt] = []
        for poly in polys_float:
            pts_i = [
                PointInt(
                    GeometryUtils.float_to_int(p.real, scale),
                    -GeometryUtils.float_to_int(p.imag, scale),
                )
                for p in poly
            ]
            if len(pts_i) >= 2:
                polylines_int.append(PolylineInt(pts=pts_i))

        return GeometryInt(polylines=polylines_int, scale=scale)
