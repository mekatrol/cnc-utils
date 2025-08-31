import math
import re
import sys
from typing import List, Iterable, Optional
from svgpathtools import Path, CubicBezier, Arc, parse_path
from geometry.Utils import Utils
from geometry.GeometryInt import GeometryInt
from geometry.PolylineInt import PolylineInt
from geometry.PointInt import PointInt
import xml.etree.ElementTree as ET
from svg.Affine2D import Affine2D


class SvgConverter:
    """Parse an SVG and return integer-scaled polylines suitable for integer geometry algorithms.

    Args:
        svg_path: Path to the SVG file.
        scale: Scaling factor applied to all coordinates before rounding.
        tol: Geometric tolerance in the same units as the SVG (before scaling).
             Smaller = more segments when flattening curves.
    """
    @staticmethod
    def convert(svg_path: str, scale: int = 10000, tol: float = 0.25) -> GeometryInt:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        polylines_float: List[List[complex]] = []
        for poly in SvgConverter._iter_svg_geometry(root, tol=tol):
            if len(poly) >= 2:
                polylines_float.append(poly)

        # Convert to integer space
        polylines_int: List[PolylineInt] = []
        for poly in polylines_float:
            pts_i = [PointInt(
                Utils.float_to_int(p.real, scale),
                # We need to flip Z on conversion:
                #  * SVG defines the origin at the top-left, with +y downward.
                #  * Matplotlib 3D (and most CAD/CAM coordinate systems) assume +y upward and +z upward.
                -Utils.float_to_int(p.imag, scale)) for p in poly]
            polylines_int.append(PolylineInt(pts=pts_i))

        return GeometryInt(polylines=polylines_int, scale=scale)

    @staticmethod
    def _flatten_segment(segment, tol: float, t0: float = 0.0, t1: float = 1.0, depth: int = 0, max_depth: int = 18) -> List[complex]:
        """Recursively approximate any segment with a polyline within tolerance.
        Returns a list of complex points from t0..t1 (including endpoints).
        """
        p0 = segment.point(t0)
        p2 = segment.point(t1)
        pm = segment.point(0.5 * (t0 + t1))

        # Error as distance of midpoint to chord
        err = Utils.point_line_dist(pm, p0, p2)
        if err <= tol or depth >= max_depth:
            return [p0, p2]
        else:
            left = SvgConverter._flatten_segment(
                segment, tol, t0, 0.5 * (t0 + t1), depth + 1, max_depth)
            right = SvgConverter._flatten_segment(
                segment, tol, 0.5 * (t0 + t1), t1, depth + 1, max_depth)
            # Avoid duplicating the midpoint
            return left[:-1] + right

    @staticmethod
    def path_to_polylines(path: Path, tol: float = 0.25) -> List[List[complex]]:
        """Convert an svgpathtools Path into a list of polylines (complex points)."""
        polylines: List[List[complex]] = []
        current: List[complex] = []
        last_end: Optional[complex] = None

        for seg in path:
            # Convert arcs to cubic(s) first for robust flattening
            segments: List = []
            if isinstance(seg, Arc):
                segments = [CubicBezier(
                    *c) if isinstance(c, CubicBezier) else c for c in seg.as_cubic_curves()]
            else:
                segments = [seg]

            for s in segments:
                pts = SvgConverter._flatten_segment(s, tol)
                if not current:
                    current.extend(pts)
                else:
                    # If the new segment continues from the last point, append; otherwise start a new polyline
                    if last_end is not None and abs(pts[0] - last_end) < 1e-9:
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
    def _circle_to_polyline(cx: float, cy: float, r: float, segments: int = 64) -> List[complex]:
        pts = []
        for i in range(segments + 1):
            th = 2 * math.pi * i / segments
            pts.append(complex(cx + r * math.cos(th), cy + r * math.sin(th)))
        return pts

    @staticmethod
    def _ellipse_to_polyline(cx: float, cy: float, rx: float, ry: float, segments: int = 96) -> List[complex]:
        pts = []
        for i in range(segments + 1):
            th = 2 * math.pi * i / segments
            pts.append(complex(cx + rx * math.cos(th), cy + ry * math.sin(th)))
        return pts

    @staticmethod
    def _rect_to_polyline(x: float, y: float, w: float, h: float) -> List[complex]:
        return [
            complex(x, y),
            complex(x + w, y),
            complex(x + w, y + h),
            complex(x, y + h),
            complex(x, y),
        ]

    @staticmethod
    def _apply_transform_to_polyline(poly: List[complex], T: Affine2D) -> List[complex]:
        return [complex(*T.apply(p.real, p.imag)) for p in poly]

    @staticmethod
    def _extract_style_visibility(el: ET.Element) -> bool:
        """Return False if element is explicitly hidden via display:none or visibility:hidden."""
        style = el.attrib.get("style", "")
        if "display:none" in style.replace(" ", "").lower():
            return False
        if "visibility:hidden" in style.replace(" ", "").lower():
            return False
        if el.attrib.get("display", "").lower() == "none":
            return False
        if el.attrib.get("visibility", "").lower() == "hidden":
            return False
        return True

    @staticmethod
    def _iter_svg_geometry(root: ET.Element, tol: float) -> Iterable[List[complex]]:
        """Yield polylines from supported SVG elements (already filtered by visibility)."""
        ns = ""
        # Handle namespaces (strip common "http://www.w3.org/2000/svg")
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        def walk(el: ET.Element, ctm: Affine2D):
            if not SvgConverter._extract_style_visibility(el):
                return

            T = ctm
            if "transform" in el.attrib:
                T = ctm @ Affine2D.from_svg_transform(el.attrib["transform"])

            tag = el.tag
            if tag.startswith("{"):
                tag = tag.split("}", 1)[1]

            if tag == "g":
                for ch in list(el):
                    walk(ch, T)
                return

            # ----- <path>
            if tag == "path" and "d" in el.attrib:
                try:
                    p = parse_path(el.attrib["d"])
                except Exception as e:
                    print(f"[WARN] Failed to parse <path>: {e}", file=sys.stderr)
                    p = None
                if p:
                    # Flatten entire path
                    polylines = SvgConverter.path_to_polylines(p, tol=tol)
                    for poly in polylines:
                        yield SvgConverter._apply_transform_to_polyline(poly, T)

            # ----- <polyline> / <polygon>
            elif tag in ("polyline", "polygon") and "points" in el.attrib:
                pts_txt = el.attrib["points"].strip()
                # Robust split for points="x1,y1 x2,y2 ..."
                nums = [float(n) for n in re.split(r"[ ,]+", pts_txt.strip()) if n]
                if len(nums) % 2 == 0:
                    pts = [complex(nums[i], nums[i + 1])
                           for i in range(0, len(nums), 2)]
                    if tag == "polygon" and (len(pts) == 0 or pts[0] != pts[-1]):
                        pts.append(pts[0])
                    if len(pts) >= 2:
                        yield SvgConverter._apply_transform_to_polyline(pts, T)

            # ----- <line>
            elif tag == "line":
                x1 = Utils.safe_float(el.attrib.get("x1", "0"))
                y1 = Utils.safe_float(el.attrib.get("y1", "0"))
                x2 = Utils.safe_float(el.attrib.get("x2", "0"))
                y2 = Utils.safe_float(el.attrib.get("y2", "0"))
                pts = [complex(x1, y1), complex(x2, y2)]
                yield SvgConverter._apply_transform_to_polyline(pts, T)

            # ----- <rect> (no rounded corners in this minimal version)
            elif tag == "rect":
                x = Utils.safe_float(el.attrib.get("x", "0"))
                y = Utils.safe_float(el.attrib.get("y", "0"))
                w = Utils.safe_float(el.attrib.get("width", "0"))
                h = Utils.safe_float(el.attrib.get("height", "0"))
                if w > 0 and h > 0:
                    pts = SvgConverter._rect_to_polyline(x, y, w, h)
                    yield SvgConverter._apply_transform_to_polyline(pts, T)

            # ----- <circle>
            elif tag == "circle":
                cx = Utils.safe_float(el.attrib.get("cx", "0"))
                cy = Utils.safe_float(el.attrib.get("cy", "0"))
                r = Utils.safe_float(el.attrib.get("r", "0"))
                if r > 0:
                    pts = SvgConverter._circle_to_polyline(cx, cy, r)
                    yield SvgConverter._apply_transform_to_polyline(pts, T)

            # ----- <ellipse>
            elif tag == "ellipse":
                cx = Utils.safe_float(el.attrib.get("cx", "0"))
                cy = Utils.safe_float(el.attrib.get("cy", "0"))
                rx = Utils.safe_float(el.attrib.get("rx", "0"))
                ry = Utils.safe_float(el.attrib.get("ry", "0"))
                if rx > 0 and ry > 0:
                    pts = SvgConverter._ellipse_to_polyline(cx, cy, rx, ry)
                    yield SvgConverter._apply_transform_to_polyline(pts, T)

            # Ignore others for now (images, text, etc.)

        # Kick off traversal
        for child in list(root):
            yield from walk(child, Affine2D.identity())
