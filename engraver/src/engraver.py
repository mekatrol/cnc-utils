
"""
engraver.py

Minimal CAM-oriented SVG loader + 3D visualizer with integer scaling.
- Loads common 2D SVG geometry (<path>, <polyline>, <polygon>, <line>, <rect>, <circle>, <ellipse>)
- Flattens curves to polylines (adaptive tolerance)
- Converts coordinates to scaled integers (default scale=10000) for robust integer geometry
- Visualizes the 2D geometry in a 3D view (z=0) for rotation/inspection
- Can export integer toolpaths as JSON or TXT (one polyline per line)
- Provides simple Point/Vector integer primitives

Dependencies:
  pip install svgpathtools matplotlib numpy

Usage:
  python engraver.py input.svg --scale 10000 --tol 0.1
  # Rotate with mouse (matplotlib 3D), press 'r' to reset view, 'g' to toggle grid.
  # Export:
  #   --export-json out.json     # or '-' for stdout
  #   --export-txt out.txt      # or '-' for stdout

Notes:
  * Units are treated as pixels; CSS and mm/in conversions are not handled here.
  * Transform attributes (translate/scale/rotate/matrix) are supported.
  * NURBS are ignored (user explicitly requested to ignore). Arcs/Bezier are flattened.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Sequence

import numpy as np
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, parse_path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - ensures 3D support is loaded
import matplotlib.pyplot as plt


# -----------------------------
# Integer primitives
# -----------------------------

@dataclass(frozen=True)
class IntPoint:
    x: int
    y: int

    def __add__(self, other: "IntVector") -> "IntPoint":
        return IntPoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "IntPoint") -> "IntVector":
        return IntVector(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class IntVector:
    x: int
    y: int

    def dot(self, other: "IntVector") -> int:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "IntVector") -> int:
        # 2D cross product (z-component)
        return self.x * other.y - self.y * other.x

    def __add__(self, other: "IntVector") -> "IntVector":
        return IntVector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "IntVector") -> "IntVector":
        return IntVector(self.x - other.x, self.y - other.y)

    def scale(self, k: int) -> "IntVector":
        return IntVector(self.x * k, self.y * k)


# -----------------------------
# Affine transforms
# -----------------------------

class Affine2D:
    """Simple 2D affine transform: 3x3 homogeneous matrix.

    Stored as numpy array with shape (3,3). Points are column vectors [x,y,1]^T.
    """

    def __init__(self, m: Optional[np.ndarray] = None):
        if m is None:
            self.m = np.eye(3, dtype=float)
        else:
            self.m = np.array(m, dtype=float).reshape(3, 3)

    def __matmul__(self, other: "Affine2D") -> "Affine2D":
        return Affine2D(self.m @ other.m)

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        v = np.array([x, y, 1.0], dtype=float)
        res = self.m @ v
        return (res[0], res[1])

    @staticmethod
    def identity() -> "Affine2D":
        return Affine2D()

    @staticmethod
    def translation(tx: float, ty: float) -> "Affine2D":
        m = np.eye(3)
        m[0, 2] = tx
        m[1, 2] = ty
        return Affine2D(m)

    @staticmethod
    def scale(sx: float, sy: Optional[float] = None) -> "Affine2D":
        if sy is None:
            sy = sx
        m = np.eye(3)
        m[0, 0] = sx
        m[1, 1] = sy
        return Affine2D(m)

    @staticmethod
    def rotation_deg(angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> "Affine2D":
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        R = np.array(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0, 0, 1]], dtype=float)
        T1 = Affine2D.translation(-cx, -cy).m
        T2 = Affine2D.translation(cx, cy).m
        return Affine2D(T2 @ R @ T1)

    @staticmethod
    def matrix(a: float, b: float, c: float, d: float, e: float, f: float) -> "Affine2D":
        m = np.array([[a, c, e], [b, d, f], [0, 0, 1]],
                     dtype=float)  # SVG's (a b c d e f)
        return Affine2D(m)

    @staticmethod
    def from_svg_transform(transform_str: str) -> "Affine2D":
        """Parse a subset of SVG transform strings."""
        if not transform_str:
            return Affine2D.identity()

        tok_re = re.compile(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)")
        transform = Affine2D.identity()
        for name, args in tok_re.findall(transform_str):
            parts = [float(p) for p in re.split(r"[ ,]+", args.strip()) if p]
            if name == "matrix" and len(parts) == 6:
                T = Affine2D.matrix(*parts)
            elif name == "translate" and len(parts) in (1, 2):
                tx = parts[0]
                ty = parts[1] if len(parts) == 2 else 0.0
                T = Affine2D.translation(tx, ty)
            elif name == "scale" and len(parts) in (1, 2):
                sx = parts[0]
                sy = parts[1] if len(parts) == 2 else None
                T = Affine2D.scale(sx, sy)
            elif name == "rotate" and len(parts) in (1, 3):
                angle = parts[0]
                if len(parts) == 3:
                    T = Affine2D.rotation_deg(angle, parts[1], parts[2])
                else:
                    T = Affine2D.rotation_deg(angle)
            else:
                continue
            transform = transform @ T
        return transform


# -----------------------------
# SVG parsing and flattening
# -----------------------------

def _float_to_int(x: float, scale: int) -> int:
    # Round to nearest integer (ties to nearest even via Python round)
    return int(round(x * scale))


def _complex_to_tuple(z: complex) -> Tuple[float, float]:
    return (z.real, z.imag)


def _midpoint(p0: complex, p1: complex) -> complex:
    return (p0 + p1) / 2.0


def _point_line_dist(p: complex, a: complex, b: complex) -> float:
    """Distance from point p to line segment ab (using doubles)."""
    ax, ay = a.real, a.imag
    bx, by = b.real, b.imag
    px, py = p.real, p.imag
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        dx, dy = px - ax, py - ay
        return math.hypot(dx, dy)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def _flatten_segment(segment, tol: float, t0: float = 0.0, t1: float = 1.0, depth: int = 0, max_depth: int = 18) -> List[complex]:
    """Recursively approximate any segment with a polyline within tolerance.
    Returns a list of complex points from t0..t1 (including endpoints).
    """
    p0 = segment.point(t0)
    p2 = segment.point(t1)
    pm = segment.point(0.5 * (t0 + t1))

    # Error as distance of midpoint to chord
    err = _point_line_dist(pm, p0, p2)
    if err <= tol or depth >= max_depth:
        return [p0, p2]
    else:
        left = _flatten_segment(
            segment, tol, t0, 0.5 * (t0 + t1), depth + 1, max_depth)
        right = _flatten_segment(
            segment, tol, 0.5 * (t0 + t1), t1, depth + 1, max_depth)
        # Avoid duplicating the midpoint
        return left[:-1] + right


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
            pts = _flatten_segment(s, tol)
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


def _safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _circle_to_polyline(cx: float, cy: float, r: float, segments: int = 64) -> List[complex]:
    pts = []
    for i in range(segments + 1):
        th = 2 * math.pi * i / segments
        pts.append(complex(cx + r * math.cos(th), cy + r * math.sin(th)))
    return pts


def _ellipse_to_polyline(cx: float, cy: float, rx: float, ry: float, segments: int = 96) -> List[complex]:
    pts = []
    for i in range(segments + 1):
        th = 2 * math.pi * i / segments
        pts.append(complex(cx + rx * math.cos(th), cy + ry * math.sin(th)))
    return pts


def _rect_to_polyline(x: float, y: float, w: float, h: float) -> List[complex]:
    return [
        complex(x, y),
        complex(x + w, y),
        complex(x + w, y + h),
        complex(x, y + h),
        complex(x, y),
    ]


def _apply_transform_to_polyline(poly: List[complex], T: Affine2D) -> List[complex]:
    return [complex(*T.apply(p.real, p.imag)) for p in poly]


def _iter_svg_geometry(root: ET.Element, tol: float) -> Iterable[List[complex]]:
    """Yield polylines from supported SVG elements (already filtered by visibility)."""
    ns = ""
    # Handle namespaces (strip common "http://www.w3.org/2000/svg")
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    def walk(el: ET.Element, ctm: Affine2D):
        if not _extract_style_visibility(el):
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
                polylines = path_to_polylines(p, tol=tol)
                for poly in polylines:
                    yield _apply_transform_to_polyline(poly, T)

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
                    yield _apply_transform_to_polyline(pts, T)

        # ----- <line>
        elif tag == "line":
            x1 = _safe_float(el.attrib.get("x1", "0"))
            y1 = _safe_float(el.attrib.get("y1", "0"))
            x2 = _safe_float(el.attrib.get("x2", "0"))
            y2 = _safe_float(el.attrib.get("y2", "0"))
            pts = [complex(x1, y1), complex(x2, y2)]
            yield _apply_transform_to_polyline(pts, T)

        # ----- <rect> (no rounded corners in this minimal version)
        elif tag == "rect":
            x = _safe_float(el.attrib.get("x", "0"))
            y = _safe_float(el.attrib.get("y", "0"))
            w = _safe_float(el.attrib.get("width", "0"))
            h = _safe_float(el.attrib.get("height", "0"))
            if w > 0 and h > 0:
                pts = _rect_to_polyline(x, y, w, h)
                yield _apply_transform_to_polyline(pts, T)

        # ----- <circle>
        elif tag == "circle":
            cx = _safe_float(el.attrib.get("cx", "0"))
            cy = _safe_float(el.attrib.get("cy", "0"))
            r = _safe_float(el.attrib.get("r", "0"))
            if r > 0:
                pts = _circle_to_polyline(cx, cy, r)
                yield _apply_transform_to_polyline(pts, T)

        # ----- <ellipse>
        elif tag == "ellipse":
            cx = _safe_float(el.attrib.get("cx", "0"))
            cy = _safe_float(el.attrib.get("cy", "0"))
            rx = _safe_float(el.attrib.get("rx", "0"))
            ry = _safe_float(el.attrib.get("ry", "0"))
            if rx > 0 and ry > 0:
                pts = _ellipse_to_polyline(cx, cy, rx, ry)
                yield _apply_transform_to_polyline(pts, T)

        # Ignore others for now (images, text, etc.)

    # Kick off traversal
    for child in list(root):
        yield from walk(child, Affine2D.identity())

# -----------------------------
# Public API
# -----------------------------


@dataclass
class PolylineI:
    """Integer-scaled polyline."""
    pts: List[IntPoint]


@dataclass
class SVGGeometry:
    polylines: List[PolylineI]
    scale: int

    def bounds(self) -> Tuple[int, int, int, int]:
        """(minx, miny, maxx, maxy) in integer space"""
        xs = [p.x for pl in self.polylines for p in pl.pts]
        ys = [p.y for pl in self.polylines for p in pl.pts]
        return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else (0, 0, 0, 0)


def load_svg_as_integer_polylines(svg_path: str, scale: int = 10000, tol: float = 0.25) -> SVGGeometry:
    """Parse an SVG and return integer-scaled polylines suitable for integer geometry algorithms.

    Args:
        svg_path: Path to the SVG file.
        scale: Scaling factor applied to all coordinates before rounding.
        tol: Geometric tolerance in the same units as the SVG (before scaling).
             Smaller = more segments when flattening curves.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    polylines_float: List[List[complex]] = []
    for poly in _iter_svg_geometry(root, tol=tol):
        if len(poly) >= 2:
            polylines_float.append(poly)

    # Convert to integer space
    polylines_int: List[PolylineI] = []
    for poly in polylines_float:
        pts_i = [IntPoint(_float_to_int(p.real, scale),
                          # We need to flip Z on conversion:
                          #  * SVG defines the origin at the top-left, with +y downward.
                          #  * Matplotlib 3D (and most CAD/CAM coordinate systems) assume +y upward and +z upward.
                          -_float_to_int(p.imag, scale)) for p in poly]
        polylines_int.append(PolylineI(pts=pts_i))

    return SVGGeometry(polylines=polylines_int, scale=scale)

# -----------------------------
# Visualization
# -----------------------------


def _set_axes_equal_3d(ax):
    """Make 3D axes have equal scale so geometry isn't distorted."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_geometry_3d(geom: SVGGeometry, show_axes: bool = True):
    """Render integer polylines at z=0 in a 3D view (rotatable)."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each polyline. Convert back to floats (unscale) for display.
    inv = 1.0 / float(geom.scale if geom.scale != 0 else 1)
    for pl in geom.polylines:
        xs = [p.x * inv for p in pl.pts]
        ys = [p.y * inv for p in pl.pts]
        zs = [0.0 for _ in pl.pts]
        ax.plot(xs, ys, zs, linewidth=1.0)

    if show_axes:
        # Add simple XY axes in the plane
        minx, miny, maxx, maxy = geom.bounds()
        invx = [minx * inv, maxx * inv]
        ax.plot(invx, [0, 0], [0, 0])
        invy = [miny * inv, maxy * inv]
        ax.plot([0, 0], invy, [0, 0])

    ax.set_xlabel("X (SVG units)")
    ax.set_ylabel("Y (SVG units)")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=-60)
    _set_axes_equal_3d(ax)
    ax.grid(True)

    def on_key(event):
        if event.key == 'r':
            ax.view_init(elev=30, azim=-60)
            fig.canvas.draw_idle()
        elif event.key == 'g':
            ax.grid(not ax.xaxis._gridOnMajor)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Toolpath export
# -----------------------------


def export_toolpath_json(geom: SVGGeometry, path: str) -> None:
    """Export as JSON: { "scale": int, "polylines": [ [[x,y], ...], ... ] }"""
    obj = {
        "scale": geom.scale,
        "polylines": [[[p.x, p.y] for p in pl.pts] for pl in geom.polylines],
    }
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if path == "-" or path == "stdout":
        print(data)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)


def export_toolpath_txt(geom: SVGGeometry, path: str) -> None:
    """Export text: one polyline per line, 'x1,y1 x2,y2 ...' (all integers)."""
    lines = []
    for pl in geom.polylines:
        line = " ".join(f"{p.x},{p.y}" for p in pl.pts)
        lines.append(line)
    data = "\n".join(lines) + "\n"
    if path == "-" or path == "stdout":
        print(data, end="")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="CAM-oriented SVG -> integer polylines + 3D viewer")
    ap.add_argument("--input", dest="svg", required=True, help="Input SVG file")
    ap.add_argument("--scale", type=int, default=10000, help="Integer scaling factor (default: 10000)")
    ap.add_argument("--tol", type=float, default=0.1, help="Flattening tolerance before scaling (default: 0.25)")
    ap.add_argument("--no-view", action="store_true", help="Do not open the 3D viewer")
    ap.add_argument("--export-json", metavar="PATH", help="Write integer toolpaths to JSON (use '-' for stdout)")
    ap.add_argument("--export-txt", metavar="PATH", help="Write integer toolpaths to TXT (use '-' for stdout)")
    args = ap.parse_args(argv)

    geom = load_svg_as_integer_polylines(args.svg, scale=args.scale, tol=args.tol)
    minx, miny, maxx, maxy = geom.bounds()
    print(f"Loaded polylines: {len(geom.polylines)}")
    print(
        f"Bounds (int): min=({minx},{miny}) max=({maxx},{maxy})  scale={geom.scale}")
    total_pts = sum(len(pl.pts) for pl in geom.polylines)
    print(f"Total points: {total_pts}")

    # Exports
    if args.export_json:
        export_toolpath_json(geom, args.export_json)
    if args.export_txt:
        export_toolpath_txt(geom, args.export_txt)

    if not args.no_view:
        visualize_geometry_3d(geom)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
