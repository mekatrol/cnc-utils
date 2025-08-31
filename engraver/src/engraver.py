
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - ensures 3D support is loaded
import matplotlib.pyplot as plt

from geometry.GeometryInt import GeometryInt
from svg.SvgConverter import SvgConverter


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


def visualize_geometry_3d(geom: GeometryInt, show_axes: bool = True):
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


def export_toolpath_json(geom: GeometryInt, path: str) -> None:
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


def export_toolpath_txt(geom: GeometryInt, path: str) -> None:
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

    geom = SvgConverter.convert(args.svg, scale=args.scale, tol=args.tol)
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
