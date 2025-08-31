from __future__ import annotations

import argparse
from typing import Optional, Sequence

from export.JsonExporter import JsonExporter
from svg.SvgConverter import SvgConverter
from view.Viewer3d import Viewer3d


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="CAM-oriented SVG -> integer polylines + 3D viewer")
    ap.add_argument("--input", dest="svg", required=True, help="Input SVG file")
    ap.add_argument("--scale", type=int, default=10000, help="Integer scaling factor (default: 10000)")
    ap.add_argument("--tol", type=float, default=0.1, help="Flattening tolerance before scaling (default: 0.25)")
    ap.add_argument("--no-view", action="store_true", help="Do not open the 3D viewer")
    ap.add_argument("--export-json", metavar="PATH", help="Write integer toolpaths to JSON (use '-' for stdout)")
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
        JsonExporter.export(geom, args.export_json)

    if not args.no_view:
        Viewer3d.visualize(geom)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
