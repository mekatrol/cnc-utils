#!/usr/bin/env python3
"""
viewer.py - 3D heightmap viewer for z_height_map.py output.

Reads the CSV produced by z_height_map.py:
  ix,iy,x_mm,y_mm,z_mm

Displays an interactive 3D surface (or scatter) using matplotlib.

Usage:
  python3 viewer.py heightmap.csv
  python3 viewer.py heightmap.csv --invert-z
  python3 viewer.py heightmap.csv --mode scatter
  python3 viewer.py heightmap.csv --stride 2 --mode surface
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource


@dataclass(frozen=True)
class Point:
    ix: int
    iy: int
    x: float
    y: float
    z: float


def load_points(csv_path: Path) -> List[Point]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"ix", "iy", "x_mm", "y_mm", "z_mm"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must contain header columns: {sorted(required)}. "
                f"Got: {reader.fieldnames!r}"
            )

        pts: List[Point] = []
        for row in reader:
            pts.append(
                Point(
                    ix=int(row["ix"]),
                    iy=int(row["iy"]),
                    x=float(row["x_mm"]),
                    y=float(row["y_mm"]),
                    z=float(row["z_mm"]),
                )
            )
        if not pts:
            raise ValueError("CSV contains no points.")
        return pts


def build_grid(points: List[Point]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Map (iy, ix) -> (x, y, z)
    by_idx: Dict[Tuple[int, int], Point] = {(p.iy, p.ix): p for p in points}

    max_ix = max(p.ix for p in points)
    max_iy = max(p.iy for p in points)

    # Create grids sized by index range
    X = np.full((max_iy + 1, max_ix + 1), np.nan, dtype=float)
    Y = np.full((max_iy + 1, max_ix + 1), np.nan, dtype=float)
    Z = np.full((max_iy + 1, max_ix + 1), np.nan, dtype=float)

    missing = 0
    for iy in range(max_iy + 1):
        for ix in range(max_ix + 1):
            p = by_idx.get((iy, ix))
            if p is None:
                missing += 1
                continue
            X[iy, ix] = p.x
            Y[iy, ix] = p.y
            Z[iy, ix] = p.z

    if missing:
        # Surface plotting can still work if sparse, but it will look broken.
        # Users can switch to scatter mode in that case.
        pass

    # Sanity: if X/Y are NaN-heavy, bail early
    if np.all(np.isnan(Z)):
        raise ValueError("All Z values are NaN after gridding (bad CSV?).")

    return X, Y, Z


def nanstats(z: np.ndarray) -> Tuple[float, float, float]:
    zf = z[np.isfinite(z)]
    if zf.size == 0:
        return (math.nan, math.nan, math.nan)
    return (float(np.min(zf)), float(np.mean(zf)), float(np.max(zf)))


def main() -> int:
    ap = argparse.ArgumentParser(description="3D viewer for GRBL heightmap CSV.")
    ap.add_argument("csv", help="Path to heightmap CSV (from z_height_map.py)")
    ap.add_argument(
        "--mode",
        choices=["surface", "wireframe", "scatter"],
        default="surface",
        help="Render mode (default: surface)",
    )
    ap.add_argument(
        "--invert-z",
        action="store_true",
        help="Invert Z (useful if your 'down' is negative and you want height positive).",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Plot every Nth sample in each axis for speed (default: 1).",
    )
    ap.add_argument(
        "--z-scale",
        type=float,
        default=100.0,
        help="Scale Z by this factor for visualization (default: 1.0).",
    )
    ap.add_argument(
        "--shade",
        action="store_true",
        help="Apply simple lighting-based shading on surface (matplotlib LightSource).",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    points = load_points(csv_path)
    X, Y, Z = build_grid(points)

    if args.invert_z:
        Z = -Z

    if args.z_scale != 1.0:
        Z = Z * float(args.z_scale)

    stride = max(1, int(args.stride))
    Xp = X[::stride, ::stride]
    Yp = Y[::stride, ::stride]
    Zp = Z[::stride, ::stride]

    zmin, zavg, zmax = nanstats(Zp)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    title = f"{csv_path.name}  Z(min/avg/max)={zmin:.4f}/{zavg:.4f}/{zmax:.4f} mm"
    if args.invert_z:
        title += "  (inverted)"
    if args.z_scale != 1.0:
        title += f"  (z-scale={args.z_scale:g})"
    if stride != 1:
        title += f"  (stride={stride})"

    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # Flatten for scatter or for safe finite mask
    finite_mask = np.isfinite(Xp) & np.isfinite(Yp) & np.isfinite(Zp)

    if args.mode == "scatter":
        ax.scatter(
            Xp[finite_mask],
            Yp[finite_mask],
            Zp[finite_mask],
            s=8,
            depthshade=True,
        )
    else:
        # For surface/wireframe, we need 2D grids; NaNs are ok but will create gaps.
        if args.mode == "wireframe":
            ax.plot_wireframe(Xp, Yp, Zp, rstride=1, cstride=1, linewidth=0.6)
        else:
            if args.shade:
                # LightSource expects Z as height; X/Y spacing influences shading.
                # Use a conservative default and let matplotlib handle the rest.
                ls = LightSource(azdeg=315, altdeg=45)
                rgb = ls.shade(
                    Zp, cmap=plt.cm.viridis, vert_exag=1.0, blend_mode="soft"
                )
                surf = ax.plot_surface(
                    Xp,
                    Yp,
                    Zp,
                    facecolors=rgb,
                    linewidth=0,
                    antialiased=True,
                )
                # Add a separate colorbar based on Z values (approximate)
                mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
                mappable.set_array(Zp[np.isfinite(Zp)])
                fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, label="Z (mm)")
            else:
                surf = ax.plot_surface(
                    Xp,
                    Yp,
                    Zp,
                    cmap="viridis",
                    linewidth=0,
                    antialiased=True,
                )
                fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08, label="Z (mm)")

    # Make axes aspect less misleading (matplotlib 3D has limitations).
    # Use data ranges to set box aspect when available (matplotlib >= 3.3).
    try:
        xr = np.nanmax(Xp) - np.nanmin(Xp)
        yr = np.nanmax(Yp) - np.nanmin(Yp)
        zr = np.nanmax(Zp) - np.nanmin(Zp)
        if np.isfinite(xr) and np.isfinite(yr) and np.isfinite(zr):
            ax.set_box_aspect((max(xr, 1e-9), max(yr, 1e-9), max(zr, 1e-9)))
    except Exception:
        pass

    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
