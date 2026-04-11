#!/usr/bin/env python3
"""Apply a probed height map to an existing NC file."""

from __future__ import annotations

import argparse
from pathlib import Path

from gerber2nc_lib.height_map_compensation import (
    apply_height_map_to_gcode,
    build_adjusted_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process an existing NC file with a probed height-map CSV and "
            "write a separate compensated NC file."
        )
    )
    parser.add_argument("input_nc", help="Path to the source NC file.")
    parser.add_argument("height_map_csv", help="Path to the height-map CSV.")
    parser.add_argument(
        "output_nc",
        nargs="?",
        help=(
            "Optional output path. Defaults to the input filename with "
            "'.height-adjusted' inserted before the .nc suffix."
        ),
    )
    parser.add_argument(
        "--max-compensated-z",
        type=float,
        default=1.0,
        help=(
            "Only compensate moves whose starting or ending commanded Z is at "
            "or below this value in mm. Default: 1.0"
        ),
    )
    parser.add_argument(
        "--max-segment-length",
        type=float,
        default=1.0,
        help="Maximum XY segment length in mm used for surface-following output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_nc)
    output_path = (
        Path(args.output_nc)
        if args.output_nc
        else build_adjusted_output_path(input_path, ".height-adjusted")
    )

    apply_height_map_to_gcode(
        input_path,
        output_path,
        args.height_map_csv,
        max_compensated_z=args.max_compensated_z,
        max_segment_length=args.max_segment_length,
    )
    print(f"Wrote height-map-adjusted G-code: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
