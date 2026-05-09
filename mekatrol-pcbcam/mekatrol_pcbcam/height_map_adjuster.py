from __future__ import annotations

import csv
import math
import re
from pathlib import Path

TOKEN_RE = re.compile(r"([A-Z])\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
HeightMap = tuple[
    list[float], list[float], dict[tuple[float, float], float], float, float, float
]


def write_flat_height_map_csv(
    path: Path,
    *,
    start_x: float,
    start_y: float,
    width: float,
    height: float,
    step_distance_x: float,
    step_distance_y: float,
) -> None:
    xs = _build_axis(start_x, width, step_distance_x)
    ys = _build_axis(start_y, height, step_distance_y)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ix", "iy", "x_mm", "y_mm", "z_mm"])
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                writer.writerow([ix, iy, f"{x:.6f}", f"{y:.6f}", "0.000000"])


def adjust_nc_file(source_path: Path, output_path: Path, csv_path: Path) -> None:
    points = _read_points(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_x = 0.0
    current_y = 0.0
    current_z = 0.0
    absolute_mode = True
    motion_mode = "G0"
    output_lines = ["(height-map adjusted from " + source_path.name + ")"]

    source_lines = source_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    for raw_line in source_lines:
        clean = _strip_comments(raw_line).upper().strip()
        if not clean:
            output_lines.append(raw_line)
            continue
        words = {axis: value for axis, value in TOKEN_RE.findall(clean)}
        command = _find_motion_command(clean)
        if command == "G90":
            absolute_mode = True
            output_lines.append(raw_line)
            continue
        if command == "G91":
            absolute_mode = False
            output_lines.append(raw_line)
            continue
        if command in {"G0", "G00", "G1", "G01"}:
            motion_mode = "G0" if command in {"G0", "G00"} else "G1"

        if not any(axis in words for axis in ("X", "Y", "Z")):
            output_lines.append(raw_line)
            continue

        target_x = _target_value(current_x, words.get("X"), absolute_mode)
        target_y = _target_value(current_y, words.get("Y"), absolute_mode)
        target_z = _target_value(current_z, words.get("Z"), absolute_mode)
        if motion_mode == "G1" and (target_x != current_x or target_y != current_y):
            xy_distance = math.dist((current_x, current_y), (target_x, target_y))
            line_steps = max(1, math.ceil(xy_distance / points[5]))
            for step in range(1, line_steps + 1):
                ratio = step / line_steps
                x = current_x + ((target_x - current_x) * ratio)
                y = current_y + ((target_y - current_y) * ratio)
                z = current_z + ((target_z - current_z) * ratio)
                adjusted_z = z + _height_at(points, x, y)
                feed = f" F{words['F']}" if step == 1 and "F" in words else ""
                output_lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{adjusted_z:.3f}{feed}")
        else:
            adjusted_z = target_z + _height_at(points, target_x, target_y)
            command_text = motion_mode
            feed = f" F{words['F']}" if "F" in words else ""
            output_lines.append(
                f"{command_text} X{target_x:.3f} Y{target_y:.3f} "
                f"Z{adjusted_z:.3f}{feed}"
            )
        current_x = target_x
        current_y = target_y
        current_z = target_z

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def _build_axis(start: float, length: float, step_distance: float) -> list[float]:
    if step_distance <= 0.0:
        raise ValueError("Step distance must be greater than 0.")
    if length < 0.0:
        raise ValueError(
            "Height map width and height must be greater than or equal to 0."
        )
    end = start + length
    values = []
    value = start
    while value < end:
        values.append(value)
        value += step_distance
    if not values or abs(values[-1] - end) > 1e-6:
        values.append(end)
    return values


def _read_points(path: Path) -> HeightMap:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            rows.append((float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])))
    if not rows:
        raise ValueError("Height map CSV has no sample points.")
    xs = sorted({row[0] for row in rows})
    ys = sorted({row[1] for row in rows})
    points = {(x, y): z for x, y, z in rows}
    step_x = min((b - a for a, b in zip(xs, xs[1:])), default=1.0)
    step_y = min((b - a for a, b in zip(ys, ys[1:])), default=1.0)
    return xs, ys, points, min(step_x, step_y), step_x, step_y


def _height_at(height_map: HeightMap, x: float, y: float) -> float:
    xs, ys, points, _, _, _ = height_map
    x0, x1 = _bracket(xs, x)
    y0, y1 = _bracket(ys, y)
    z00 = points.get((x0, y0), 0.0)
    z10 = points.get((x1, y0), z00)
    z01 = points.get((x0, y1), z00)
    z11 = points.get((x1, y1), z10)
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    z0 = z00 + ((z10 - z00) * tx)
    z1 = z01 + ((z11 - z01) * tx)
    return z0 + ((z1 - z0) * ty)


def _bracket(values: list[float], value: float) -> tuple[float, float]:
    if value <= values[0]:
        return values[0], values[0]
    if value >= values[-1]:
        return values[-1], values[-1]
    for index, current in enumerate(values[:-1]):
        next_value = values[index + 1]
        if current <= value <= next_value:
            return current, next_value
    return values[-1], values[-1]


def _strip_comments(line: str) -> str:
    no_paren = re.sub(r"\([^)]*\)", "", line)
    return no_paren.split(";", maxsplit=1)[0]


def _find_motion_command(line: str) -> str | None:
    for axis, value in TOKEN_RE.findall(line):
        if axis != "G":
            continue
        if value in {"0", "00", "1", "01", "90", "91"}:
            return f"G{value}"
    return None


def _target_value(current: float, raw_value: str | None, absolute_mode: bool) -> float:
    if raw_value is None:
        return current
    value = float(raw_value)
    if absolute_mode:
        return value
    return current + value
