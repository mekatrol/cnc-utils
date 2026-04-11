"""Apply a probed Z height map to existing G-code motion."""

from __future__ import annotations

import csv
import math
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path


WORD_RE = re.compile(r"([A-Z])([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)


@dataclass(frozen=True)
class HeightPoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class CompensationConfig:
    enabled: bool
    map_path: str
    output_suffix: str
    max_compensated_z: float
    max_segment_length: float


class HeightMap:
    def __init__(self, x_values: list[float], y_values: list[float], z_grid: list[list[float]]):
        if len(x_values) < 2 or len(y_values) < 2:
            raise ValueError("Height map must contain at least a 2x2 grid of samples.")
        self.x_values = x_values
        self.y_values = y_values
        self.z_grid = z_grid

    @classmethod
    def from_csv(cls, path: str | Path) -> "HeightMap":
        csv_path = Path(path)
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required = {"x_mm", "y_mm", "z_mm"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Height-map CSV must contain columns {sorted(required)}. "
                    f"Got {reader.fieldnames!r}."
                )

            points: list[HeightPoint] = []
            for row in reader:
                points.append(
                    HeightPoint(
                        x=float(row["x_mm"]),
                        y=float(row["y_mm"]),
                        z=float(row["z_mm"]),
                    )
                )

        if not points:
            raise ValueError(f"Height-map CSV '{csv_path}' contains no points.")

        x_values = sorted({point.x for point in points})
        y_values = sorted({point.y for point in points})
        x_index = {value: index for index, value in enumerate(x_values)}
        y_index = {value: index for index, value in enumerate(y_values)}
        z_grid: list[list[float | None]] = [
            [None for _ in x_values] for _ in y_values
        ]

        for point in points:
            z_grid[y_index[point.y]][x_index[point.x]] = point.z

        missing = [
            (x_values[xi], y_values[yi])
            for yi, row in enumerate(z_grid)
            for xi, value in enumerate(row)
            if value is None
        ]
        if missing:
            missing_summary = ", ".join(
                f"({x:.3f}, {y:.3f})" for x, y in missing[:6]
            )
            if len(missing) > 6:
                missing_summary += ", ..."
            raise ValueError(
                "Height-map CSV is missing grid points required for interpolation: "
                f"{missing_summary}"
            )

        return cls(
            x_values=x_values,
            y_values=y_values,
            z_grid=[[float(value) for value in row] for row in z_grid],
        )

    def offset_at(self, x: float, y: float) -> float:
        clamped_x = min(max(x, self.x_values[0]), self.x_values[-1])
        clamped_y = min(max(y, self.y_values[0]), self.y_values[-1])

        x0_index, x1_index, tx = self._axis_bracket(self.x_values, clamped_x)
        y0_index, y1_index, ty = self._axis_bracket(self.y_values, clamped_y)

        z00 = self.z_grid[y0_index][x0_index]
        z10 = self.z_grid[y0_index][x1_index]
        z01 = self.z_grid[y1_index][x0_index]
        z11 = self.z_grid[y1_index][x1_index]

        z0 = z00 + (z10 - z00) * tx
        z1 = z01 + (z11 - z01) * tx
        return z0 + (z1 - z0) * ty

    @staticmethod
    def _axis_bracket(axis_values: list[float], value: float) -> tuple[int, int, float]:
        if value <= axis_values[0]:
            return 0, 1, 0.0
        if value >= axis_values[-1]:
            upper = len(axis_values) - 1
            return upper - 1, upper, 1.0

        upper = bisect_right(axis_values, value)
        lower = upper - 1
        lower_value = axis_values[lower]
        upper_value = axis_values[upper]
        fraction = (value - lower_value) / (upper_value - lower_value)
        return lower, upper, fraction


def parse_compensation_config(config: dict) -> CompensationConfig:
    height_map_config = config.get("height_map", {})
    if not isinstance(height_map_config, dict):
        raise ValueError("height_map config must be a YAML mapping.")

    return CompensationConfig(
        enabled=bool(height_map_config.get("enabled", False)),
        map_path=str(height_map_config.get("input", "")),
        output_suffix=str(height_map_config.get("output_suffix", ".height-adjusted")),
        max_compensated_z=float(height_map_config.get("max_compensated_z", 1.0)),
        max_segment_length=float(height_map_config.get("max_segment_length", 1.0)),
    )


def validate_compensation_config(config: dict, config_path: Path) -> None:
    height_map_config = config.get("height_map")
    if height_map_config is None:
        return
    if not isinstance(height_map_config, dict):
        raise ValueError(f"'{config_path}': height_map must be a YAML mapping.")

    enabled = height_map_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"'{config_path}': height_map.enabled must be true or false.")

    map_path = height_map_config.get("input", "")
    if not isinstance(map_path, str):
        raise ValueError(f"'{config_path}': height_map.input must be a string path.")

    output_suffix = height_map_config.get("output_suffix", ".height-adjusted")
    if not isinstance(output_suffix, str) or not output_suffix:
        raise ValueError(
            f"'{config_path}': height_map.output_suffix must be a non-empty string."
        )

    _validate_positive_number(
        height_map_config.get("max_segment_length", 1.0),
        "height_map.max_segment_length",
        config_path,
    )

    max_compensated_z = height_map_config.get("max_compensated_z", 1.0)
    if not isinstance(max_compensated_z, (int, float)) or isinstance(
        max_compensated_z, bool
    ):
        raise ValueError(
            f"'{config_path}': height_map.max_compensated_z must be a number."
        )


def apply_height_map_to_gcode(
    input_path: str | Path,
    output_path: str | Path,
    height_map_path: str | Path,
    *,
    max_compensated_z: float,
    max_segment_length: float,
) -> None:
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be greater than 0.")

    source_path = Path(input_path)
    destination_path = Path(output_path)
    height_map = HeightMap.from_csv(height_map_path)

    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    output_lines = [
        f"(height map applied from {Path(height_map_path).name})",
        f"(max compensated Z {max_compensated_z:.3f} mm, max segment length {max_segment_length:.3f} mm)",
    ]

    current_motion: str | None = None
    current_x = 0.0
    current_y = 0.0
    current_z = 0.0
    have_position = False

    for raw_line in source_lines:
        motion, axis_values, feed_value, comment = _parse_motion_line(raw_line)

        if motion is not None:
            current_motion = motion
        active_motion = motion or current_motion

        if active_motion not in {"G0", "G1"} or not axis_values:
            output_lines.append(raw_line)
            if axis_values:
                current_x = axis_values.get("X", current_x)
                current_y = axis_values.get("Y", current_y)
                current_z = axis_values.get("Z", current_z)
                have_position = True
            continue

        if not have_position:
            have_position = True

        start_x = current_x
        start_y = current_y
        start_z = current_z
        target_x = axis_values.get("X", current_x)
        target_y = axis_values.get("Y", current_y)
        target_z = axis_values.get("Z", current_z)

        has_xy_motion = abs(target_x - start_x) > 1e-9 or abs(target_y - start_y) > 1e-9
        has_explicit_z = "Z" in axis_values
        compensate_move = (
            (has_explicit_z and target_z <= max_compensated_z)
            or (has_xy_motion and start_z <= max_compensated_z)
        )
        if not compensate_move:
            output_lines.append(raw_line)
        else:
            adjusted_lines = _build_adjusted_motion_lines(
                motion=active_motion,
                start_x=start_x,
                start_y=start_y,
                start_z=start_z,
                target_x=target_x,
                target_y=target_y,
                target_z=target_z,
                feed_value=feed_value,
                comment=comment,
                height_map=height_map,
                max_segment_length=max_segment_length,
            )
            output_lines.extend(adjusted_lines)

        current_x = target_x
        current_y = target_y
        current_z = target_z

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def build_adjusted_output_path(input_path: str | Path, output_suffix: str) -> Path:
    source_path = Path(input_path)
    return source_path.with_name(f"{source_path.stem}{output_suffix}{source_path.suffix}")


def _build_adjusted_motion_lines(
    *,
    motion: str,
    start_x: float,
    start_y: float,
    start_z: float,
    target_x: float,
    target_y: float,
    target_z: float,
    feed_value: float | None,
    comment: str | None,
    height_map: HeightMap,
    max_segment_length: float,
) -> list[str]:
    distance_xy = math.hypot(target_x - start_x, target_y - start_y)

    if distance_xy <= 1e-9:
        adjusted_z = target_z + height_map.offset_at(target_x, target_y)
        line = _format_motion_line(
            motion,
            x=None,
            y=None,
            z=adjusted_z,
            feed_value=feed_value,
            comment=comment,
        )
        return [line]

    segment_count = max(1, math.ceil(distance_xy / max_segment_length))
    adjusted_lines: list[str] = []

    for segment_index in range(1, segment_count + 1):
        t = segment_index / segment_count
        x_pos = start_x + (target_x - start_x) * t
        y_pos = start_y + (target_y - start_y) * t
        z_pos = start_z + (target_z - start_z) * t
        adjusted_z = z_pos + height_map.offset_at(x_pos, y_pos)
        adjusted_lines.append(
            _format_motion_line(
                motion,
                x=x_pos,
                y=y_pos,
                z=adjusted_z,
                feed_value=feed_value if segment_index == 1 else None,
                comment=comment if segment_index == 1 else None,
            )
        )

    return adjusted_lines


def _format_motion_line(
    motion: str,
    *,
    x: float | None,
    y: float | None,
    z: float | None,
    feed_value: float | None,
    comment: str | None,
) -> str:
    parts = [motion]
    if x is not None:
        parts.append(f"X{x:.3f}")
    if y is not None:
        parts.append(f"Y{y:.3f}")
    if z is not None:
        parts.append(f"Z{z:.3f}")
    if feed_value is not None:
        parts.append(f"F{feed_value:g}")
    line = " ".join(parts)
    if comment:
        return f"{line} {comment}"
    return line


def _parse_motion_line(
    line: str,
) -> tuple[str | None, dict[str, float], float | None, str | None]:
    command_text, comment = _split_comment(line)
    matches = WORD_RE.findall(command_text.upper())

    motion: str | None = None
    axis_values: dict[str, float] = {}
    feed_value: float | None = None

    for word, value in matches:
        parsed_value = float(value)
        if word == "G":
            if math.isclose(parsed_value, 0.0):
                motion = "G0"
            elif math.isclose(parsed_value, 1.0):
                motion = "G1"
        elif word in {"X", "Y", "Z"}:
            axis_values[word] = parsed_value
        elif word == "F":
            feed_value = parsed_value

    return motion, axis_values, feed_value, comment


def _split_comment(line: str) -> tuple[str, str | None]:
    stripped = line.rstrip()
    if ";" in stripped:
        command_text, comment_text = stripped.split(";", 1)
        return command_text.rstrip(), f";{comment_text}"
    return stripped, None


def _validate_positive_number(
    value: object, setting_name: str, config_path: Path
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"'{config_path}': {setting_name} must be a number.")
    if value <= 0:
        raise ValueError(f"'{config_path}': {setting_name} must be greater than 0.")
