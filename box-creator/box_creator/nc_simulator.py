from __future__ import annotations

import re
from dataclasses import dataclass


WORD_PATTERN = re.compile(r"([A-Z])\s*([-+]?\d+(?:\.\d*)?|\.\d+)")
STOCK_COMMENT_PATTERN = re.compile(r"\bstock\s+sheet\s+(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ToolPosition:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class MotionSegment:
    start: ToolPosition
    end: ToolPosition
    command: str
    feed_rate: float
    spindle_speed: int
    duration_seconds: float
    stock_index: int

    @property
    def is_cutting_move(self) -> bool:
        return self.command == "G1" and min(self.start.z, self.end.z) <= 0.0


@dataclass(frozen=True)
class SimulatorProgram:
    segments: list[MotionSegment]
    total_seconds: float


def parse_nc_program(
    text: str, stock_origins: dict[int, tuple[float, float]], rapid_rate: float = 3000.0
) -> SimulatorProgram:
    x_pos = 0.0
    y_pos = 0.0
    z_pos = 0.0
    feed_rate = 1.0
    spindle_speed = 0
    active_motion = "G0"
    active_stock_index = 0
    absolute_positioning = True
    metric_units = True
    segments: list[MotionSegment] = []

    for raw_line in text.splitlines():
        comment_text = " ".join(re.findall(r"\((.*?)\)", raw_line))
        stock_match = STOCK_COMMENT_PATTERN.search(comment_text)
        if stock_match:
            active_stock_index = max(0, int(stock_match.group(1)) - 1)

        line = re.sub(r"\(.*?\)", "", raw_line).split(";", 1)[0].strip().upper()
        if not line or line in {"%", "/"}:
            continue

        words = [(letter, float(value)) for letter, value in WORD_PATTERN.findall(line)]
        if not words:
            continue

        saw_axis_word = False
        next_x = x_pos
        next_y = y_pos
        next_z = z_pos
        next_motion = active_motion

        for letter, value in words:
            if letter == "G":
                code = int(value)
                if code in {0, 1}:
                    next_motion = f"G{code}"
                elif code == 20:
                    metric_units = False
                elif code == 21:
                    metric_units = True
                elif code == 90:
                    absolute_positioning = True
                elif code == 91:
                    absolute_positioning = False
            elif letter == "F":
                feed_rate = max(1.0, _to_millimetres(value, metric_units))
            elif letter == "S":
                spindle_speed = max(0, int(value))
            elif letter in {"X", "Y", "Z"}:
                saw_axis_word = True
                value = _to_millimetres(value, metric_units)
                if letter == "X":
                    next_x = value if absolute_positioning else x_pos + value
                elif letter == "Y":
                    next_y = value if absolute_positioning else y_pos + value
                else:
                    next_z = value if absolute_positioning else z_pos + value

        active_motion = next_motion
        if not saw_axis_word:
            continue

        start = _with_stock_origin(
            ToolPosition(x_pos, y_pos, z_pos), active_stock_index, stock_origins
        )
        end = _with_stock_origin(
            ToolPosition(next_x, next_y, next_z), active_stock_index, stock_origins
        )
        distance = _distance(start, end)
        x_pos = next_x
        y_pos = next_y
        z_pos = next_z
        if distance <= 0.0001:
            continue

        move_rate = rapid_rate if active_motion == "G0" else feed_rate
        duration_seconds = distance / max(move_rate, 1.0) * 60.0
        segments.append(
            MotionSegment(
                start=start,
                end=end,
                command=active_motion,
                feed_rate=move_rate,
                spindle_speed=spindle_speed,
                duration_seconds=duration_seconds,
                stock_index=active_stock_index,
            )
        )

    total_seconds = sum(segment.duration_seconds for segment in segments)
    return SimulatorProgram(segments=segments, total_seconds=total_seconds)


def position_at_elapsed(
    segments: list[MotionSegment], elapsed_seconds: float
) -> tuple[int, float, ToolPosition | None]:
    if not segments:
        return 0, 0.0, None
    remaining = max(0.0, elapsed_seconds)
    for index, segment in enumerate(segments):
        if remaining <= segment.duration_seconds:
            ratio = (
                remaining / segment.duration_seconds
                if segment.duration_seconds > 0.0
                else 1.0
            )
            return index, ratio, _interpolate(segment.start, segment.end, ratio)
        remaining -= segment.duration_seconds
    last = segments[-1]
    return len(segments) - 1, 1.0, last.end


def _with_stock_origin(
    position: ToolPosition,
    stock_index: int,
    stock_origins: dict[int, tuple[float, float]],
) -> ToolPosition:
    origin_x, origin_y = stock_origins.get(stock_index, (0.0, 0.0))
    return ToolPosition(position.x + origin_x, position.y + origin_y, position.z)


def _to_millimetres(value: float, metric_units: bool) -> float:
    return value if metric_units else value * 25.4


def _distance(start: ToolPosition, end: ToolPosition) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    dz = end.z - start.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _interpolate(start: ToolPosition, end: ToolPosition, ratio: float) -> ToolPosition:
    clamped = max(0.0, min(1.0, ratio))
    return ToolPosition(
        x=start.x + (end.x - start.x) * clamped,
        y=start.y + (end.y - start.y) * clamped,
        z=start.z + (end.z - start.z) * clamped,
    )
