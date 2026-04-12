from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import math
import re
from typing import Iterable, List, Sequence

AXES = ("X", "Y", "Z")
TOKEN_RE = re.compile(r"([A-Z])\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Segment3D:
    start: Point3D
    end: Point3D
    rapid: bool
    line_number: int
    source: str


@dataclass(frozen=True)
class ToolpathStats:
    min_point: Point3D
    max_point: Point3D
    segment_count: int
    rapid_count: int
    cut_count: int
    path_length: float


@dataclass(frozen=True)
class ToolpathDocument:
    path: Path
    segments: Sequence[Segment3D]
    stats: ToolpathStats


class GCodeParser:
    """Minimal linear-motion G-code parser for viewer bootstrapping."""

    def parse_file(self, path: str | Path) -> ToolpathDocument:
        file_path = Path(path)
        logger.debug("Parsing G-code file: %s", file_path)
        segments: List[Segment3D] = []
        current = Point3D(0.0, 0.0, 0.0)
        absolute_mode = True
        motion_mode = "G0"

        min_x = max_x = current.x
        min_y = max_y = current.y
        min_z = max_z = current.z
        total_length = 0.0
        rapid_count = 0
        cut_count = 0

        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            clean = self._strip_comments(raw_line).upper().strip()
            if not clean:
                continue

            words = {axis: value for axis, value in TOKEN_RE.findall(clean)}
            command = self._find_motion_command(clean)
            if command == "G90":
                absolute_mode = True
            elif command == "G91":
                absolute_mode = False
            elif command in {"G0", "G00", "G1", "G01"}:
                motion_mode = "G0" if command in {"G0", "G00"} else "G1"

            target = current
            if any(axis in words for axis in AXES):
                target = self._resolve_target(current, words, absolute_mode)
                if target != current:
                    segment = Segment3D(
                        start=current,
                        end=target,
                        rapid=(motion_mode == "G0"),
                        line_number=line_number,
                        source=raw_line.rstrip(),
                    )
                    segments.append(segment)
                    step = math.dist(
                        (current.x, current.y, current.z),
                        (target.x, target.y, target.z),
                    )
                    total_length += step
                    if segment.rapid:
                        rapid_count += 1
                    else:
                        cut_count += 1
                    for point in (segment.start, segment.end):
                        min_x = min(min_x, point.x)
                        min_y = min(min_y, point.y)
                        min_z = min(min_z, point.z)
                        max_x = max(max_x, point.x)
                        max_y = max(max_y, point.y)
                        max_z = max(max_z, point.z)
                current = target

        stats = ToolpathStats(
            min_point=Point3D(min_x, min_y, min_z),
            max_point=Point3D(max_x, max_y, max_z),
            segment_count=len(segments),
            rapid_count=rapid_count,
            cut_count=cut_count,
            path_length=total_length,
        )
        logger.debug(
            "Parsed G-code file: %s segments=%d cut=%d rapid=%d path_length=%.3f",
            file_path,
            stats.segment_count,
            stats.cut_count,
            stats.rapid_count,
            stats.path_length,
        )
        return ToolpathDocument(path=file_path, segments=segments, stats=stats)

    def _strip_comments(self, line: str) -> str:
        no_paren = re.sub(r"\([^)]*\)", "", line)
        return no_paren.split(";", maxsplit=1)[0]

    def _find_motion_command(self, line: str) -> str | None:
        for axis, value in TOKEN_RE.findall(line):
            if axis != "G":
                continue
            if value in {"0", "00", "1", "01", "90", "91"}:
                return f"G{value}"
        return None

    def _resolve_target(self, current: Point3D, words: dict[str, str], absolute_mode: bool) -> Point3D:
        next_values = {axis: getattr(current, axis.lower()) for axis in AXES}
        for axis in AXES:
            if axis not in words:
                continue
            value = float(words[axis])
            if absolute_mode:
                next_values[axis] = value
            else:
                next_values[axis] += value
        return Point3D(next_values["X"], next_values["Y"], next_values["Z"])
