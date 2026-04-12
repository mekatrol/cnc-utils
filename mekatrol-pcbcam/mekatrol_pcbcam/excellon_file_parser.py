from __future__ import annotations

from pathlib import Path
import re

from .board_bounds import BoardBounds
from .imported_drill_file import ImportedDrillFile


class ExcellonFileParser:
    def parse_file(self, path: str | Path) -> ImportedDrillFile:
        file_path = Path(path).resolve()
        tool_diameters: dict[str, float] = {}
        holes: list[tuple[float, float, float]] = []
        units_mult = 1.0
        current_tool: str | None = None
        bounds = BoardBounds()

        with file_path.open("r", encoding="utf-8") as drill_file:
            for raw_line in drill_file:
                line = raw_line.strip()
                if not line or line.startswith(";"):
                    continue
                upper = line.upper()
                if "METRIC" in upper:
                    units_mult = 1.0
                elif "INCH" in upper:
                    units_mult = 25.4

                match_tool = re.match(r"^T(\d+)C([\d\.]+)", line)
                if match_tool:
                    tool_diameters[match_tool.group(1)] = (
                        float(match_tool.group(2)) * units_mult
                    )
                    continue

                match_tool_change = re.match(r"^T(\d+)$", line)
                if match_tool_change:
                    current_tool = match_tool_change.group(1)
                    continue

                match_coord = re.match(r"^X(-?\d+(\.\d+)?)Y(-?\d+(\.\d+)?)", line)
                if not match_coord or not current_tool:
                    continue

                x = float(match_coord.group(1)) * units_mult
                y = float(match_coord.group(3)) * units_mult
                diameter = tool_diameters[current_tool]
                holes.append((x, y, diameter))
                bounds.include_point(x, y, diameter * 0.5)

        return ImportedDrillFile(
            path=file_path,
            display_name=file_path.name,
            holes=holes,
            bounds=bounds,
        )
