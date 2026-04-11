"""Shared board extents discovered while parsing the manufacturing files.

Gerber and Excellon data are imported from separate files, but later steps need
one common bounding box so they can:
- size the preview window,
- shift all geometry into one local origin,
- and derive a sensible end-of-program parking position.
"""

from dataclasses import dataclass


@dataclass
class BoardContext:
    base_name: str
    x_min: float = 1000000.0
    x_max: float = -1000000.0
    y_min: float = 1000000.0
    y_max: float = -1000000.0

    def update_bounds(self, x: float, y: float, margin: float) -> None:
        # Some features occupy more area than a single coordinate implies. For
        # example, the center of a pad is not enough to describe its full
        # footprint. Callers provide a margin so the bounds safely contain the
        # actual copper or mechanical feature around that point.
        if x - margin < self.x_min:
            self.x_min = x - margin
        if x + margin > self.x_max:
            self.x_max = x + margin
        if y - margin < self.y_min:
            self.y_min = y - margin
        if y + margin > self.y_max:
            self.y_max = y + margin

    def update_point(self, x: float, y: float) -> None:
        # Use this when the coordinate itself already represents the feature
        # extent of interest, such as drill centers in this simplified workflow.
        if x < self.x_min:
            self.x_min = x
        if x > self.x_max:
            self.x_max = x
        if y < self.y_min:
            self.y_min = y
        if y > self.y_max:
            self.y_max = y
