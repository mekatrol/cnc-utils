import re

from gerber2nc_lib.board_context import BoardContext


class DrillfileParser:
    def __init__(self, filename: str, context: BoardContext):
        # The drill parser shares the board context with the other importers so
        # hole locations contribute to the overall board extents used later for
        # previewing and for rebasing coordinates into a local machine origin.
        self.context = context
        # Excellon drill files define tools first, then reference them by tool
        # number when listing coordinates. This table maps tool IDs like T1 or
        # T2 to their physical drill diameters in millimetres.
        self.tool_diameters: dict = {}
        # Parsed holes are stored as (x, y, diameter) tuples so downstream code
        # can both visualize the holes and emit the corresponding drilling
        # operations in the generated G-code.
        self.holes: list[tuple[float, float, float]] = []
        # Excellon files can be metric or imperial. The parser normalizes all
        # coordinates and diameters to millimetres for the rest of the project.
        self.units_mult = 1.0

        # Like Gerber, drill files are modal: after a tool change command such
        # as T1, subsequent coordinates use that selected tool until another
        # tool number appears.
        current_tool = None

        try:
            with open(filename, "r", encoding="utf-8") as drill_file:
                for line in drill_file:
                    line = line.strip()
                    if not line or line.startswith(";"):
                        # Blank lines and semicolon-prefixed comments do not
                        # carry drill geometry, so they can be skipped outright.
                        continue

                    # Example:
                    #   METRIC,TZ
                    #   INCH,TZ
                    #
                    # Excellon headers declare whether coordinates and tool
                    # sizes are expressed in metric units or inches. The rest
                    # of the codebase works in millimetres, so inch-based files
                    # are converted using the standard 25.4 mm/inch factor.
                    if "METRIC" in line.upper():
                        self.units_mult = 1.0
                    elif "INCH" in line.upper():
                        self.units_mult = 25.4

                    # Example:
                    #   T1C0.800
                    #
                    # This defines drill tool T1 with a cutting diameter of
                    # 0.800 in the active file units. Later coordinate records
                    # only refer to "T1", so the diameter must be stored now.
                    match_tool = re.match(r"^T(\d+)C([\d\.]+)", line)
                    if match_tool:
                        tool_num = match_tool.group(1)
                        diameter = float(match_tool.group(2)) * self.units_mult
                        self.tool_diameters[tool_num] = diameter
                        continue

                    # Example:
                    #   T1
                    #
                    # A bare tool number means "switch to this drill bit". All
                    # following hole coordinates use that tool until another
                    # tool change appears.
                    match_tool_change = re.match(r"^T(\d+)$", line)
                    if match_tool_change:
                        current_tool = match_tool_change.group(1)
                        continue

                    # Example:
                    #   X12.300Y8.400
                    #
                    # In a drill file this means "drill a hole at this
                    # coordinate using the currently selected tool". Unlike the
                    # copper and edge-cuts Gerbers, there is no draw-vs-move
                    # distinction here: each coordinate is simply a drill hit.
                    match_coord = re.match(r"^X(-?\d+(\.\d+)?)Y(-?\d+(\.\d+)?)", line)
                    if match_coord and current_tool:
                        x = float(match_coord.group(1)) * self.units_mult
                        y = float(match_coord.group(3)) * self.units_mult
                        diameter = self.tool_diameters[current_tool]

                        # Store the fully resolved hole so visualization and
                        # G-code generation do not need to know anything about
                        # Excellon tool tables or modal tool state.
                        self.holes.append((x, y, diameter))
                        print("Hole (%5.1f,%5.1f),%5.2f" % (x, y, diameter))
                        # Holes may extend beyond copper or profile features,
                        # so they also update the board bounds.
                        self.context.update_point(x, y)
        except FileNotFoundError:
            print("No drill file, thats OK")
            return

    def shift(self, x_shift: float, y_shift: float) -> None:
        # After all board features are known, holes are rebased into the same
        # local coordinate system as traces and edge cuts so all generated
        # machine coordinates share one origin.
        for i, hole in enumerate(self.holes):
            self.holes[i] = (hole[0] - x_shift, hole[1] - y_shift, hole[2])
