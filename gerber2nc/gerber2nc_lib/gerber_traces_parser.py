import re
from pathlib import Path

from gerber2nc_lib.board_context import BoardContext


class GerberTracesParser:
    def __init__(self, filename: str, context: BoardContext, optional: bool = False):
        # The parser needs access to the shared board context so it can grow the
        # overall PCB bounds while Gerber features are discovered. Those bounds
        # are later used by downstream tooling for visualization and machining.
        self.context = context
        # Gerber files define "apertures" first, then refer to them later by
        # numeric ID. We keep a lookup table so drawing commands can resolve the
        # currently selected tool shape and size.
        self.apertures: dict = {}
        # Aperture selection persists in Gerber state until another D-code
        # selects a different one, so the parser mirrors that state here.
        self.current_aperture: int = -1
        # Traces are stored as line segments with an effective tool width.
        # Pads are stored separately because a flashed aperture is not the same
        # as a stroked line even if both use the same aperture definition.
        self.traces: list = []
        self.pads: list = []
        self.regions: list = []
        # Gerber coordinates are unitless integers whose meaning depends on the
        # file's declared unit mode. This multiplier converts parsed values into
        # millimetres so the rest of the project can work in one consistent unit.
        self.unit_mult: float = 1.0
        # Gerber operations are modal: once D01/D02/D03 is selected it stays
        # active until another operation code appears. Default to D02 (move)
        # so the parser does not invent a trace before the file explicitly
        # enters draw mode.
        self.current_operation: int = 2
        # The format uses the previous coordinate as the start point for a draw
        # operation, so we keep the current pen position as parser state.
        self.current_x: float | None = None
        self.current_y: float | None = None
        # G36/G37 region mode describes filled copper polygons instead of
        # stroked traces. While inside a region, D01/D02 build polygon
        # boundaries rather than ordinary aperture-width tracks.
        self.in_region: bool = False
        self.current_region_points: list[tuple[float, float]] = []

        if optional and not Path(filename).exists():
            print(f"No front copper file found at '{filename}', continuing without traces")
            return

        self._parse_gerber_file(filename)

    def _process_extended_command(self, line: str) -> None:
        # Lines wrapped in '%' are Gerber extended commands. These configure the
        # parser state rather than directly creating copper geometry.
        if not line:
            return

        # Example:
        #   %ADD10C,0.150*%
        #
        # "AD" means aperture definition. This declares aperture D10 as a
        # circle ("C") with diameter 0.150 in the file's current units.
        #
        # The regex captures:
        #   1. The aperture number after ADD, such as 10
        #   2. The aperture template/type, such as C, R, or RoundRect
        #   3. The parameter list after the comma, such as 0.150 or 1.0X2.0
        #
        # We parse this because later drawing commands only say "use D10";
        # without the earlier ADD command, there is no way to know whether D10
        # means a 0.15 mm circular trace, a rectangular pad, etc.
        aperture_match = re.match(r"%ADD(\d+)([^,]+),([^*]+)\*%", line)
        if aperture_match:
            aperture_num = int(aperture_match.group(1))
            aperture_type = aperture_match.group(2)
            # Aperture parameters in Gerber are commonly separated by 'X'.
            # For rectangles this is width X height. For more complex vendor or
            # tool-specific apertures it can be several shape parameters.
            params = aperture_match.group(3).split("X")

            if aperture_type == "C":
                # Circular aperture: used both for drawing round-ended traces
                # and for flashing circular pads/vias.
                diameter = float(params[0])
                self.apertures[aperture_num] = {"type": "circle", "diameter": diameter}
            elif aperture_type == "R":
                # Rectangular aperture: commonly used for rectangular pads.
                # If height is omitted, fall back to a square.
                width = float(params[0])
                height = float(params[1]) if len(params) > 1 else width
                self.apertures[aperture_num] = {
                    "type": "rectangle",
                    "width": width,
                    "height": height,
                }
            elif aperture_type == "RoundRect":
                # Rounded rectangles appear in some Gerber exports as a custom
                # aperture format. The downstream code only needs the overall pad
                # width and height, so this parser reduces the detailed corner
                # geometry to a simple rectangle-sized bounding shape.
                corner_radius = float(params[0])
                x1, y1 = float(params[1]), float(params[2])
                x2, y2 = float(params[3]), float(params[4])
                width = abs(x2) + abs(x1) + corner_radius
                height = abs(y2) + abs(y1) + corner_radius
                self.apertures[aperture_num] = {
                    "type": "rectangle",
                    "width": width,
                    "height": height,
                }

        # Unit mode matters because Gerber numbers are just coordinate digits.
        # The rest of this project assumes millimetres, so inch-based files are
        # converted by multiplying by 25.4.
        if "MOMM*%" in line:
            self.unit_mult = 1
        elif "MOIN*%" in line:
            self.unit_mult = 25.4

    def _process_command(self, line: str) -> None:
        # Normal Gerber commands typically end with '*'. Removing it simplifies
        # the regexes below without changing the command meaning.
        line = line.rstrip("*")

        if line == "G36":
            self.in_region = True
            self.current_region_points = []
            return

        if line == "G37":
            self._finish_region()
            return

        # Example:
        #   D10*
        #   G54D10*
        #
        # A bare D-code >= 10 usually selects the active aperture/tool. It does
        # not draw anything immediately; it changes how subsequent draw/flash
        # commands should be interpreted.
        aperture_match = re.match(r"(?:G54)?D(\d+)", line)
        if aperture_match:
            aperture_num = int(aperture_match.group(1))
            if aperture_num >= 10:
                self.current_aperture = aperture_num
            elif aperture_num <= 3:
                # D01/D02/D03 are modal operations. A line can select one of
                # these modes without also supplying coordinates, so record it
                # for the next coordinate command.
                self.current_operation = aperture_num
            return

        # Example:
        #   X012345Y067890D01*
        #   X012345Y067890D03*
        #   X012345Y067890*
        #   X012345D02*
        #   Y067890*
        #
        # This parser handles coordinate commands that specify X, Y, and an
        # operation code. Gerber coordinates are also modal, so a command may
        # omit X or Y to mean "reuse the previous value on that axis".
        #   D01 = draw from the previous point to this point
        #   D02 = move to this point without drawing
        #   D03 = flash the current aperture at this point
        #
        # In Gerber, "flash" means "stamp the currently selected aperture shape
        # here as a pad", while "draw" means "sweep the current aperture from
        # the old point to the new point", which creates a trace.
        coord_match = re.match(r"^(?:X(-?[0-9.]+))?(?:Y(-?[0-9.]+))?(?:D0([123]))?$", line)
        if coord_match:
            if coord_match.group(1) is None and coord_match.group(2) is None:
                return

            # This code assumes the file uses six decimal places worth of Gerber
            # coordinate precision, so values are scaled by 1e-6 and then
            # converted into millimetres using the active unit multiplier.
            if coord_match.group(1) is not None:
                x = float(coord_match.group(1)) * 0.000001 * self.unit_mult
            else:
                x = self.current_x

            if coord_match.group(2) is not None:
                y = float(coord_match.group(2)) * 0.000001 * self.unit_mult
            else:
                y = self.current_y

            if x is None or y is None:
                # A modal axis cannot be reused before an initial absolute value
                # has been established, so ignore malformed early coordinates.
                return

            if coord_match.group(3) is not None:
                self.current_operation = int(coord_match.group(3))
            operation = self.current_operation

            # Bounds need a little safety margin around the literal coordinate.
            # Flashed pads are larger than simple line centers, so they receive
            # a larger margin to keep the overall board extents conservative.
            margin = 1.5 if operation == 3 else 0.6
            self.context.update_bounds(x, y, margin)

            if self.in_region:
                self._process_region_point(x, y, operation)
                self.current_x, self.current_y = x, y
                return

            if operation == 1:
                # D01 draws using the currently selected aperture. For circular
                # apertures the diameter is the trace width; for rectangular
                # apertures we use the width as the effective stroke width.
                if (
                    self.current_x is not None
                    and self.current_y is not None
                    and self.current_aperture in self.apertures
                ):
                    aperture = self.apertures[self.current_aperture]
                    width = aperture.get("diameter", aperture.get("width", 0.1))
                    self.traces.append(
                        [(self.current_x, self.current_y), (x, y), width]
                    )
            elif operation == 3:
                # D03 flashes the aperture in one spot instead of sweeping it
                # along a path. This is how Gerber usually represents pads,
                # lands, and via annuli.
                if self.current_aperture and self.current_aperture in self.apertures:
                    aperture = self.apertures[self.current_aperture]
                    self.pads.append([(x, y), aperture])

            # Even a non-drawing move updates the current coordinate because
            # subsequent D01 commands start from the latest known position.
            self.current_x, self.current_y = x, y

    def _process_region_point(self, x: float, y: float, operation: int) -> None:
        point = (x, y)

        if operation == 2 or not self.current_region_points:
            self.current_region_points = [point]
            return

        if operation == 1:
            if self.current_region_points[-1] != point:
                self.current_region_points.append(point)

    def _finish_region(self) -> None:
        self.in_region = False

        if len(self.current_region_points) < 3:
            self.current_region_points = []
            return

        if self.current_region_points[0] == self.current_region_points[-1]:
            region_points = self.current_region_points[:-1]
        else:
            region_points = self.current_region_points

        if len(region_points) >= 3:
            self.regions.append(region_points)

        self.current_region_points = []

    def _parse_gerber_file(self, filename: str) -> None:
        with open(filename, "r", encoding="utf-8") as gerber_file:
            content = gerber_file.read()

        # Gerber is line-oriented in practice for most CAM exports. Extended
        # commands start with '%', while ordinary plotting commands do not.
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("%"):
                self._process_extended_command(line)
            else:
                self._process_command(line)

    def shift(self, x_shift: float, y_shift: float) -> None:
        # After parsing, geometry may need to be re-based into a local origin
        # for machining or visualization. Shifting both traces and pads keeps
        # every generated feature aligned in the same coordinate system.
        for trace in self.traces:
            for startstop in range(2):
                x, y = trace[startstop]
                trace[startstop] = [x - x_shift, y - y_shift]

        for pad in self.pads:
            x, y = pad[0]
            pad[0] = [x - x_shift, y - y_shift]

        for i, region in enumerate(self.regions):
            self.regions[i] = [(x - x_shift, y - y_shift) for x, y in region]
