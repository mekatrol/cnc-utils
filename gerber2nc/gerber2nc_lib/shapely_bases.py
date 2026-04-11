"""Convert parsed PCB copper features into machinable isolation-routing paths.

Gerber copper layers describe the metal that should remain on the PCB after
fabrication. For a subtractive CNC workflow the job is inverted: the spindle
must remove material around those copper features while leaving the traces and
pads untouched. This module turns the parsed copper features into Shapely
geometry, unions them into a "copper to keep" shape, and then offsets that
shape outward to create routing passes.
"""

from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import unary_union


class ShapelyBases:
    def __init__(self, parser):
        # The parser separates copper into stroked traces, flashed pads, and
        # filled polygon regions. Shapely gives us a common geometry engine for
        # merging all three into one shape that represents "do not mill here".
        traces = []
        pads = []
        regions = []

        for trace in parser.traces:
            width_mm = trace[2]
            # A Gerber draw command means "sweep the aperture along this line".
            # Buffering the centerline by half the aperture width reconstructs
            # the actual copper area occupied by that trace.
            traces.append(LineString([trace[0], trace[1]]).buffer(width_mm / 2))

        for pad in parser.pads:
            x_mm, y_mm = pad[0]
            aperture = pad[1]

            if aperture["type"] == "circle":
                print("circle pad at: (%7.2f,%7.2f)" % (x_mm, y_mm))
                radius = aperture["diameter"] / 2
                # A flashed circular aperture becomes a circular copper island.
                pads.append(Point(x_mm, y_mm).buffer(radius))
            elif aperture["type"] == "rectangle":
                width = aperture["width"] / 2
                height = aperture["height"] / 2
                # Rectangular pads are stored by center point plus dimensions,
                # so convert them into their true copper footprint here.
                pads.append(
                    box(x_mm - width, y_mm - height, x_mm + width, y_mm + height)
                )
            else:
                print("Pad type :", aperture["type"], "ignored")

        for region in parser.regions:
            if len(region) >= 3:
                # Region mode in Gerber is already an explicit filled polygon,
                # so it can be used directly as a copper keep-area.
                regions.append(Polygon(region))

        # Unioning the copper features prevents the toolpath generator from
        # treating overlapping pads/traces as separate islands with redundant
        # internal boundaries.
        self.combined_geometry = unary_union(traces + pads + regions)

    def compute_trace_toolpaths(
        self, offset_distance: float, num_passes: int, path_spacing: float
    ):
        if self.combined_geometry.is_empty:
            return MultiLineString()

        all_passes = []

        for passnum in range(0, num_passes):
            offset = offset_distance + path_spacing * passnum
            # Buffering outward creates a clearance envelope around the copper.
            # Taking its boundary yields the cutter centerline for one
            # isolation pass at that distance from the preserved copper.
            #
            # `simplify(0.03)` smooths tiny geometric artifacts from repeated
            # buffering so the resulting G-code is less noisy and easier for a
            # hobby CNC controller to execute cleanly.
            thispath = self.combined_geometry.buffer(offset).simplify(0.03).boundary

            if thispath.geom_type == "LineString":
                all_passes.append(thispath)
            elif hasattr(thispath, "geoms"):
                # Multi-island boards produce many separate loops. Flatten them
                # into one MultiLineString so downstream code can iterate over
                # each individual milling path uniformly.
                all_passes += list(thispath.geoms)

        return MultiLineString(all_passes)
