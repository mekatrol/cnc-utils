"""Emit the final CNC program for isolation routing, edge marking, and drilling.

The generated program assumes a manual tool-change workflow:
- Tool 1 engraves isolation paths around the copper and marks the edge cut.
- Additional tools drill holes using configured bit diameters.
- When a drilled hole is still undersized, a milling cutter enlarges it in
  radial steps at each depth until the requested finished diameter is reached.
"""

from math import cos, pi, sin
from pathlib import Path


class GcodeGenerator:
    def __init__(self, board_height: float, drilling_config: dict):
        # Isolation-routing settings remain fixed for now because the existing
        # workflow already depends on these depths and feeds.
        self.board_height = board_height
        self.isolation_spindle_speed = 12000
        self.cut_depth = -0.1
        self.edge_cut_depth = -0.2
        self.surface_start_height = 0.1
        self.feed_rate = 450
        self.plunge_feed_rate = 200

        # Hole-making parameters come from the YAML config so the operator can
        # describe the available drills, board thickness, and enlargement flow
        # without editing code.
        self.available_drill_bits = sorted(
            float(bit) for bit in drilling_config["available_drill_bits"]
        )
        self.pcb_thickness = float(drilling_config["pcb_thickness"])
        self.breakthrough_depth = float(drilling_config["breakthrough_depth"])
        self.safe_height = float(drilling_config["safe_height"])
        self.hole_start = float(drilling_config["start_height"])
        self.drill_spindle_speed = int(drilling_config["spindle_speed"])
        self.drill_plunge_feed_rate = int(drilling_config["plunge_feed_rate"])
        self.peck_depth = float(drilling_config["peck_depth"])
        self.chip_clear_height = float(drilling_config["chip_clear_height"])
        self.mill_tool_diameter = float(drilling_config["mill_tool_diameter"])
        self.mill_feed_rate = int(drilling_config["mill_feed_rate"])
        self.mill_plunge_feed_rate = int(drilling_config["mill_plunge_feed_rate"])
        self.mill_depth_step = float(drilling_config["mill_depth_step"])
        self.mill_start_height = float(drilling_config["mill_start_height"])
        self.max_enlarging_step = float(drilling_config["max_enlarging_step"])
        self.circle_segment_count = int(drilling_config["circle_segment_count"])
        self.final_hole_depth = -(self.pcb_thickness + self.breakthrough_depth)

    def output_gcode(
        self, filename: str, edgecuts: list, trace_mill_geometry, holes: list
    ) -> None:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        drill_groups, enlargement_groups = self.plan_hole_operations(holes)

        with output_path.open("w", encoding="utf-8") as gcode_file:
            gcode_file.write("%\n")
            gcode_file.write("G21  ; Set units to mm\n")
            gcode_file.write("G90  ; Absolute positioning\n")
            gcode_file.write(f"G0 Z{self.safe_height:.2f}  ; Move to safe height\n")
            gcode_file.write("(load 0.2 mm engraving tool)\nT1 M06\n")
            gcode_file.write(
                f"S{self.isolation_spindle_speed:d} M3  ; Start spindle clockwise\n"
            )

            for path in trace_mill_geometry.geoms:
                started = False
                for x, y in path.coords:
                    if not started:
                        gcode_file.write(f"G0 X{x:.2f} Y{y:.2f}\n")
                        gcode_file.write(f"G0 Z{self.surface_start_height:.2f}\n")
                        gcode_file.write(
                            f"G1 Z{self.cut_depth:.3f} F{self.plunge_feed_rate:d}\n"
                        )
                        gcode_file.write(f"G1 F{self.feed_rate:d}\n")
                        started = True
                    else:
                        gcode_file.write(f"G1 X{x:.2f} Y{y:.2f}\n")

                gcode_file.write(f"G0 Z{self.safe_height:.2f}\n")

            gcode_file.write("(mill edgecut mark)\n")
            started = False
            for x, y in edgecuts:
                if not started:
                    gcode_file.write(f"G0 X{x:.2f} Y{y:.2f}\n")
                    gcode_file.write(f"G0 Z{self.surface_start_height:.2f}\n")
                    gcode_file.write(
                        f"G1 Z{self.edge_cut_depth:.3f} F{self.plunge_feed_rate:d}\n"
                    )
                    gcode_file.write(f"G1 F{self.feed_rate:d}\n")
                    started = True
                else:
                    gcode_file.write(f"G1 X{x:.2f} Y{y:.2f}\n")

            gcode_file.write(f"G0 Z{self.safe_height:.2f}\n")
            gcode_file.write("M5  ; Stop spindle\n")

            next_tool_number = 2
            for drill_diameter in sorted(drill_groups):
                hole_group = drill_groups[drill_diameter]
                gcode_file.write(
                    f"(load {drill_diameter:.2f} mm drill)\nT{next_tool_number:d} M06\n"
                )
                gcode_file.write(
                    f"S{self.drill_spindle_speed:d} M3  ; Start spindle clockwise\n"
                )
                for x, y, target_diameter in hole_group:
                    gcode_file.write(
                        f"(drill {target_diameter:.2f} mm hole using {drill_diameter:.2f} mm drill)\n"
                    )
                    self.write_peck_drill_cycle(gcode_file, x, y)
                gcode_file.write("M5  ; Stop spindle\n")
                next_tool_number += 1

            if enlargement_groups:
                gcode_file.write(
                    f"(load {self.mill_tool_diameter:.2f} mm hole-enlarging mill)\n"
                    f"T{next_tool_number:d} M06\n"
                )
                gcode_file.write(
                    f"S{self.drill_spindle_speed:d} M3  ; Start spindle clockwise\n"
                )
                for target_diameter in sorted(enlargement_groups):
                    for x, y, drilled_diameter in enlargement_groups[target_diameter]:
                        gcode_file.write(
                            f"(mill hole from {drilled_diameter:.2f} mm to {target_diameter:.2f} mm)\n"
                        )
                        self.write_hole_enlargement(gcode_file, x, y, drilled_diameter, target_diameter)
                gcode_file.write("M5  ; Stop spindle\n")

            gcode_file.write(
                f"G0 X0.00 Y{self.board_height:.1f} Z50.00  ; Return home, raise spindle out of the way\n"
            )
            gcode_file.write("M30 ; End of program\n")
            gcode_file.write("%\n")

        print("\nG-code generated in '%s'" % (output_path))

    def plan_hole_operations(self, holes: list) -> tuple[dict, dict]:
        drill_groups: dict[float, list] = {}
        enlargement_groups: dict[float, list] = {}

        for x, y, target_diameter in holes:
            drill_diameter = self.select_drill_bit(float(target_diameter))
            drill_groups.setdefault(drill_diameter, []).append((x, y, target_diameter))

            if drill_diameter < target_diameter:
                if drill_diameter < self.mill_tool_diameter:
                    raise ValueError(
                        "Cannot enlarge %.2f mm hole at X%.2f Y%.2f: selected drill %.2f mm is smaller than mill tool %.2f mm."
                        % (
                            target_diameter,
                            x,
                            y,
                            drill_diameter,
                            self.mill_tool_diameter,
                        )
                    )
                enlargement_groups.setdefault(float(target_diameter), []).append(
                    (x, y, drill_diameter)
                )

        return drill_groups, enlargement_groups

    def select_drill_bit(self, target_diameter: float) -> float:
        matching_bits = [
            diameter for diameter in self.available_drill_bits if diameter <= target_diameter
        ]
        if not matching_bits:
            configured_sizes = ", ".join(
                f"{diameter:.2f}" for diameter in self.available_drill_bits
            )
            raise ValueError(
                "No configured drill bit can make %.2f mm hole without oversizing it. "
                "Available drill bits: [%s] mm. Add a drill <= %.2f mm to "
                "drilling.available_drill_bits in gerber2nc.yaml."
                % (target_diameter, configured_sizes, target_diameter)
            )
        return matching_bits[-1]

    def write_peck_drill_cycle(self, gcode_file, x: float, y: float) -> None:
        gcode_file.write(f"G0 X{x:.2f} Y{y:.2f}\n")
        gcode_file.write(f"G0 Z{self.hole_start:.2f}\n")

        current_depth = 0.0
        while current_depth > self.final_hole_depth:
            next_depth = max(current_depth - self.peck_depth, self.final_hole_depth)
            gcode_file.write(
                f"G1 Z{next_depth:.3f} F{self.drill_plunge_feed_rate:d}\n"
            )
            gcode_file.write(f"G0 Z{self.chip_clear_height:.2f}\n")
            current_depth = next_depth

        gcode_file.write(f"G0 Z{self.safe_height:.2f}\n")

    def write_hole_enlargement(
        self,
        gcode_file,
        x_center: float,
        y_center: float,
        drilled_diameter: float,
        target_diameter: float,
    ) -> None:
        target_radius = target_diameter / 2.0
        drill_radius = drilled_diameter / 2.0
        tool_radius = self.mill_tool_diameter / 2.0
        start_offset = max(0.0, drill_radius - tool_radius)
        target_offset = target_radius - tool_radius

        if target_offset <= 0:
            return

        current_depth = 0.0
        while current_depth > self.final_hole_depth:
            next_depth = max(current_depth - self.mill_depth_step, self.final_hole_depth)
            gcode_file.write(f"G0 X{x_center:.2f} Y{y_center:.2f}\n")
            gcode_file.write(f"G0 Z{self.mill_start_height:.2f}\n")
            gcode_file.write(
                f"G1 Z{next_depth:.3f} F{self.mill_plunge_feed_rate:d}\n"
            )
            self.write_enlarging_passes(
                gcode_file,
                x_center,
                y_center,
                start_offset,
                target_offset,
            )
            gcode_file.write(f"G0 Z{self.safe_height:.2f}\n")
            current_depth = next_depth

    def write_enlarging_passes(
        self,
        gcode_file,
        x_center: float,
        y_center: float,
        start_offset: float,
        target_offset: float,
    ) -> None:
        current_offset = start_offset
        gcode_file.write(f"G1 F{self.mill_feed_rate:d}\n")
        if current_offset > 0:
            gcode_file.write(f"G1 X{x_center + current_offset:.3f} Y{y_center:.3f}\n")

        while current_offset < target_offset:
            next_offset = min(current_offset + self.max_enlarging_step, target_offset)
            gcode_file.write(f"G1 X{x_center + next_offset:.3f} Y{y_center:.3f}\n")
            for segment in range(1, self.circle_segment_count + 1):
                angle = (2.0 * pi * segment) / self.circle_segment_count
                x_pos = x_center + cos(angle) * next_offset
                y_pos = y_center + sin(angle) * next_offset
                gcode_file.write(f"G1 X{x_pos:.3f} Y{y_pos:.3f}\n")
            current_offset = next_offset
