from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

from shapely import affinity
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import unary_union

from .nc_origin import format_origin_point


class CamGenerator:
    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.safe_height = 3.0
        self.surface_height = 0.1
        self.board_thickness = 1.6
        self.breakthrough_depth = 0.2
        self.isolation_depth = -0.1
        self.feed_rate = 300
        self.plunge_rate = 150
        self.edge_depth_step = 0.4
        self.circle_segment_count = 24

    def generate_front_isolation(
        self,
        gerber,
        *,
        output_name: str,
        tool_tip_diameter: float,
        origin_point: tuple[float, float],
    ) -> Path:
        geometry = self._combined_copper_geometry(gerber)
        paths = self._isolation_paths(geometry, tool_tip_diameter)
        translated_paths = self._translate_paths_to_origin(
            paths,
            origin_point,
        )
        output_path = self.output_directory / output_name
        self._write_line_operation(
            output_path,
            translated_paths,
            cut_depth=self.isolation_depth,
            spindle_speed=12000,
            tool_comment=(
                f"V-bit tip {tool_tip_diameter:.3f} mm, "
                f"origin {format_origin_point(origin_point)}"
            ),
        )
        return output_path

    def generate_back_isolation(
        self,
        gerber,
        *,
        output_name: str,
        tool_tip_diameter: float,
        mirror_edge: str,
        board_bounds: tuple[float, float, float, float],
        origin_point: tuple[float, float],
    ) -> Path:
        geometry = self._combined_copper_geometry(gerber)
        mirrored = self._mirror_geometry(geometry, mirror_edge, board_bounds)
        paths = self._isolation_paths(mirrored, tool_tip_diameter)
        translated_paths = self._translate_paths_to_origin(
            paths,
            origin_point,
        )
        output_path = self.output_directory / output_name
        tool_comment = (
            f"V-bit tip {tool_tip_diameter:.3f} mm, "
            f"origin {format_origin_point(origin_point)}"
        )
        if mirror_edge:
            tool_comment += f" mirrored on {mirror_edge}"
        self._write_line_operation(
            output_path,
            translated_paths,
            cut_depth=self.isolation_depth,
            spindle_speed=12000,
            tool_comment=tool_comment,
        )
        return output_path

    def generate_drill_operations(
        self,
        holes: list[tuple[float, float, float]],
        *,
        output_name: str,
        drill_diameter: float,
        mill_diameter: float,
        origin_point: tuple[float, float],
    ) -> Path:
        output_path = self.output_directory / output_name
        final_hole_depth = -(self.board_thickness + self.breakthrough_depth)
        origin_x, origin_y = origin_point
        translated_holes = [
            (x - origin_x, y - origin_y, diameter) for x, y, diameter in holes
        ]
        with output_path.open("w", encoding="utf-8") as gcode_file:
            self._write_header(gcode_file)
            gcode_file.write(
                f"(load drill {drill_diameter:.3f} mm, "
                f"origin {format_origin_point(origin_point)})\nT1 M06\nS12000 M3\n"
            )
            for x, y, target_diameter in translated_holes:
                if target_diameter + 1e-9 < drill_diameter:
                    raise ValueError(
                        f"Hole diameter {target_diameter:.3f} mm is smaller than selected drill {drill_diameter:.3f} mm."
                    )
                self._write_peck_drill(
                    gcode_file,
                    x,
                    y,
                    final_hole_depth,
                )
            gcode_file.write("M5\n")

            holes_to_enlarge = [
                (x, y, target_diameter)
                for x, y, target_diameter in translated_holes
                if target_diameter > drill_diameter + 1e-9
            ]
            if holes_to_enlarge:
                gcode_file.write(
                    f"(load end mill {mill_diameter:.3f} mm for hole enlargement)\nT2 M06\nS12000 M3\n"
                )
                for x, y, target_diameter in holes_to_enlarge:
                    self._write_hole_enlargement(
                        gcode_file,
                        x,
                        y,
                        drill_diameter,
                        target_diameter,
                        mill_diameter,
                        final_hole_depth,
                    )
                gcode_file.write("M5\n")
            self._write_footer(gcode_file)
        return output_path

    def generate_edge_cuts(
        self,
        outlines: list[list[tuple[float, float]]],
        *,
        output_name: str,
        mill_diameter: float,
        origin_point: tuple[float, float],
    ) -> Path:
        output_path = self.output_directory / output_name
        final_depth = -(self.board_thickness + self.breakthrough_depth)
        translated_outlines = [
            self._translate_points_to_origin(outline, origin_point) for outline in outlines
        ]
        with output_path.open("w", encoding="utf-8") as gcode_file:
            self._write_header(gcode_file)
            gcode_file.write(
                f"(load square end mill {mill_diameter:.3f} mm, "
                f"origin {format_origin_point(origin_point)})\nT1 M06\nS12000 M3\n"
            )
            current_depth = 0.0
            while current_depth > final_depth:
                current_depth = max(current_depth - self.edge_depth_step, final_depth)
                for outline in translated_outlines:
                    self._write_polyline(
                        gcode_file,
                        outline,
                        cut_depth=current_depth,
                    )
            gcode_file.write("M5\n")
            self._write_footer(gcode_file)
        return output_path

    def _combined_copper_geometry(self, gerber):
        traces = [
            LineString([start, end]).buffer(width * 0.5)
            for start, end, width in gerber.traces
        ]
        pads = []
        for center, aperture in gerber.pads:
            if aperture["type"] == "circle":
                pads.append(Point(center[0], center[1]).buffer(float(aperture["diameter"]) * 0.5))
            elif aperture["type"] == "rectangle":
                half_width = float(aperture["width"]) * 0.5
                half_height = float(aperture["height"]) * 0.5
                pads.append(
                    box(
                        center[0] - half_width,
                        center[1] - half_height,
                        center[0] + half_width,
                        center[1] + half_height,
                    )
                )
        regions = [
            Polygon(region).buffer(0)
            for region in gerber.regions
            if len(region) >= 3
        ]
        geometry = unary_union(traces + pads + regions)
        if geometry.is_empty:
            raise ValueError(f"No copper geometry found in {gerber.display_name}.")
        return geometry

    def _isolation_paths(self, geometry, tool_tip_diameter: float):
        offset = max(tool_tip_diameter * 0.5, 0.1)
        boundary = geometry.buffer(offset).simplify(0.03).boundary
        if isinstance(boundary, LineString):
            return [boundary]
        if isinstance(boundary, MultiLineString):
            return list(boundary.geoms)
        if hasattr(boundary, "geoms"):
            return [item for item in boundary.geoms if isinstance(item, LineString)]
        raise ValueError("Unsupported geometry returned for isolation paths.")

    def _mirror_geometry(
        self,
        geometry,
        mirror_edge: str,
        board_bounds: tuple[float, float, float, float],
    ):
        x_min, x_max, y_min, y_max = board_bounds
        if mirror_edge == "left":
            return affinity.scale(geometry, xfact=-1, yfact=1, origin=(x_min, 0))
        if mirror_edge == "right":
            return affinity.scale(geometry, xfact=-1, yfact=1, origin=(x_max, 0))
        if mirror_edge == "top":
            return affinity.scale(geometry, xfact=1, yfact=-1, origin=(0, y_max))
        if mirror_edge == "bottom":
            return affinity.scale(geometry, xfact=1, yfact=-1, origin=(0, y_min))
        return geometry

    def _translate_paths_to_origin(
        self,
        paths: list[LineString],
        origin_point: tuple[float, float],
    ) -> list[LineString]:
        origin_x, origin_y = origin_point
        translated = [
            affinity.translate(path, xoff=-origin_x, yoff=-origin_y) for path in paths
        ]
        return translated

    def _translate_points_to_origin(
        self,
        points: list[tuple[float, float]],
        origin_point: tuple[float, float],
    ) -> list[tuple[float, float]]:
        origin_x, origin_y = origin_point
        translated = [(x - origin_x, y - origin_y) for x, y in points]
        return translated

    def _write_header(self, gcode_file) -> None:
        gcode_file.write("%\nG21\nG90\n")
        gcode_file.write(f"G0 Z{self.safe_height:.3f}\n")

    def _write_footer(self, gcode_file) -> None:
        gcode_file.write(f"G0 Z{self.safe_height:.3f}\n")
        gcode_file.write("M30\n%\n")

    def _write_line_operation(
        self,
        output_path: Path,
        paths: list[LineString],
        *,
        cut_depth: float,
        spindle_speed: int,
        tool_comment: str,
    ) -> None:
        with output_path.open("w", encoding="utf-8") as gcode_file:
            self._write_header(gcode_file)
            gcode_file.write(f"(load {tool_comment})\nT1 M06\nS{spindle_speed:d} M3\n")
            for path in paths:
                self._write_polyline(gcode_file, list(path.coords), cut_depth=cut_depth)
            gcode_file.write("M5\n")
            self._write_footer(gcode_file)

    def _write_polyline(self, gcode_file, points, *, cut_depth: float) -> None:
        if not points:
            return
        start_x, start_y = points[0]
        gcode_file.write(f"G0 X{start_x:.3f} Y{start_y:.3f}\n")
        gcode_file.write(f"G0 Z{self.surface_height:.3f}\n")
        gcode_file.write(f"G1 Z{cut_depth:.3f} F{self.plunge_rate:d}\n")
        gcode_file.write(f"G1 F{self.feed_rate:d}\n")
        for x, y in points[1:]:
            gcode_file.write(f"G1 X{x:.3f} Y{y:.3f}\n")
        gcode_file.write(f"G0 Z{self.safe_height:.3f}\n")

    def _write_peck_drill(self, gcode_file, x: float, y: float, final_depth: float) -> None:
        gcode_file.write(f"G0 X{x:.3f} Y{y:.3f}\n")
        gcode_file.write(f"G0 Z{self.surface_height:.3f}\n")
        current_depth = 0.0
        while current_depth > final_depth:
            current_depth = max(current_depth - 0.5, final_depth)
            gcode_file.write(f"G1 Z{current_depth:.3f} F{self.plunge_rate:d}\n")
            gcode_file.write("G0 Z1.000\n")
        gcode_file.write(f"G0 Z{self.safe_height:.3f}\n")

    def _write_hole_enlargement(
        self,
        gcode_file,
        x_center: float,
        y_center: float,
        drilled_diameter: float,
        target_diameter: float,
        mill_diameter: float,
        final_depth: float,
    ) -> None:
        target_radius = target_diameter * 0.5
        start_offset = max(0.0, (drilled_diameter - mill_diameter) * 0.5)
        target_offset = max(0.0, target_radius - (mill_diameter * 0.5))
        current_depth = 0.0
        while current_depth > final_depth:
            current_depth = max(current_depth - 0.4, final_depth)
            gcode_file.write(f"G0 X{x_center:.3f} Y{y_center:.3f}\n")
            gcode_file.write(f"G0 Z{self.surface_height:.3f}\n")
            gcode_file.write(f"G1 Z{current_depth:.3f} F{self.plunge_rate:d}\n")
            current_offset = start_offset
            while current_offset < target_offset:
                current_offset = min(current_offset + 0.1, target_offset)
                gcode_file.write(
                    f"G1 X{x_center + current_offset:.3f} Y{y_center:.3f} F{self.feed_rate:d}\n"
                )
                for segment in range(1, self.circle_segment_count + 1):
                    angle = (2.0 * pi * segment) / self.circle_segment_count
                    x_pos = x_center + cos(angle) * current_offset
                    y_pos = y_center + sin(angle) * current_offset
                    gcode_file.write(f"G1 X{x_pos:.3f} Y{y_pos:.3f}\n")
            gcode_file.write(f"G0 Z{self.safe_height:.3f}\n")
