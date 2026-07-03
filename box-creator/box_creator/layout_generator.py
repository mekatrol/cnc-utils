from __future__ import annotations

from math import ceil

from .box_settings import BoxSettings
from .geometry import Panel, Point


class LayoutGenerator:
    def generate(self, settings: BoxSettings) -> list[Panel]:
        x_size = settings.size_x
        y_size = settings.size_y
        z_size = settings.size_z
        panels = [
            self._panel(
                "front", x_size, z_size, ("slot", "tab", "tab", "slot"), settings
            ),
            self._panel(
                "back", x_size, z_size, ("slot", "tab", "tab", "slot"), settings
            ),
            self._panel(
                "left", y_size, z_size, ("tab", "slot", "slot", "tab"), settings
            ),
            self._panel(
                "right", y_size, z_size, ("tab", "slot", "slot", "tab"), settings
            ),
        ]
        if settings.box_kind == "box":
            panels.extend(
                [
                    self._panel(
                        "bottom", x_size, y_size, ("tab", "tab", "tab", "tab"), settings
                    ),
                    self._panel(
                        "top",
                        x_size,
                        y_size,
                        ("slot", "slot", "slot", "slot"),
                        settings,
                    ),
                ]
            )
        else:
            panels.append(
                self._panel(
                    "bottom", x_size, y_size, ("tab", "tab", "tab", "tab"), settings
                )
            )

        self._place_panels_on_stock(panels, settings)
        return panels

    def _place_panels_on_stock(
        self, panels: list[Panel], settings: BoxSettings
    ) -> None:
        stock_width = settings.stock_width
        stock_height = settings.stock_height
        sheet_gap = settings.layout_gap * 2.0
        margin = max(settings.bit_diameter, settings.relief_diameter) * 0.5
        cursor_x = margin
        cursor_y = margin
        row_height = 0.0
        stock_index = 0
        stock_origin_x = 0.0
        stock_origin_y = 0.0

        for panel in panels:
            min_x, max_x, min_y, max_y = panel.bounds
            panel_width = max_x - min_x
            panel_height = max_y - min_y
            if (
                panel_width + margin * 2.0 > stock_width
                or panel_height + margin * 2.0 > stock_height
            ):
                raise ValueError(
                    f"{panel.name} panel does not fit on "
                    f"{stock_width:.3f} x {stock_height:.3f} mm stock"
                )
            if cursor_x > margin and cursor_x + panel_width + margin > stock_width:
                cursor_x = margin
                cursor_y += row_height + settings.layout_gap
                row_height = 0.0
            if cursor_y > margin and cursor_y + panel_height + margin > stock_height:
                stock_index += 1
                stock_origin_x += stock_width + sheet_gap
                cursor_x = margin
                cursor_y = margin
                row_height = 0.0
            self._move_panel(panel, cursor_x - min_x, cursor_y - min_y)
            self._assign_stock(panel, stock_index, stock_origin_x, stock_origin_y)
            cursor_x += panel_width + settings.layout_gap
            row_height = max(row_height, panel_height)

    def _panel(
        self,
        name: str,
        width: float,
        height: float,
        edge_modes: tuple[str, str, str, str],
        settings: BoxSettings,
    ) -> Panel:
        top, right, bottom, left = edge_modes
        outline: list[Point] = [Point(0.0, 0.0)]
        outline.extend(self._horizontal_edge(0.0, 0.0, width, "top", top, settings)[1:])
        outline.extend(
            self._vertical_edge(width, 0.0, height, "right", right, settings)[1:]
        )
        outline.extend(
            self._horizontal_edge(width, height, -width, "bottom", bottom, settings)[1:]
        )
        outline.extend(
            self._vertical_edge(0.0, height, -height, "left", left, settings)[1:]
        )
        if outline[-1] != outline[0]:
            outline.append(outline[0])
        panel = Panel(
            name=name,
            width=width,
            height=height,
            origin_x=0.0,
            origin_y=0.0,
            outline=outline,
        )
        panel.relief_points = self._relief_points(panel, settings)
        return panel

    def _horizontal_edge(
        self,
        start_x: float,
        y: float,
        length: float,
        edge_name: str,
        mode: str,
        settings: BoxSettings,
    ) -> list[Point]:
        sign = 1.0 if length >= 0.0 else -1.0
        normal = -1.0 if edge_name == "top" else 1.0
        return self._finger_edge(
            start_x, y, length, sign, 0.0, 0.0, normal, mode, settings
        )

    def _vertical_edge(
        self,
        x: float,
        start_y: float,
        length: float,
        edge_name: str,
        mode: str,
        settings: BoxSettings,
    ) -> list[Point]:
        sign = 1.0 if length >= 0.0 else -1.0
        normal = 1.0 if edge_name == "right" else -1.0
        return self._finger_edge(
            x, start_y, length, 0.0, sign, normal, 0.0, mode, settings
        )

    def _finger_edge(
        self,
        start_x: float,
        start_y: float,
        length: float,
        direction_x: float,
        direction_y: float,
        normal_x: float,
        normal_y: float,
        mode: str,
        settings: BoxSettings,
    ) -> list[Point]:
        run = abs(length)
        finger_count = max(3, int(ceil(run / max(settings.finger_width, 1.0))))
        if finger_count % 2 == 0:
            finger_count += 1
        pitch = run / finger_count
        depth = settings.material_thickness
        points = [Point(start_x, start_y)]
        for index in range(finger_count):
            along_a = index * pitch
            along_b = (index + 1) * pitch
            is_raised = index % 2 == 0
            offset = (
                depth
                if (mode == "tab" and is_raised) or (mode == "slot" and not is_raised)
                else 0.0
            )
            offset *= 1.0 if mode == "tab" else -1.0
            a = Point(
                start_x + (direction_x * along_a) + (normal_x * offset),
                start_y + (direction_y * along_a) + (normal_y * offset),
            )
            b = Point(
                start_x + (direction_x * along_b) + (normal_x * offset),
                start_y + (direction_y * along_b) + (normal_y * offset),
            )
            if points[-1] != a:
                points.append(a)
            points.append(b)
        end_point = Point(start_x + direction_x * run, start_y + direction_y * run)
        if points[-1] != end_point:
            points.append(end_point)
        return points

    def _relief_points(self, panel: Panel, settings: BoxSettings) -> list[Point]:
        if settings.relief_diameter <= 0.0:
            return []
        tolerance = 0.001
        outline = (
            panel.outline[:-1]
            if panel.outline[-1] == panel.outline[0]
            else panel.outline
        )
        if len(outline) < 3:
            return []
        orientation = self._signed_area(outline)
        if abs(orientation) < tolerance:
            return []
        relief_points: list[Point] = []
        for index, point in enumerate(outline):
            previous_point = outline[index - 1]
            next_point = outline[(index + 1) % len(outline)]
            turn = self._turn_cross(previous_point, point, next_point)
            is_inside_corner = (
                turn < -tolerance if orientation > 0.0 else turn > tolerance
            )
            if is_inside_corner and point not in relief_points:
                relief_points.append(point)
        return relief_points

    def _signed_area(self, outline: list[Point]) -> float:
        area = 0.0
        for start, end in zip(outline, [*outline[1:], outline[0]]):
            area += start.x * end.y - end.x * start.y
        return area * 0.5

    def _turn_cross(
        self, previous_point: Point, point: Point, next_point: Point
    ) -> float:
        incoming_x = point.x - previous_point.x
        incoming_y = point.y - previous_point.y
        outgoing_x = next_point.x - point.x
        outgoing_y = next_point.y - point.y
        return incoming_x * outgoing_y - incoming_y * outgoing_x

    def _move_panel(self, panel: Panel, offset_x: float, offset_y: float) -> None:
        panel.outline = [
            Point(point.x + offset_x, point.y + offset_y) for point in panel.outline
        ]
        panel.relief_points = [
            Point(point.x + offset_x, point.y + offset_y)
            for point in panel.relief_points
        ]
        panel.origin_x += offset_x
        panel.origin_y += offset_y

    def _assign_stock(
        self,
        panel: Panel,
        stock_index: int,
        stock_origin_x: float,
        stock_origin_y: float,
    ) -> None:
        self._move_panel(panel, stock_origin_x, stock_origin_y)
        panel.stock_index = stock_index
        panel.stock_origin_x = stock_origin_x
        panel.stock_origin_y = stock_origin_y
