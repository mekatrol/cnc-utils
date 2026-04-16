from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from .board_bounds import BoardBounds
from .imported_drill_file import ImportedDrillFile
from .imported_gerber_file import ImportedGerberFile
from .nc_origin import format_origin_point
from .theme import AppTheme


class PcbPreviewWidget(QWidget):
    origin_selected = Signal(float, float)

    def __init__(self, theme: AppTheme, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setMinimumSize(720, 480)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._gerber_files: list[ImportedGerberFile] = []
        self._drill_files: list[ImportedDrillFile] = []
        self._alignment_holes: list[tuple[float, float, float]] = []
        self._bounds = BoardBounds()
        self._mirror_axis_bounds: tuple[float, float, float, float] | None = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._last_pos = QPointF()
        self._dragging = False
        self._back_copper_path = None
        self._edges_path = None
        self._mirror_edge = ""
        self._mirror_preview_mode = "side_by_side"
        self._origin_marker_bounds: tuple[float, float, float, float] | None = None
        self._origin_marker_point: tuple[float, float] | None = None
        self._origin_hotspot_points_override: dict[str, tuple[float, float]] = {}
        self._origin_selection_enabled = False
        self._origin_hotspots_visible = False
        self._hovered_origin_key: str | None = None
        self._cursor_world_position: tuple[float, float] | None = None
        self._validated_edge_path: Path | None = None
        self._validated_edge_polygons: list[list[tuple[float, float]]] = []
        self._edge_error_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def load_project_geometry(
        self,
        gerber_files: list[ImportedGerberFile],
        drill_files: list[ImportedDrillFile],
        alignment_holes: list[tuple[float, float, float]],
        *,
        reference_gerber_files: list[ImportedGerberFile] | None = None,
        reference_drill_files: list[ImportedDrillFile] | None = None,
    ) -> None:
        self._gerber_files = gerber_files
        self._drill_files = drill_files
        self._alignment_holes = alignment_holes
        self._rebuild_bounds(
            reference_gerber_files=reference_gerber_files,
            reference_drill_files=reference_drill_files,
        )
        self.fit_to_view()

    def set_mirror_setup(
        self,
        *,
        back_copper_path: Path | None,
        edges_path: Path | None,
        board_bounds: tuple[float, float, float, float] | None,
        mirror_edge: str,
        preview_mode: str,
    ) -> None:
        self._back_copper_path = back_copper_path
        self._edges_path = edges_path
        self._mirror_axis_bounds = board_bounds
        self._mirror_edge = mirror_edge.strip()
        self._mirror_preview_mode = preview_mode.strip() or "side_by_side"
        self._rebuild_bounds()
        self.fit_to_view()

    def set_origin_marker(
        self,
        board_bounds: tuple[float, float, float, float] | None,
        point: tuple[float, float] | None,
        *,
        hotspot_points: dict[str, tuple[float, float]] | None = None,
        selection_enabled: bool = False,
    ) -> None:
        self._origin_marker_bounds = board_bounds
        self._origin_marker_point = point
        self._origin_hotspot_points_override = dict(hotspot_points or {})
        self._origin_selection_enabled = selection_enabled
        if not selection_enabled:
            self._origin_hotspots_visible = False
            self._hovered_origin_key = None
        self.update()

    def set_edge_validation(
        self,
        edge_path: Path | None,
        *,
        polygons: list[list[tuple[float, float]]] | None = None,
        error_segments: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    ) -> None:
        self._validated_edge_path = None if edge_path is None else edge_path.resolve()
        self._validated_edge_polygons = list(polygons or [])
        self._edge_error_segments = list(error_segments or [])
        self.update()

    def _rebuild_bounds(
        self,
        *,
        reference_gerber_files: list[ImportedGerberFile] | None = None,
        reference_drill_files: list[ImportedDrillFile] | None = None,
    ) -> None:
        self._bounds = BoardBounds()
        bounds_gerbers = (
            self._gerber_files if reference_gerber_files is None else reference_gerber_files
        )
        bounds_drills = (
            self._drill_files if reference_drill_files is None else reference_drill_files
        )
        gerber_bounds = BoardBounds()
        for gerber in bounds_gerbers:
            gerber_bounds.include_bounds(gerber.bounds)
            self._include_gerber_bounds(gerber)
        if self._mirror_preview_mode != "overlay":
            self._include_mirrored_panel_bounds(gerber_bounds)
        for drill in bounds_drills:
            self._bounds.include_bounds(drill.bounds)
        for x, y, diameter in self._alignment_holes:
            self._bounds.include_point(x, y, diameter * 0.5)

    def fit_to_view(self) -> None:
        self._pan_x = 0.0
        self._pan_y = 0.0
        if self._bounds.is_empty:
            self._zoom = 1.0
            self.update()
            return
        span_x = max(self._bounds.width, 10.0)
        span_y = max(self._bounds.height, 10.0)
        usable_width = max(self.width() - 120, 1)
        usable_height = max(self.height() - 120, 1)
        self._zoom = min(usable_width / span_x, usable_height / span_y)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            selected_key = self._origin_at_position(event.position())
            if selected_key is not None:
                selected_point = self._origin_hotspot_points().get(selected_key)
                if selected_point is None:
                    return
                self._origin_marker_point = selected_point
                self._hovered_origin_key = selected_key
                self._origin_hotspots_visible = True
                self.origin_selected.emit(selected_point[0], selected_point[1])
                self.update()
                return
            self._dragging = True
            self._last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cursor_world_position = self._screen_to_world(event.position())
        if cursor_world_position != self._cursor_world_position:
            self._cursor_world_position = cursor_world_position
            self.update()
        if self._origin_selection_enabled and self._origin_marker_bounds is not None:
            hovered_origin = self._origin_at_position(event.position())
            cursor_within_board = self._screen_position_within_board(event.position())
            hotspots_visible = cursor_within_board or hovered_origin is not None
            if (
                hovered_origin != self._hovered_origin_key
                or hotspots_visible != self._origin_hotspots_visible
            ):
                self._hovered_origin_key = hovered_origin
                self._origin_hotspots_visible = hotspots_visible
                self.update()
        if not self._dragging:
            return
        delta = event.position() - self._last_pos
        self._last_pos = event.position()
        self._pan_x += delta.x()
        self._pan_y += delta.y()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._cursor_world_position = None
        if self._origin_hotspots_visible or self._hovered_origin_key is not None:
            self._origin_hotspots_visible = False
            self._hovered_origin_key = None
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        cursor_position = event.position()
        anchor_world_position = self._screen_to_world(cursor_position)
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        self._zoom = max(0.1, min(1000.0, self._zoom * factor))
        if anchor_world_position is not None:
            anchor_screen_position = self._world_to_screen(
                anchor_world_position[0], anchor_world_position[1]
            )
            self._pan_x += cursor_position.x() - anchor_screen_position.x()
            self._pan_y += cursor_position.y() - anchor_screen_position.y()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._bounds.is_empty and abs(self._pan_x) < 1e-6 and abs(self._pan_y) < 1e-6:
            self.fit_to_view()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self._theme.named_color("pcb_preview_background"))
        self._draw_grid(painter)
        self._draw_geometry(painter)
        self._draw_overlay(painter)
        painter.end()

    def _draw_grid(self, painter: QPainter) -> None:
        if self._bounds.is_empty:
            return
        spacing = self._nice_spacing(max(self._bounds.width, self._bounds.height) / 10.0)
        x = math.floor(self._bounds.x_min / spacing) * spacing
        while x <= self._bounds.x_max + spacing:
            pen = QPen(
                self._theme.named_color("pcb_preview_grid_major")
                if abs(round(x / spacing)) % 5 == 0
                else self._theme.named_color("pcb_preview_grid_minor"),
                1,
            )
            painter.setPen(pen)
            self._draw_world_line(
                painter,
                (x, self._bounds.y_min - spacing),
                (x, self._bounds.y_max + spacing),
            )
            x += spacing
        y = math.floor(self._bounds.y_min / spacing) * spacing
        while y <= self._bounds.y_max + spacing:
            pen = QPen(
                self._theme.named_color("pcb_preview_grid_major")
                if abs(round(y / spacing)) % 5 == 0
                else self._theme.named_color("pcb_preview_grid_minor"),
                1,
            )
            painter.setPen(pen)
            self._draw_world_line(
                painter,
                (self._bounds.x_min - spacing, y),
                (self._bounds.x_max + spacing, y),
            )
            y += spacing

    def _draw_geometry(self, painter: QPainter) -> None:
        for index, gerber in enumerate(self._gerber_files):
            palette = self._theme.gerber_palette()
            color = palette[index % len(palette)]
            self._draw_gerber(painter, gerber, color, mirrored=False)
            if self._should_duplicate_edges_gerber(gerber):
                self._draw_gerber(painter, gerber, color, mirrored=True)

        painter.setPen(QPen(self._theme.named_color("pcb_preview_drill"), 1.3))
        for drill in self._drill_files:
            for hole in drill.holes:
                self._draw_hole(painter, hole)

        painter.setPen(QPen(self._theme.named_color("pcb_preview_alignment"), 1.8))
        for hole in self._alignment_holes:
            self._draw_alignment_hole(painter, hole)

        self._draw_origin_hotspots(painter)
        self._draw_origin_marker(painter)

    def _draw_gerber(
        self,
        painter: QPainter,
        gerber: ImportedGerberFile,
        color: QColor,
        *,
        mirrored: bool,
    ) -> None:
        painter.setPen(QPen(self._theme.named_color("pcb_preview_outline"), 2.2))
        outline_polygons = self._outline_polygons_for(gerber)
        if outline_polygons:
            for polygon in outline_polygons:
                self._draw_outline(
                    painter,
                    self._transform_polygon(gerber, polygon, mirrored=mirrored),
                )
        elif gerber.segments:
            for start, end in gerber.segments:
                self._draw_world_line(
                    painter,
                    self._transform_point(gerber, start, mirrored=mirrored),
                    self._transform_point(gerber, end, mirrored=mirrored),
                )

        fill_color = QColor(color)
        fill_color.setAlpha(55)
        stroke_color = QColor(color)
        stroke_color.setAlpha(190)
        painter.setPen(QPen(stroke_color, 1.4))
        for region in gerber.regions:
            self._draw_polygon(
                painter,
                self._transform_polygon(gerber, region, mirrored=mirrored),
                fill_color,
            )
        for trace_start, trace_end, width in gerber.traces:
            pen = QPen(stroke_color, max(1.2, width * self._zoom))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            self._draw_world_line(
                painter,
                self._transform_point(gerber, trace_start, mirrored=mirrored),
                self._transform_point(gerber, trace_end, mirrored=mirrored),
            )
        painter.setPen(QPen(stroke_color, 1.2))
        for center, aperture in gerber.pads:
            self._draw_pad(
                painter,
                self._transform_point(gerber, center, mirrored=mirrored),
                aperture,
                fill_color,
                stroke_color,
            )
        self._draw_edge_errors(painter, gerber, mirrored=mirrored)

    def _draw_outline(self, painter: QPainter, outline: list[tuple[float, float]]) -> None:
        if len(outline) < 2:
            return
        for start, end in zip(outline, outline[1:]):
            self._draw_world_line(painter, start, end)

    def _draw_polygon(
        self,
        painter: QPainter,
        polygon: list[tuple[float, float]],
        fill_color: QColor,
    ) -> None:
        if len(polygon) < 3:
            return
        painter.save()
        painter.setBrush(fill_color)
        # Gerber regions can contain cut-in edges that are needed to define pours
        # but should not be shown as visible copper boundaries.
        painter.setPen(Qt.PenStyle.NoPen)
        points = QPolygonF([self._world_to_screen(x, y) for x, y in polygon])
        painter.drawPolygon(points)
        painter.restore()

    def _draw_pad(
        self,
        painter: QPainter,
        center: tuple[float, float],
        aperture: dict[str, float | str],
        fill_color: QColor,
        stroke_color: QColor,
    ) -> None:
        painter.save()
        painter.setBrush(fill_color)
        painter.setPen(QPen(stroke_color, 1.1))
        screen_center = self._world_to_screen(center[0], center[1])
        if aperture["type"] == "circle":
            radius = float(aperture["diameter"]) * 0.5 * self._zoom
            painter.drawEllipse(screen_center, radius, radius)
        else:
            width = float(aperture["width"]) * self._zoom
            height = float(aperture["height"]) * self._zoom
            painter.drawRect(
                screen_center.x() - (width * 0.5),
                screen_center.y() - (height * 0.5),
                width,
                height,
            )
        painter.restore()

    def _draw_hole(self, painter: QPainter, hole: tuple[float, float, float]) -> None:
        screen_center = self._world_to_screen(hole[0], hole[1])
        radius = max(1.5, hole[2] * 0.5 * self._zoom)
        painter.save()
        painter.setBrush(self._theme.named_color("pcb_preview_background"))
        painter.drawEllipse(screen_center, radius, radius)
        painter.restore()

    def _draw_alignment_hole(
        self,
        painter: QPainter,
        hole: tuple[float, float, float],
    ) -> None:
        screen_center = self._world_to_screen(hole[0], hole[1])
        radius = max(2.0, hole[2] * 0.5 * self._zoom)
        painter.save()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(screen_center, radius, radius)
        painter.drawLine(
            QPointF(screen_center.x() - radius - 5.0, screen_center.y()),
            QPointF(screen_center.x() + radius + 5.0, screen_center.y()),
        )
        painter.drawLine(
            QPointF(screen_center.x(), screen_center.y() - radius - 5.0),
            QPointF(screen_center.x(), screen_center.y() + radius + 5.0),
        )
        painter.restore()

    def _draw_edge_errors(
        self,
        painter: QPainter,
        gerber: ImportedGerberFile,
        *,
        mirrored: bool,
    ) -> None:
        if gerber.path != self._validated_edge_path or not self._edge_error_segments:
            return
        error_color = self._theme.named_color("pcb_preview_error")
        painter.save()
        painter.setPen(QPen(error_color, 3.0))
        for start, end in self._edge_error_segments:
            self._draw_world_line(
                painter,
                self._transform_point(gerber, start, mirrored=mirrored),
                self._transform_point(gerber, end, mirrored=mirrored),
            )
        painter.restore()

    def _draw_origin_marker(self, painter: QPainter) -> None:
        if self._origin_marker_point is None:
            return
        x_pos, y_pos = self._origin_marker_point
        screen_center = self._world_to_screen(x_pos, y_pos)
        radius = 10.0
        painter.save()
        pen = QPen(self._theme.named_color("pcb_preview_alignment"), 2.0)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(screen_center, radius, radius)
        painter.drawLine(
            QPointF(screen_center.x() - (radius + 6.0), screen_center.y()),
            QPointF(screen_center.x() + (radius + 6.0), screen_center.y()),
        )
        painter.drawLine(
            QPointF(screen_center.x(), screen_center.y() - (radius + 6.0)),
            QPointF(screen_center.x(), screen_center.y() + (radius + 6.0)),
        )
        painter.drawText(
            QPointF(screen_center.x() + radius + 10.0, screen_center.y() - radius - 4.0),
            "(0, 0)",
        )
        painter.restore()

    def _draw_origin_hotspots(self, painter: QPainter) -> None:
        if (
            not self._origin_selection_enabled
            or not self._origin_hotspots_visible
            or self._origin_marker_bounds is None
        ):
            return
        for location, screen_center in self._origin_hotspot_screen_points().items():
            point = self._origin_hotspot_points().get(location)
            is_selected = point == self._origin_marker_point
            is_hovered = location == self._hovered_origin_key
            radius = 8.0 if is_hovered else 6.0
            painter.save()
            pen = QPen(self._theme.named_color("pcb_preview_alignment"), 2.0)
            painter.setPen(pen)
            fill = QColor(self._theme.named_color("pcb_preview_alignment"))
            fill.setAlpha(220 if (is_selected or is_hovered) else 90)
            painter.setBrush(fill)
            painter.drawEllipse(screen_center, radius, radius)
            if is_hovered:
                painter.drawText(
                    QPointF(screen_center.x() + 10.0, screen_center.y() - 10.0),
                    format_origin_point(point),
                )
            painter.restore()

    def _draw_world_line(
        self,
        painter: QPainter,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        painter.drawLine(
            self._world_to_screen(start[0], start[1]),
            self._world_to_screen(end[0], end[1]),
        )

    def _origin_hotspot_points(self) -> dict[str, tuple[float, float]]:
        if self._origin_marker_bounds is None:
            return {}
        return dict(self._origin_hotspot_points_override)

    def _origin_hotspot_screen_points(self) -> dict[str, QPointF]:
        return {
            location: self._world_to_screen(point[0], point[1])
            for location, point in self._origin_hotspot_points().items()
        }

    def _origin_at_position(self, position: QPointF) -> str | None:
        if not self._origin_selection_enabled or self._origin_marker_bounds is None:
            return None
        for location, screen_center in self._origin_hotspot_screen_points().items():
            if self._distance(position, screen_center) <= 14.0:
                return location
        return None

    def _screen_position_within_board(self, position: QPointF) -> bool:
        if self._origin_marker_bounds is None:
            return False
        x_min, x_max, y_min, y_max = self._origin_marker_bounds
        left = self._world_to_screen(x_min, y_min).x()
        right = self._world_to_screen(x_max, y_min).x()
        top = self._world_to_screen(x_min, y_max).y()
        bottom = self._world_to_screen(x_min, y_min).y()
        min_x = min(left, right)
        max_x = max(left, right)
        min_y = min(top, bottom)
        max_y = max(top, bottom)
        return min_x <= position.x() <= max_x and min_y <= position.y() <= max_y

    def _distance(self, first: QPointF, second: QPointF) -> float:
        delta_x = first.x() - second.x()
        delta_y = first.y() - second.y()
        return math.hypot(delta_x, delta_y)

    def _world_to_screen(self, x: float, y: float) -> QPointF:
        center_x = self._bounds.center_x
        center_y = self._bounds.center_y
        screen_x = (x - center_x) * self._zoom + (self.width() * 0.5) + self._pan_x
        screen_y = (center_y - y) * self._zoom + (self.height() * 0.5) + self._pan_y
        return QPointF(screen_x, screen_y)

    def _screen_to_world(self, position: QPointF) -> tuple[float, float] | None:
        if self._bounds.is_empty or self._zoom == 0:
            return None
        center_x = self._bounds.center_x
        center_y = self._bounds.center_y
        world_x = ((position.x() - (self.width() * 0.5) - self._pan_x) / self._zoom) + center_x
        world_y = center_y - ((position.y() - (self.height() * 0.5) - self._pan_y) / self._zoom)
        return world_x, world_y

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.setPen(self._theme.named_color("pcb_preview_text"))
        painter.drawText(16, 28, "PCB Preview")
        if self._bounds.is_empty:
            painter.drawText(16, 50, "Import Gerber and drill files to preview the board.")
            painter.drawText(16, 72, "Left drag: pan   Wheel: zoom")
            return
        painter.drawText(
            16,
            50,
            "Gerber files: "
            f"{len(self._gerber_files)}   Drill files: {len(self._drill_files)}   "
            f"Alignment holes: {len(self._alignment_holes)}",
        )
        painter.drawText(
            16,
            72,
            f"Bounds: X {self._bounds.x_min:.2f}..{self._bounds.x_max:.2f} mm   "
            f"Y {self._bounds.y_min:.2f}..{self._bounds.y_max:.2f} mm",
        )
        if self._cursor_world_position is not None:
            painter.drawText(
                16,
                94,
                f"Cursor: X {self._cursor_world_position[0]:.3f} mm   "
                f"Y {self._cursor_world_position[1]:.3f} mm",
            )

    def _include_gerber_bounds(self, gerber: ImportedGerberFile) -> None:
        if self._should_duplicate_edges_gerber(gerber):
            self._bounds.include_bounds(gerber.bounds)
            if self._mirror_preview_mode != "overlay":
                self._include_mirrored_gerber_bounds(gerber)
            return
        if self._mirror_preview_mode == "overlay" or not self._should_mirror_gerber(gerber):
            self._bounds.include_bounds(gerber.bounds)
            return
        self._include_mirrored_gerber_bounds(gerber)

    def _include_mirrored_gerber_bounds(self, gerber: ImportedGerberFile) -> None:
        for start, end, width in gerber.traces:
            self._include_segment_bounds(
                self._transform_point(gerber, start, mirrored=True),
                self._transform_point(gerber, end, mirrored=True),
                width * 0.5,
            )
        for center, aperture in gerber.pads:
            transformed_center = self._transform_point(gerber, center, mirrored=True)
            if aperture["type"] == "circle":
                radius = float(aperture["diameter"]) * 0.5
                self._bounds.include_point(
                    transformed_center[0],
                    transformed_center[1],
                    radius,
                )
            else:
                half_width = float(aperture["width"]) * 0.5
                half_height = float(aperture["height"]) * 0.5
                self._include_rect_bounds(transformed_center, half_width, half_height)
        for region in gerber.regions:
            for point in self._transform_polygon(gerber, region, mirrored=True):
                self._bounds.include_point(point[0], point[1])
        outline_polygons = self._outline_polygons_for(gerber)
        if outline_polygons:
            for polygon in outline_polygons:
                for point in self._transform_polygon(gerber, polygon, mirrored=True):
                    self._bounds.include_point(point[0], point[1])
        else:
            for start, end in gerber.segments:
                transformed_start = self._transform_point(gerber, start, mirrored=True)
                transformed_end = self._transform_point(gerber, end, mirrored=True)
                self._bounds.include_point(transformed_start[0], transformed_start[1])
                self._bounds.include_point(transformed_end[0], transformed_end[1])

    def _include_mirrored_panel_bounds(self, source_bounds: BoardBounds) -> None:
        if (
            source_bounds.is_empty
            or self._mirror_axis_bounds is None
            or not self._mirror_edge
            or (self._back_copper_path is None and self._edges_path is None)
        ):
            return
        corners = [
            (source_bounds.x_min, source_bounds.y_min),
            (source_bounds.x_min, source_bounds.y_max),
            (source_bounds.x_max, source_bounds.y_min),
            (source_bounds.x_max, source_bounds.y_max),
        ]
        for point in corners:
            mirrored_point = self._mirror_point(point)
            self._bounds.include_point(mirrored_point[0], mirrored_point[1])

    def _include_segment_bounds(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        margin: float,
    ) -> None:
        self._bounds.include_point(start[0], start[1], margin)
        self._bounds.include_point(end[0], end[1], margin)

    def _include_rect_bounds(
        self,
        center: tuple[float, float],
        half_width: float,
        half_height: float,
    ) -> None:
        self._bounds.include_point(center[0] - half_width, center[1] - half_height)
        self._bounds.include_point(center[0] + half_width, center[1] + half_height)

    def _transform_polygon(
        self,
        gerber: ImportedGerberFile,
        polygon: list[tuple[float, float]],
        mirrored: bool = False,
    ) -> list[tuple[float, float]]:
        return [self._transform_point(gerber, point, mirrored=mirrored) for point in polygon]

    def _transform_point(
        self,
        gerber: ImportedGerberFile,
        point: tuple[float, float],
        mirrored: bool = False,
    ) -> tuple[float, float]:
        if not mirrored:
            if self._should_duplicate_edges_gerber(gerber):
                return point
            if not self._should_mirror_gerber(gerber):
                return point
        elif not self._should_mirror_gerber(gerber):
            return point
        mirrored_point = self._mirror_point(point)
        if self._mirror_preview_mode != "overlay":
            return mirrored_point
        return self._overlay_point(mirrored_point)

    def _should_mirror_gerber(self, gerber: ImportedGerberFile) -> bool:
        if (
            (self._back_copper_path is None and self._edges_path is None)
            or not self._mirror_edge
            or self._mirror_axis_bounds is None
        ):
            return False
        return gerber.path in {self._back_copper_path, self._edges_path}

    def _should_duplicate_edges_gerber(self, gerber: ImportedGerberFile) -> bool:
        return (
            self._edges_path is not None
            and gerber.path == self._edges_path
            and self._should_mirror_gerber(gerber)
        )

    def _outline_polygons_for(self, gerber: ImportedGerberFile) -> list[list[tuple[float, float]]]:
        if gerber.path == self._validated_edge_path and self._validated_edge_polygons:
            return self._validated_edge_polygons
        if gerber.outline:
            return [gerber.outline]
        return []


    def _mirror_point(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        x_min, x_max, y_min, y_max = self._mirror_axis_bounds or (0.0, 0.0, 0.0, 0.0)
        if self._mirror_edge == "left":
            return ((2.0 * x_min) - x, y)
        if self._mirror_edge == "right":
            return ((2.0 * x_max) - x, y)
        if self._mirror_edge == "top":
            return (x, (2.0 * y_max) - y)
        if self._mirror_edge == "bottom":
            return (x, (2.0 * y_min) - y)
        return point

    def _overlay_point(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        x_min, x_max, y_min, y_max = self._mirror_axis_bounds or (0.0, 0.0, 0.0, 0.0)
        width = x_max - x_min
        height = y_max - y_min
        if self._mirror_edge == "left":
            return (x + width, y)
        if self._mirror_edge == "right":
            return (x - width, y)
        if self._mirror_edge == "top":
            return (x, y - height)
        if self._mirror_edge == "bottom":
            return (x, y + height)
        return point

    def _nice_spacing(self, value: float) -> float:
        if value <= 0:
            return 1.0
        exponent = math.floor(math.log10(value))
        fraction = value / (10 ** exponent)
        if fraction < 1.5:
            nice = 1.0
        elif fraction < 3.5:
            nice = 2.0
        elif fraction < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10 ** exponent)
