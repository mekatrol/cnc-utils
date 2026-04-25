from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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
from shapely.geometry import LineString, Polygon

from .board_bounds import BoardBounds
from .imported_drill_file import ImportedDrillFile
from .imported_gerber_file import ImportedGerberFile
from .nc_origin import format_origin_point
from .theme import AppTheme


class PcbPreviewWidget(QWidget):
    origin_selected = Signal(float, float)
    edge_polygon_selected = Signal(int, bool)
    alignment_hole_selected = Signal(int)
    alignment_hole_position_selected = Signal(float, float)

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
        self._origin_marker_label = "(0, 0)"
        self._origin_hotspot_points_override: dict[str, tuple[float, float]] = {}
        self._origin_selection_enabled = False
        self._origin_hotspots_visible = False
        self._hovered_origin_key: str | None = None
        self._cursor_world_position: tuple[float, float] | None = None
        self._validated_edge_path: Path | None = None
        self._validated_edge_polygons: list[list[tuple[float, float]]] = []
        self._edge_error_segments: list[
            tuple[tuple[float, float], tuple[float, float]]
        ] = []
        self._generated_edge_cut_paths: list[list[tuple[float, float]]] = []
        self._suppress_validated_edge_geometry = False
        self._edge_selection_enabled = False
        self._selected_edge_polygon_indices: set[int] = set()
        self._edge_polygon_modes: dict[int, str] = {}
        self._alignment_selection_bounds: tuple[float, float, float, float] | None = (
            None
        )
        self._alignment_selection_enabled = False
        self._selected_alignment_hole_index: int | None = None
        self._alignment_grid_spacing = 5.0
        self._alignment_hover_diameter = 1.0
        self._hovered_alignment_grid_point: tuple[float, float] | None = None

    def load_project_geometry(
        self,
        gerber_files: list[ImportedGerberFile],
        drill_files: list[ImportedDrillFile],
        alignment_holes: list[tuple[float, float, float]],
        *,
        reference_gerber_files: list[ImportedGerberFile] | None = None,
        reference_drill_files: list[ImportedDrillFile] | None = None,
        fit_view: bool = True,
    ) -> None:
        self._gerber_files = gerber_files
        self._drill_files = drill_files
        self._alignment_holes = alignment_holes
        self._rebuild_bounds(
            reference_gerber_files=reference_gerber_files,
            reference_drill_files=reference_drill_files,
        )
        if fit_view:
            self.fit_to_view()
        else:
            self.update()

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
        marker_label: str = "(0, 0)",
    ) -> None:
        self._origin_marker_bounds = board_bounds
        self._origin_marker_point = point
        self._origin_marker_label = marker_label
        self._origin_hotspot_points_override = dict(hotspot_points or {})
        self._origin_selection_enabled = selection_enabled
        if not selection_enabled:
            self._origin_hotspots_visible = False
            self._hovered_origin_key = None
        self._rebuild_bounds()
        if board_bounds is not None:
            self.fit_to_view()
        else:
            self.update()

    def set_edge_validation(
        self,
        edge_path: Path | None,
        *,
        polygons: list[list[tuple[float, float]]] | None = None,
        error_segments: list[tuple[tuple[float, float], tuple[float, float]]]
        | None = None,
        selection_enabled: bool = False,
        selected_polygon_indices: set[int] | None = None,
        polygon_modes: dict[int, str] | None = None,
        suppress_source_geometry: bool = False,
    ) -> None:
        self._validated_edge_path = None if edge_path is None else edge_path.resolve()
        self._validated_edge_polygons = list(polygons or [])
        self._edge_error_segments = list(error_segments or [])
        self._suppress_validated_edge_geometry = suppress_source_geometry
        self._edge_selection_enabled = selection_enabled
        self._selected_edge_polygon_indices = set(selected_polygon_indices or set())
        self._edge_polygon_modes = dict(polygon_modes or {})
        self.update()

    def set_edge_cut_preview_paths(
        self, paths: list[list[tuple[float, float]]] | None
    ) -> None:
        self._generated_edge_cut_paths = list(paths or [])
        self.update()

    def set_alignment_hole_selection(
        self,
        board_bounds: tuple[float, float, float, float] | None,
        *,
        selection_enabled: bool = False,
        selected_hole_index: int | None = None,
        grid_spacing: float = 5.0,
        hover_diameter: float = 1.0,
    ) -> None:
        self._alignment_selection_bounds = board_bounds
        self._alignment_selection_enabled = selection_enabled
        self._selected_alignment_hole_index = selected_hole_index
        self._alignment_grid_spacing = max(0.1, grid_spacing)
        self._alignment_hover_diameter = max(0.01, hover_diameter)
        if not selection_enabled:
            self._selected_alignment_hole_index = None
            self._hovered_alignment_grid_point = None
        self._rebuild_bounds()
        self.update()

    def _rebuild_bounds(
        self,
        *,
        reference_gerber_files: list[ImportedGerberFile] | None = None,
        reference_drill_files: list[ImportedDrillFile] | None = None,
    ) -> None:
        self._bounds = BoardBounds()
        bounds_gerbers = (
            self._gerber_files
            if reference_gerber_files is None
            else reference_gerber_files
        )
        bounds_drills = (
            self._drill_files
            if reference_drill_files is None
            else reference_drill_files
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
        if self._origin_marker_bounds is not None:
            x_min, x_max, y_min, y_max = self._origin_marker_bounds
            self._bounds.include_point(x_min, y_min)
            self._bounds.include_point(x_max, y_max)
        if (
            self._alignment_selection_enabled
            and self._alignment_selection_bounds is not None
        ):
            x_min, x_max, y_min, y_max = self._alignment_selection_bounds
            self._bounds.include_point(x_min, y_min)
            self._bounds.include_point(x_max, y_max)

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
            selected_polygon_index = self._edge_polygon_index_at_position(
                event.position()
            )
            if selected_polygon_index is not None:
                modifiers = event.modifiers()
                ctrl_pressed = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
                if ctrl_pressed:
                    if selected_polygon_index in self._selected_edge_polygon_indices:
                        self._selected_edge_polygon_indices.discard(
                            selected_polygon_index
                        )
                    else:
                        self._selected_edge_polygon_indices.add(selected_polygon_index)
                else:
                    self._selected_edge_polygon_indices = {selected_polygon_index}
                self.edge_polygon_selected.emit(selected_polygon_index, ctrl_pressed)
                self.update()
                return
            if self._edge_selection_enabled and self._selected_edge_polygon_indices:
                self._selected_edge_polygon_indices.clear()
                self.edge_polygon_selected.emit(-1, False)
                self.update()
                return
            selected_alignment_index = self._alignment_hole_index_at_position(
                event.position()
            )
            if selected_alignment_index is not None:
                self._selected_alignment_hole_index = selected_alignment_index
                self.alignment_hole_selected.emit(selected_alignment_index)
                self.update()
                return
            if self._alignment_selection_enabled:
                world_position = self._screen_to_world(event.position())
                if world_position is not None:
                    grid_position = self._alignment_grid_intersection(world_position)
                    if grid_position is not None:
                        self.alignment_hole_position_selected.emit(
                            grid_position[0], grid_position[1]
                        )
                        return
            self._dragging = True
            self._last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        cursor_world_position = self._screen_to_world(event.position())
        if cursor_world_position != self._cursor_world_position:
            self._cursor_world_position = cursor_world_position
            self.update()
        hovered_alignment_grid_point = (
            self._alignment_grid_intersection(cursor_world_position)
            if self._alignment_selection_enabled and cursor_world_position is not None
            else None
        )
        if hovered_alignment_grid_point != self._hovered_alignment_grid_point:
            self._hovered_alignment_grid_point = hovered_alignment_grid_point
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
        needs_update = self._hovered_alignment_grid_point is not None
        self._hovered_alignment_grid_point = None
        if self._origin_hotspots_visible or self._hovered_origin_key is not None:
            self._origin_hotspots_visible = False
            self._hovered_origin_key = None
            needs_update = True
        if needs_update:
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
        if (
            not self._bounds.is_empty
            and abs(self._pan_x) < 1e-6
            and abs(self._pan_y) < 1e-6
        ):
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
        spacing = self._nice_spacing(
            max(self._bounds.width, self._bounds.height) / 10.0
        )
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
        self._draw_origin_selection_bounds(painter)
        self._draw_alignment_selection_grid(painter)
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
        for index, hole in enumerate(self._alignment_holes):
            self._draw_alignment_hole(
                painter, hole, selected=index == self._selected_alignment_hole_index
            )

        self._draw_edge_polygon_annotations(painter)
        self._draw_generated_edge_cut_paths(painter)
        self._draw_alignment_hover_indicator(painter)
        self._draw_origin_hotspots(painter)
        self._draw_origin_marker(painter)

    def _draw_origin_selection_bounds(self, painter: QPainter) -> None:
        if self._origin_marker_bounds is None:
            return
        x_min, x_max, y_min, y_max = self._origin_marker_bounds
        painter.save()
        outline_color = QColor(self._theme.named_color("pcb_preview_outline"))
        outline_color.setAlpha(220)
        fill_color = QColor(self._theme.named_color("pcb_preview_selection"))
        fill_color.setAlpha(32)
        pen = QPen(outline_color, 2.2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(fill_color)
        top_left = self._world_to_screen(x_min, y_max)
        bottom_right = self._world_to_screen(x_max, y_min)
        painter.drawRect(
            QRectF(
                min(top_left.x(), bottom_right.x()),
                min(top_left.y(), bottom_right.y()),
                abs(bottom_right.x() - top_left.x()),
                abs(bottom_right.y() - top_left.y()),
            )
        )
        painter.restore()

    def _draw_alignment_selection_grid(self, painter: QPainter) -> None:
        if (
            not self._alignment_selection_enabled
            or self._alignment_selection_bounds is None
        ):
            return
        x_min, x_max, y_min, y_max = self._alignment_selection_bounds
        spacing = max(0.1, self._alignment_grid_spacing)
        painter.save()
        minor_color = QColor(self._theme.named_color("pcb_preview_grid_major"))
        minor_color.setAlpha(170)
        center_color = QColor(self._theme.named_color("pcb_preview_alignment"))
        center_color.setAlpha(210)
        pen = QPen(minor_color, 1.3)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        x_pos = x_min
        while x_pos <= x_max + (spacing * 0.5):
            self._draw_world_line(painter, (x_pos, y_min), (x_pos, y_max))
            x_pos += spacing
        y_pos = y_min
        while y_pos <= y_max + (spacing * 0.5):
            self._draw_world_line(painter, (x_min, y_pos), (x_max, y_pos))
            y_pos += spacing
        center_pen = QPen(center_color, 1.8)
        center_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(center_pen)
        x_mid = (x_min + x_max) * 0.5
        y_mid = (y_min + y_max) * 0.5
        self._draw_world_line(painter, (x_mid, y_min), (x_mid, y_max))
        self._draw_world_line(painter, (x_min, y_mid), (x_max, y_mid))
        painter.restore()

    def _draw_alignment_hover_indicator(self, painter: QPainter) -> None:
        if (
            not self._alignment_selection_enabled
            or self._hovered_alignment_grid_point is None
        ):
            return
        screen_center = self._world_to_screen(
            self._hovered_alignment_grid_point[0], self._hovered_alignment_grid_point[1]
        )
        radius = max(2.0, self._alignment_hover_diameter * 0.5 * self._zoom)
        painter.save()
        fill = QColor(self._theme.named_color("pcb_preview_alignment_hover"))
        fill.setAlpha(55)
        stroke = QColor(self._theme.named_color("pcb_preview_alignment_hover"))
        stroke.setAlpha(235)
        painter.setBrush(fill)
        painter.setPen(QPen(stroke, 2.2))
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

    def _draw_gerber(
        self,
        painter: QPainter,
        gerber: ImportedGerberFile,
        color: QColor,
        *,
        mirrored: bool,
    ) -> None:
        if (
            gerber.path == self._validated_edge_path
            and self._suppress_validated_edge_geometry
        ):
            self._draw_edge_errors(painter, gerber, mirrored=mirrored)
            return
        painter.setPen(QPen(self._theme.named_color("pcb_preview_outline"), 2.2))
        outline_polygons = self._outline_polygons_for(gerber)
        if outline_polygons:
            for polygon in outline_polygons:
                self._draw_outline(
                    painter, self._transform_polygon(gerber, polygon, mirrored=mirrored)
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

    def _draw_outline(
        self, painter: QPainter, outline: list[tuple[float, float]]
    ) -> None:
        if len(outline) < 2:
            return
        for start, end in zip(outline, outline[1:]):
            self._draw_world_line(painter, start, end)

    def _draw_polygon(
        self, painter: QPainter, polygon: list[tuple[float, float]], fill_color: QColor
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
        self, painter: QPainter, hole: tuple[float, float, float], *, selected: bool
    ) -> None:
        screen_center = self._world_to_screen(hole[0], hole[1])
        radius = max(2.0, hole[2] * 0.5 * self._zoom)
        painter.save()
        if selected:
            highlight = QColor(self._theme.named_color("pcb_preview_selection"))
            highlight.setAlpha(70)
            painter.setBrush(highlight)
            painter.setPen(QPen(self._theme.named_color("pcb_preview_selection"), 2.8))
            painter.drawEllipse(screen_center, radius + 6.0, radius + 6.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(self._theme.named_color("pcb_preview_alignment"), 1.8))
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
        self, painter: QPainter, gerber: ImportedGerberFile, *, mirrored: bool
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

    def _draw_generated_edge_cut_paths(self, painter: QPainter) -> None:
        if not self._generated_edge_cut_paths:
            return
        path_color = QColor(self._theme.named_color("pcb_preview_error"))
        painter.save()
        pen = QPen(path_color, 3.4)
        pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        for path in self._generated_edge_cut_paths:
            self._draw_outline(painter, path)
        painter.restore()

    def _draw_edge_polygon_annotations(self, painter: QPainter) -> None:
        if not self._edge_selection_enabled or not self._validated_edge_polygons:
            return
        painter.save()
        for index, polygon in enumerate(self._validated_edge_polygons):
            if index in self._selected_edge_polygon_indices:
                selection_color = QColor(
                    self._theme.named_color("pcb_preview_selection")
                )
                selection_color.setAlpha(255)
                painter.setPen(QPen(selection_color, 4.0))
                self._draw_outline(painter, polygon)
            mode = self._edge_polygon_modes.get(index, "none").strip()
            if mode == "none":
                continue
            indicator_polygon = self._offset_polygon_for_mode(polygon, mode)
            if not indicator_polygon:
                continue
            indicator_color = QColor(self._theme.named_color("pcb_preview_alignment"))
            indicator_color.setAlpha(220)
            pen = QPen(indicator_color, 2.0)
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            self._draw_outline(painter, indicator_polygon)
        painter.restore()

    def _offset_polygon_for_mode(
        self, polygon: list[tuple[float, float]], mode: str
    ) -> list[tuple[float, float]]:
        if len(polygon) < 2:
            return []
        if mode == "on_contour":
            return list(polygon)
        offset_world = 8.0 / max(self._zoom, 0.001)
        shape = Polygon(polygon)
        if shape.is_empty or not shape.is_valid or shape.area <= 0.0:
            return []
        offset_shape = shape.buffer(
            offset_world if mode == "outside_profile" else -offset_world
        )
        if offset_shape.is_empty:
            return []
        if hasattr(offset_shape, "geoms"):
            offset_shape = max(offset_shape.geoms, key=lambda item: item.area)
        boundary = offset_shape.boundary
        if not isinstance(boundary, LineString):
            return []
        return [(float(x_pos), float(y_pos)) for x_pos, y_pos in boundary.coords]

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
            QPointF(
                screen_center.x() + radius + 10.0, screen_center.y() - radius - 4.0
            ),
            self._origin_marker_label,
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
        self, painter: QPainter, start: tuple[float, float], end: tuple[float, float]
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

    def _edge_polygon_index_at_position(self, position: QPointF) -> int | None:
        if not self._edge_selection_enabled or not self._validated_edge_polygons:
            return None
        world_position = self._screen_to_world(position)
        if world_position is None:
            return None
        max_distance = 18.0 / max(self._zoom, 0.001)
        closest_index = None
        closest_distance = None
        for index, polygon in enumerate(self._validated_edge_polygons):
            distance = self._distance_to_polygon_boundary(world_position, polygon)
            if distance > max_distance:
                continue
            if closest_distance is None or distance < closest_distance:
                closest_index = index
                closest_distance = distance
        return closest_index

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

    def _alignment_hole_index_at_position(self, position: QPointF) -> int | None:
        if not self._alignment_selection_enabled or not self._alignment_holes:
            return None
        closest_index = None
        closest_distance = None
        for index, hole in enumerate(self._alignment_holes):
            screen_center = self._world_to_screen(hole[0], hole[1])
            hole_radius = max(8.0, (hole[2] * 0.5 * self._zoom) + 8.0)
            distance = self._distance(position, screen_center)
            if distance > hole_radius:
                continue
            if closest_distance is None or distance < closest_distance:
                closest_index = index
                closest_distance = distance
        return closest_index

    def _alignment_grid_intersection(
        self, position: tuple[float, float]
    ) -> tuple[float, float] | None:
        if self._alignment_selection_bounds is None:
            return position
        x_min, x_max, y_min, y_max = self._alignment_selection_bounds
        if not (x_min <= position[0] <= x_max and y_min <= position[1] <= y_max):
            return None
        spacing = max(0.1, self._alignment_grid_spacing)
        x_pos = x_min + round((position[0] - x_min) / spacing) * spacing
        y_pos = y_min + round((position[1] - y_min) / spacing) * spacing
        return (min(max(x_pos, x_min), x_max), min(max(y_pos, y_min), y_max))

    def _distance(self, first: QPointF, second: QPointF) -> float:
        delta_x = first.x() - second.x()
        delta_y = first.y() - second.y()
        return math.hypot(delta_x, delta_y)

    def _distance_to_polygon_boundary(
        self, point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> float:
        if len(polygon) < 2:
            return float("inf")
        return min(
            self._distance_to_segment(point, start, end)
            for start, end in zip(polygon, polygon[1:])
        )

    def _distance_to_segment(
        self,
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        start_x, start_y = start
        end_x, end_y = end
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        if abs(delta_x) < 1e-9 and abs(delta_y) < 1e-9:
            return math.hypot(point[0] - start_x, point[1] - start_y)
        fraction = (
            ((point[0] - start_x) * delta_x) + ((point[1] - start_y) * delta_y)
        ) / ((delta_x * delta_x) + (delta_y * delta_y))
        fraction = max(0.0, min(1.0, fraction))
        nearest_x = start_x + (fraction * delta_x)
        nearest_y = start_y + (fraction * delta_y)
        return math.hypot(point[0] - nearest_x, point[1] - nearest_y)

    def _point_in_polygon(
        self, point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        if len(polygon) < 4:
            return False
        inside = False
        point_x, point_y = point
        for start, end in zip(polygon, polygon[1:]):
            x1, y1 = start
            x2, y2 = end
            intersects = ((y1 > point_y) != (y2 > point_y)) and (
                point_x < (((x2 - x1) * (point_y - y1)) / ((y2 - y1) or 1e-12)) + x1
            )
            if intersects:
                inside = not inside
        return inside

    def _polygon_area(self, polygon: list[tuple[float, float]]) -> float:
        if len(polygon) < 4:
            return 0.0
        area = 0.0
        for start, end in zip(polygon, polygon[1:]):
            area += (start[0] * end[1]) - (end[0] * start[1])
        return area * 0.5

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
        world_x = (
            (position.x() - (self.width() * 0.5) - self._pan_x) / self._zoom
        ) + center_x
        world_y = center_y - (
            (position.y() - (self.height() * 0.5) - self._pan_y) / self._zoom
        )
        return world_x, world_y

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.setPen(self._theme.named_color("pcb_preview_text"))
        painter.drawText(16, 28, "PCB Preview")
        if self._bounds.is_empty:
            painter.drawText(
                16, 50, "Import Gerber and drill files to preview the board."
            )
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
        if self._mirror_preview_mode == "overlay" or not self._should_mirror_gerber(
            gerber
        ):
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
                    transformed_center[0], transformed_center[1], radius
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
        self, start: tuple[float, float], end: tuple[float, float], margin: float
    ) -> None:
        self._bounds.include_point(start[0], start[1], margin)
        self._bounds.include_point(end[0], end[1], margin)

    def _include_rect_bounds(
        self, center: tuple[float, float], half_width: float, half_height: float
    ) -> None:
        self._bounds.include_point(center[0] - half_width, center[1] - half_height)
        self._bounds.include_point(center[0] + half_width, center[1] + half_height)

    def _transform_polygon(
        self,
        gerber: ImportedGerberFile,
        polygon: list[tuple[float, float]],
        mirrored: bool = False,
    ) -> list[tuple[float, float]]:
        return [
            self._transform_point(gerber, point, mirrored=mirrored) for point in polygon
        ]

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

    def _outline_polygons_for(
        self, gerber: ImportedGerberFile
    ) -> list[list[tuple[float, float]]]:
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
        fraction = value / (10**exponent)
        if fraction < 1.5:
            nice = 1.0
        elif fraction < 3.5:
            nice = 2.0
        elif fraction < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10**exponent)
