from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
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

BACKGROUND = QColor("#11151c")
GRID_MINOR = QColor("#1e2630")
GRID_MAJOR = QColor("#2d3947")
OUTLINE = QColor("#ffe066")
DRILL = QColor("#dfe7ef")
ALIGNMENT = QColor("#6ee7b7")
TEXT = QColor("#dfe7ef")
PALETTE = [
    QColor("#ff7f50"),
    QColor("#5dd39e"),
    QColor("#7aa2ff"),
    QColor("#f28482"),
    QColor("#ffd166"),
    QColor("#a78bfa"),
]


class PcbPreviewWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 480)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._gerber_files: list[ImportedGerberFile] = []
        self._drill_files: list[ImportedDrillFile] = []
        self._alignment_holes: list[tuple[float, float, float]] = []
        self._bounds = BoardBounds()
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._last_pos = QPointF()
        self._dragging = False

    def load_project_geometry(
        self,
        gerber_files: list[ImportedGerberFile],
        drill_files: list[ImportedDrillFile],
        alignment_holes: list[tuple[float, float, float]],
    ) -> None:
        self._gerber_files = gerber_files
        self._drill_files = drill_files
        self._alignment_holes = alignment_holes
        self._bounds = BoardBounds()
        for gerber in gerber_files:
            self._bounds.include_bounds(gerber.bounds)
        for drill in drill_files:
            self._bounds.include_bounds(drill.bounds)
        for x, y, diameter in alignment_holes:
            self._bounds.include_point(x, y, diameter * 0.5)
        self.fit_to_view()

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
            self._dragging = True
            self._last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
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

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        self._zoom = max(0.1, min(1000.0, self._zoom * factor))
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._bounds.is_empty and abs(self._pan_x) < 1e-6 and abs(self._pan_y) < 1e-6:
            self.fit_to_view()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), BACKGROUND)
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
            pen = QPen(GRID_MAJOR if abs(round(x / spacing)) % 5 == 0 else GRID_MINOR, 1)
            painter.setPen(pen)
            self._draw_world_line(
                painter,
                (x, self._bounds.y_min - spacing),
                (x, self._bounds.y_max + spacing),
            )
            x += spacing
        y = math.floor(self._bounds.y_min / spacing) * spacing
        while y <= self._bounds.y_max + spacing:
            pen = QPen(GRID_MAJOR if abs(round(y / spacing)) % 5 == 0 else GRID_MINOR, 1)
            painter.setPen(pen)
            self._draw_world_line(
                painter,
                (self._bounds.x_min - spacing, y),
                (self._bounds.x_max + spacing, y),
            )
            y += spacing

    def _draw_geometry(self, painter: QPainter) -> None:
        for index, gerber in enumerate(self._gerber_files):
            color = PALETTE[index % len(PALETTE)]
            painter.setPen(QPen(OUTLINE, 2.2))
            self._draw_outline(painter, gerber.outline)

            fill_color = QColor(color)
            fill_color.setAlpha(55)
            stroke_color = QColor(color)
            stroke_color.setAlpha(190)
            painter.setPen(QPen(stroke_color, 1.4))
            for region in gerber.regions:
                self._draw_polygon(painter, region, fill_color)
            for trace_start, trace_end, width in gerber.traces:
                pen = QPen(stroke_color, max(1.2, width * self._zoom))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                self._draw_world_line(painter, trace_start, trace_end)
            painter.setPen(QPen(stroke_color, 1.2))
            for center, aperture in gerber.pads:
                self._draw_pad(painter, center, aperture, fill_color, stroke_color)

        painter.setPen(QPen(DRILL, 1.3))
        for drill in self._drill_files:
            for hole in drill.holes:
                self._draw_hole(painter, hole)

        painter.setPen(QPen(ALIGNMENT, 1.8))
        for hole in self._alignment_holes:
            self._draw_alignment_hole(painter, hole)

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
        painter.setBrush(BACKGROUND)
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

    def _world_to_screen(self, x: float, y: float) -> QPointF:
        center_x = self._bounds.center_x
        center_y = self._bounds.center_y
        screen_x = (x - center_x) * self._zoom + (self.width() * 0.5) + self._pan_x
        screen_y = (center_y - y) * self._zoom + (self.height() * 0.5) + self._pan_y
        return QPointF(screen_x, screen_y)

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.setPen(TEXT)
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
            f"Bounds: X {self._bounds.x_min:.2f}..{self._bounds.x_max:.2f}   "
            f"Y {self._bounds.y_min:.2f}..{self._bounds.y_max:.2f}",
        )

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
