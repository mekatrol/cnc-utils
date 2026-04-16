from __future__ import annotations

import logging
import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPainter, QPaintEvent, QPen, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .camera_state import CameraState
from .point_3d import Point3D
from .theme import AppTheme
from .toolpath_document import ToolpathDocument


logger = logging.getLogger(__name__)


class ToolpathViewer(QOpenGLWidget):
    def __init__(self, theme: AppTheme, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setMinimumSize(720, 480)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.document: ToolpathDocument | None = None
        self.camera = CameraState()
        self._last_pos = None
        self._left_drag = False
        self._right_drag = False
        self._pivot = Point3D(0.0, 0.0, 0.0)
        self._extent = 100.0
        self._bounds_min = Point3D(-50.0, -50.0, 0.0)
        self._bounds_max = Point3D(50.0, 50.0, 0.0)
        self._span_x = 100.0
        self._span_y = 100.0
        self._span_z = 0.0
        logger.debug("Toolpath viewer initialized")

    def load_document(self, document: ToolpathDocument | None) -> None:
        self.document = document
        if document is None or not document.segments:
            self._pivot = Point3D(0.0, 0.0, 0.0)
            self._extent = 100.0
            self._bounds_min = Point3D(-50.0, -50.0, 0.0)
            self._bounds_max = Point3D(50.0, 50.0, 0.0)
            self._span_x = 100.0
            self._span_y = 100.0
            self._span_z = 0.0
            self.camera = CameraState()
            self.update()
            logger.debug("Viewer reset to empty document state")
            return

        stats = document.stats
        self._bounds_min = stats.min_point
        self._bounds_max = stats.max_point
        self._pivot = Point3D(
            (stats.min_point.x + stats.max_point.x) * 0.5,
            (stats.min_point.y + stats.max_point.y) * 0.5,
            (stats.min_point.z + stats.max_point.z) * 0.5,
        )
        self._span_x = stats.max_point.x - stats.min_point.x
        self._span_y = stats.max_point.y - stats.min_point.y
        self._span_z = stats.max_point.z - stats.min_point.z
        self._extent = max(self._span_x, self._span_y, self._span_z, 10.0)
        logger.debug(
            "Viewer loaded document: %s extent=%.3f pivot=(%.3f, %.3f, %.3f)",
            document.path,
            self._extent,
            self._pivot.x,
            self._pivot.y,
            self._pivot.z,
        )
        self.fit_to_view()

    def fit_to_view(self) -> None:
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        bounds = self._projected_bounds_at_unit_zoom()
        bounds_width = max(bounds[2] - bounds[0], 1e-6)
        bounds_height = max(bounds[3] - bounds[1], 1e-6)
        padding = 0.82
        zoom = min((width * padding) / bounds_width, (height * padding) / bounds_height)
        center_x = (bounds[0] + bounds[2]) * 0.5
        center_y = (bounds[1] + bounds[3]) * 0.5
        self.camera = CameraState(
            zoom=max(zoom, 0.0005),
            pan_x=-center_x * zoom,
            pan_y=-center_y * zoom,
        )
        self.update()
        logger.debug(
            "Viewer fit-to-view applied: width=%d height=%d zoom=%.5f pan=(%.2f, %.2f)",
            width,
            height,
            self.camera.zoom,
            self.camera.pan_x,
            self.camera.pan_y,
        )

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self._theme.named_color("toolpath_background"))
        self._draw_grid(painter)
        self._draw_axes(painter)
        self._draw_toolpath(painter)
        self._draw_overlay(painter)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_drag = True
        elif event.button() == Qt.MouseButton.RightButton:
            self._right_drag = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_pos is None:
            self._last_pos = event.position()
            return
        delta = event.position() - self._last_pos
        self._last_pos = event.position()

        if self._left_drag:
            self.camera.yaw += delta.x() * 0.01
            self.camera.pitch += delta.y() * 0.01
            self.camera.pitch = max(
                math.radians(-89.0),
                min(math.radians(89.0), self.camera.pitch),
            )
            self.update()
        elif self._right_drag:
            self.camera.pan_x += delta.x()
            self.camera.pan_y += delta.y()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._left_drag = False
        self._right_drag = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        cursor_position = event.position()
        center_x = self.width() * 0.5
        center_y = self.height() * 0.5
        offset_x = cursor_position.x() - center_x - self.camera.pan_x
        offset_y = cursor_position.y() - center_y - self.camera.pan_y
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        self.camera.zoom = max(0.0005, min(1e6, self.camera.zoom * factor))
        self.camera.pan_x = cursor_position.x() - center_x - (offset_x * factor)
        self.camera.pan_y = cursor_position.y() - center_y - (offset_y * factor)
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.document and self.camera.zoom <= 0:
            self.fit_to_view()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F:
            self.fit_to_view()
            event.accept()
            return
        super().keyPressEvent(event)

    def _draw_grid(self, painter: QPainter) -> None:
        spacing = self._grid_spacing()
        margin = max(
            spacing * 4.0,
            self._span_x * 0.2,
            self._span_y * 0.2,
            5.0,
        )
        min_x = min(self._bounds_min.x, 0.0) - margin
        max_x = max(self._bounds_max.x, 0.0) + margin
        min_y = min(self._bounds_min.y, 0.0) - margin
        max_y = max(self._bounds_max.y, 0.0) + margin
        start_x = math.floor(min_x / spacing) * spacing
        end_x = math.ceil(max_x / spacing) * spacing
        start_y = math.floor(min_y / spacing) * spacing
        end_y = math.ceil(max_y / spacing) * spacing
        minor_pen = QPen(self._theme.named_color("toolpath_grid_minor"), 1)
        major_pen = QPen(self._theme.named_color("toolpath_grid_major"), 1)
        value = start_x
        while value <= end_x + 1e-9:
            is_major = self._is_major_grid_line(value, spacing)
            painter.setPen(major_pen if is_major else minor_pen)
            self._draw_line(
                painter,
                Point3D(value, start_y, 0.0),
                Point3D(value, end_y, 0.0),
            )
            value += spacing
        value = start_y
        while value <= end_y + 1e-9:
            is_major = self._is_major_grid_line(value, spacing)
            painter.setPen(major_pen if is_major else minor_pen)
            self._draw_line(
                painter,
                Point3D(start_x, value, 0.0),
                Point3D(end_x, value, 0.0),
            )
            value += spacing

    def _draw_axes(self, painter: QPainter) -> None:
        axis_length = max(self._span_x, self._span_y, 10.0) * 0.6
        painter.setPen(QPen(self._theme.named_color("toolpath_axis_x"), 2))
        self._draw_line(
            painter,
            Point3D(0.0, 0.0, 0.0),
            Point3D(axis_length, 0.0, 0.0),
        )
        painter.setPen(QPen(self._theme.named_color("toolpath_axis_y"), 2))
        self._draw_line(
            painter,
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.0, axis_length, 0.0),
        )
        painter.setPen(QPen(self._theme.named_color("toolpath_axis_z"), 2))
        self._draw_line(
            painter,
            Point3D(0.0, 0.0, 0.0),
            Point3D(0.0, 0.0, axis_length * 0.35),
        )

    def _draw_toolpath(self, painter: QPainter) -> None:
        if not self.document:
            return
        for segment in self.document.segments:
            painter.setPen(
                QPen(
                    self._theme.named_color("toolpath_rapid")
                    if segment.rapid
                    else self._theme.named_color("toolpath_cut"),
                    1.8 if not segment.rapid else 1.2,
                )
            )
            self._draw_line(painter, segment.start, segment.end)

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.setPen(self._theme.named_color("toolpath_text"))
        painter.drawText(16, 28, "mekatrol-pcbcam")
        if not self.document:
            painter.drawText(16, 50, "Open an .nc file to inspect its motion path.")
            painter.drawText(
                16,
                72,
                "Left drag: orbit   Right drag: pan   Wheel: zoom   F: fit",
            )
            return

        stats = self.document.stats
        painter.drawText(16, 50, self.document.path.name)
        painter.drawText(
            16,
            72,
            f"Segments: {stats.segment_count}  Cut: {stats.cut_count}  Rapid: {stats.rapid_count}",
        )
        painter.drawText(16, 94, f"Path length: {stats.path_length:.2f} mm")

    def _draw_line(self, painter: QPainter, start: Point3D, end: Point3D) -> None:
        p1 = self._project(start)
        p2 = self._project(end)
        painter.drawLine(p1, p2)

    def _project(self, point: Point3D) -> QPointF:
        x1, y2, z2 = self._camera_space(point)
        depth = 1.0 / max(0.35, 2.4 - (z2 / max(self._extent, 1.0)))
        scale = self.camera.zoom * depth
        sx = x1 * scale + (self.width() * 0.5) + self.camera.pan_x
        sy_screen = -y2 * scale + (self.height() * 0.5) + self.camera.pan_y
        return QPointF(sx, sy_screen)

    def _camera_space(self, point: Point3D) -> tuple[float, float, float]:
        x = point.x - self._pivot.x
        y = point.y - self._pivot.y
        z = point.z - self._pivot.z

        cy = math.cos(self.camera.yaw)
        sy = math.sin(self.camera.yaw)
        cp = math.cos(self.camera.pitch)
        sp = math.sin(self.camera.pitch)

        x1 = x * cy - y * sy
        y1 = x * sy + y * cy
        z1 = z

        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp
        return x1, y2, z2

    def _projected_bounds_at_unit_zoom(self) -> tuple[float, float, float, float]:
        corners = (
            Point3D(x, y, z)
            for x in (self._bounds_min.x, self._bounds_max.x)
            for y in (self._bounds_min.y, self._bounds_max.y)
            for z in (self._bounds_min.z, self._bounds_max.z)
        )
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        for corner in corners:
            x1, y2, z2 = self._camera_space(corner)
            depth = 1.0 / max(0.35, 2.4 - (z2 / max(self._extent, 1.0)))
            px = x1 * depth
            py = -y2 * depth
            min_x = min(min_x, px)
            min_y = min(min_y, py)
            max_x = max(max_x, px)
            max_y = max(max_y, py)
        if not math.isfinite(min_x) or not math.isfinite(min_y):
            fallback = max(self._extent, 10.0) * 0.5
            return (-fallback, -fallback, fallback, fallback)
        return (min_x, min_y, max_x, max_y)

    def _grid_spacing(self) -> float:
        target_pixels = 40.0
        zoom = max(self.camera.zoom, 1e-6)
        return self._nice_spacing(target_pixels / zoom)

    def _is_major_grid_line(self, value: float, spacing: float) -> bool:
        major_spacing = spacing * 5.0
        return math.isclose(value / major_spacing, round(value / major_spacing), abs_tol=1e-6)

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
