from __future__ import annotations

from dataclasses import dataclass
import math
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .gcode_parser import Point3D, ToolpathDocument

BACKGROUND = QColor("#11151c")
GRID_MAJOR = QColor("#2f3945")
GRID_MINOR = QColor("#202831")
AXIS_X = QColor("#d84d4d")
AXIS_Y = QColor("#3ecf8e")
AXIS_Z = QColor("#4ea1ff")
RAPID = QColor("#f2c94c")
CUT = QColor("#f97316")
TEXT = QColor("#dfe7ef")


@dataclass
class CameraState:
    yaw: float = math.radians(35.0)
    pitch: float = math.radians(-25.0)
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0


class ToolpathViewer(QOpenGLWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 480)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.document: ToolpathDocument | None = None
        self.camera = CameraState()
        self._last_pos = None
        self._left_drag = False
        self._right_drag = False
        self._pivot = Point3D(0.0, 0.0, 0.0)
        self._extent = 100.0

    def load_document(self, document: ToolpathDocument | None) -> None:
        self.document = document
        if document is None or not document.segments:
            self._pivot = Point3D(0.0, 0.0, 0.0)
            self._extent = 100.0
            self.camera = CameraState()
            self.update()
            return

        stats = document.stats
        self._pivot = Point3D(
            (stats.min_point.x + stats.max_point.x) * 0.5,
            (stats.min_point.y + stats.max_point.y) * 0.5,
            (stats.min_point.z + stats.max_point.z) * 0.5,
        )
        span_x = stats.max_point.x - stats.min_point.x
        span_y = stats.max_point.y - stats.min_point.y
        span_z = stats.max_point.z - stats.min_point.z
        self._extent = max(span_x, span_y, span_z, 10.0)
        self.fit_to_view()

    def fit_to_view(self) -> None:
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        scale = min(width, height) * 0.72
        self.camera = CameraState(zoom=scale / max(self._extent, 1.0))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), BACKGROUND)
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
            self.camera.pitch = max(math.radians(-89.0), min(math.radians(89.0), self.camera.pitch))
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
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        self.camera.zoom = max(0.0005, min(1e6, self.camera.zoom * factor))
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
        spacing = self._nice_spacing(self._extent / 10.0)
        half = max(self._extent * 0.75, spacing * 5.0)
        minor_pen = QPen(GRID_MINOR, 1)
        major_pen = QPen(GRID_MAJOR, 1)
        value = -half
        while value <= half + 1e-9:
            is_major = abs(round(value / spacing)) % 5 == 0
            painter.setPen(major_pen if is_major else minor_pen)
            self._draw_line(painter, Point3D(value, -half, 0.0), Point3D(value, half, 0.0))
            self._draw_line(painter, Point3D(-half, value, 0.0), Point3D(half, value, 0.0))
            value += spacing

    def _draw_axes(self, painter: QPainter) -> None:
        axis_length = max(self._extent * 0.6, 10.0)
        painter.setPen(QPen(AXIS_X, 2))
        self._draw_line(painter, Point3D(0.0, 0.0, 0.0), Point3D(axis_length, 0.0, 0.0))
        painter.setPen(QPen(AXIS_Y, 2))
        self._draw_line(painter, Point3D(0.0, 0.0, 0.0), Point3D(0.0, axis_length, 0.0))
        painter.setPen(QPen(AXIS_Z, 2))
        self._draw_line(painter, Point3D(0.0, 0.0, 0.0), Point3D(0.0, 0.0, axis_length * 0.35))

    def _draw_toolpath(self, painter: QPainter) -> None:
        if not self.document:
            return
        for segment in self.document.segments:
            painter.setPen(QPen(RAPID if segment.rapid else CUT, 1.8 if not segment.rapid else 1.2))
            self._draw_line(painter, segment.start, segment.end)

    def _draw_overlay(self, painter: QPainter) -> None:
        painter.setPen(TEXT)
        painter.drawText(16, 28, "mekatrol-pcbcam")
        if not self.document:
            painter.drawText(16, 50, "Open an .nc file to inspect its motion path.")
            painter.drawText(16, 72, "Left drag: orbit   Right drag: pan   Wheel: zoom   F: fit")
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

        depth = 1.0 / max(0.35, 2.4 - (z2 / max(self._extent, 1.0)))
        scale = self.camera.zoom * depth
        sx = x1 * scale + (self.width() * 0.5) + self.camera.pan_x
        sy_screen = -y2 * scale + (self.height() * 0.5) + self.camera.pan_y
        return QPointF(sx, sy_screen)

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
