from __future__ import annotations

from math import atan2, cos, sin

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .box_settings import BoxSettings
from .geometry import Panel, Point


class PreviewWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(620, 460)
        self._panels: list[Panel] = []
        self._settings = BoxSettings()
        self._mode = "flat"

    def set_preview(
        self, panels: list[Panel], settings: BoxSettings, mode: str
    ) -> None:
        self._panels = panels
        self._settings = settings
        self._mode = mode
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#11151c"))
        if self._mode in {"assembled", "box"}:
            self._draw_assembled(painter)
            if self._mode == "box":
                self._draw_box_dimensions(painter)
        elif self._mode == "material":
            self._draw_material_preview(painter)
        elif self._mode == "joints":
            self._draw_joint_preview(painter)
        elif self._mode == "tabs":
            self._draw_flat(painter, show_tabs=True)
        elif self._mode == "generate":
            self._draw_flat(painter, show_tabs=self._settings.include_tabs)
            self._draw_job_summary(painter)
        else:
            self._draw_flat(painter)
        painter.end()

    def _draw_flat(self, painter: QPainter, show_tabs: bool = False) -> None:
        bounds = self._layout_bounds()
        scale, offset = self._view_transform(bounds)
        self._draw_grid(painter, scale, offset)
        self._draw_stock_sheets(painter, scale, offset)
        tab_count = 0
        for panel in self._panels:
            self._draw_flat_panel(painter, panel, scale, offset)
            if show_tabs and self._settings.include_tabs:
                tab_count += self._draw_holding_tabs(painter, panel, scale, offset)
            painter.setPen(QPen(QColor("#dfe7ef"), 1.0))
            text_point = self._map_point(
                Point(
                    panel.origin_x + panel.width * 0.5,
                    panel.origin_y + panel.height * 0.5,
                ),
                scale,
                offset,
            )
            painter.drawText(
                QRectF(text_point.x() - 45.0, text_point.y() - 10.0, 90.0, 20.0),
                Qt.AlignmentFlag.AlignCenter,
                panel.name,
            )
        if show_tabs and self._settings.include_tabs and tab_count == 0:
            self._draw_status_label(
                "no holding tabs fit current edge segments", painter
            )

    def _draw_flat_panel(
        self, painter: QPainter, panel: Panel, scale: float, offset: QPointF
    ) -> None:
        polygon = QPolygonF(
            [self._map_point(point, scale, offset) for point in panel.outline]
        )
        panel_path = QPainterPath()
        panel_path.addPolygon(polygon)
        panel_path.closeSubpath()

        painter.setBrush(QColor(48, 87, 74, 120))
        painter.setPen(QPen(QColor("#ffd166"), 1.6))
        painter.drawPolygon(polygon)

        if not panel.relief_points or self._settings.relief_diameter <= 0.0:
            return

        relief_radius = (
            max(self._settings.relief_diameter, self._settings.bit_diameter)
            * scale
            * 0.5
        )
        relief_pen = QPen(QColor("#ff9f43"), 1.2)

        painter.save()
        painter.setClipPath(panel_path)
        painter.setBrush(QColor("#11151c"))
        painter.setPen(relief_pen)
        for point in panel.relief_points:
            center = self._map_point(point, scale, offset)
            painter.drawEllipse(center, relief_radius, relief_radius)
        painter.restore()

        nominal_corner_pen = QPen(QColor(255, 209, 102, 150), 1.0)
        nominal_corner_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(nominal_corner_pen)
        painter.drawPolygon(polygon)

    def _draw_material_preview(self, painter: QPainter) -> None:
        material = self._settings.material_thickness
        edge_bit = self._settings.bit_diameter
        relief_bit = max(self._settings.relief_diameter, self._settings.bit_diameter)
        sample_width = max(material * 7.0, relief_bit * 6.0, 44.0)
        sample_height = max(material * 5.0, relief_bit * 5.0, 34.0)
        notch = max(material, relief_bit * 1.5)
        bounds = (-18.0, sample_width + 34.0, -18.0, sample_height + 36.0)
        scale, offset = self._view_transform(bounds)
        self._draw_grid(painter, scale, offset)

        panel_points = [
            Point(0.0, 0.0),
            Point(sample_width, 0.0),
            Point(sample_width, sample_height),
            Point(0.0, sample_height),
            Point(0.0, sample_height - notch),
            Point(notch, sample_height - notch),
            Point(notch, notch),
            Point(0.0, notch),
        ]
        polygon = QPolygonF(
            [self._map_point(point, scale, offset) for point in panel_points]
        )
        painter.setBrush(QColor(48, 87, 74, 135))
        painter.setPen(QPen(QColor("#ffd166"), 1.8))
        painter.drawPolygon(polygon)

        relief_center = self._map_point(
            Point(notch, sample_height - notch), scale, offset
        )
        relief_radius = relief_bit * scale * 0.5
        painter.setBrush(QColor("#11151c"))
        painter.setPen(QPen(QColor("#ff9f43"), 1.5))
        painter.drawEllipse(relief_center, relief_radius, relief_radius)

        edge_bit_center = self._map_point(
            Point(sample_width + edge_bit, sample_height * 0.55), scale, offset
        )
        painter.setBrush(QColor(78, 161, 255, 85))
        painter.setPen(QPen(QColor("#4ea1ff"), 1.3))
        painter.drawEllipse(
            edge_bit_center, edge_bit * scale * 0.5, edge_bit * scale * 0.5
        )
        painter.setPen(QPen(QColor("#dfe7ef"), 1.0))
        painter.drawText(
            QRectF(edge_bit_center.x() - 42.0, edge_bit_center.y() + 10.0, 84.0, 18.0),
            Qt.AlignmentFlag.AlignCenter,
            f"{edge_bit:.3f} mm",
        )

        relief_label = QPointF(
            relief_center.x() + relief_radius + 8.0, relief_center.y()
        )
        painter.setPen(QPen(QColor("#dfe7ef"), 1.0))
        painter.drawText(
            QRectF(relief_label.x(), relief_label.y() - 9.0, 120.0, 18.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{relief_bit:.3f} mm relief",
        )
        thickness_start = self._map_point(Point(0.0, -material), scale, offset)
        thickness_end = self._map_point(Point(material, -material), scale, offset)
        painter.setPen(QPen(QColor("#9bd88f"), 1.4))
        painter.drawLine(thickness_start, thickness_end)
        painter.setPen(QPen(QColor("#dfe7ef"), 1.0))
        painter.drawText(
            QRectF(thickness_start.x(), thickness_start.y() + 4.0, 130.0, 18.0),
            Qt.AlignmentFlag.AlignLeft,
            f"{material:.3f} mm material",
        )

    def _draw_joint_preview(self, painter: QPainter) -> None:
        panel = self._panel_named("front") or (
            self._panels[0] if self._panels else None
        )
        if panel is None:
            return
        bounds = self._expanded_bounds(
            panel.bounds, self._settings.material_thickness * 2.0
        )
        scale, offset = self._view_transform(bounds)
        self._draw_grid(painter, scale, offset)
        self._draw_flat_panel(painter, panel, scale, offset)

        painter.setPen(QPen(QColor("#9bd88f"), 2.4))
        for start, end in zip(panel.outline, panel.outline[1:]):
            if self._segment_length(start, end) < self._settings.finger_width * 0.45:
                continue
            midpoint = Point((start.x + end.x) * 0.5, (start.y + end.y) * 0.5)
            if not self._point_is_near_panel_edge(panel, midpoint):
                painter.drawLine(
                    self._map_point(start, scale, offset),
                    self._map_point(end, scale, offset),
                )

        painter.setPen(QPen(QColor("#dfe7ef"), 1.1))
        painter.drawText(
            QRectF(16.0, 14.0, self.width() - 32.0, 22.0),
            Qt.AlignmentFlag.AlignLeft,
            f"finger pitch about {self._finger_pitch(panel):.3f} mm",
        )

    def _draw_holding_tabs(
        self, painter: QPainter, panel: Panel, scale: float, offset: QPointF
    ) -> int:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        placed_tabs = 0
        for start, end in zip(panel.outline, panel.outline[1:]):
            if not self._is_bounding_edge(panel, start, end):
                continue
            dx = end.x - start.x
            dy = end.y - start.y
            length = self._segment_length(start, end)
            painter.setPen(QPen(QColor(78, 161, 255, 120), 1.4))
            painter.drawLine(
                self._map_point(start, scale, offset),
                self._map_point(end, scale, offset),
            )
            if length < self._settings.tab_width * 2.0:
                continue
            tab_count = max(1, int(length // 85.0))
            painter.setPen(QPen(QColor("#4ea1ff"), 3.0))
            for index in range(tab_count):
                center = length * (index + 1) / (tab_count + 1)
                tab_start = max(0.0, center - self._settings.tab_width * 0.5)
                tab_end = min(length, center + self._settings.tab_width * 0.5)
                start_point = self._point_along_segment(
                    start, dx, dy, length, tab_start
                )
                end_point = self._point_along_segment(start, dx, dy, length, tab_end)
                painter.drawLine(
                    self._map_point(start_point, scale, offset),
                    self._map_point(end_point, scale, offset),
                )
                placed_tabs += 1
        return placed_tabs

    def _draw_job_summary(self, painter: QPainter) -> None:
        sheet_count = self._stock_sheet_count()
        summary = (
            f"{len(self._panels)} panels  |  "
            f"{sheet_count} stock sheet{'' if sheet_count == 1 else 's'}  |  "
            f"{self._settings.bit_diameter:.3f} mm cutter  |  "
            f"final Z {self._settings.final_cut_depth:.3f} mm"
        )
        rect = QRectF(12.0, 12.0, self.width() - 24.0, 26.0)
        painter.setBrush(QColor(17, 21, 28, 210))
        painter.setPen(QPen(QColor("#596270"), 1.0))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor("#dfe7ef"), 1.0))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, summary)

    def _draw_status_label(self, text: str, painter: QPainter) -> None:
        rect = QRectF(12.0, self.height() - 38.0, self.width() - 24.0, 26.0)
        painter.setBrush(QColor(17, 21, 28, 220))
        painter.setPen(QPen(QColor("#596270"), 1.0))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor("#dfe7ef"), 1.0))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_stock_sheets(
        self, painter: QPainter, scale: float, offset: QPointF
    ) -> None:
        painter.setBrush(QColor(26, 30, 38, 130))
        painter.setPen(QPen(QColor("#596270"), 1.4))
        for stock_index in range(self._stock_sheet_count()):
            origin_x = self._stock_origin_x(stock_index)
            stock_rect = QRectF(
                self._map_point(
                    Point(origin_x, self._settings.stock_height), scale, offset
                ),
                self._map_point(
                    Point(origin_x + self._settings.stock_width, 0.0), scale, offset
                ),
            ).normalized()
            painter.drawRect(stock_rect)
            painter.setPen(QPen(QColor("#8d97a3"), 1.0))
            painter.drawText(
                QRectF(stock_rect.x() + 8.0, stock_rect.y() + 6.0, 140.0, 18.0),
                Qt.AlignmentFlag.AlignLeft,
                f"stock {stock_index + 1}",
            )
            painter.setPen(QPen(QColor("#596270"), 1.4))

    def _draw_assembled(self, painter: QPainter) -> None:
        x_size = self._settings.size_x
        y_size = self._settings.size_y
        z_size = self._settings.size_z
        iso_x = y_size * 0.45
        iso_y = y_size * 0.32
        points = [
            Point(0.0, z_size),
            Point(x_size, z_size),
            Point(x_size + iso_x, z_size - iso_y),
            Point(x_size + iso_x, -iso_y),
            Point(iso_x, -iso_y),
            Point(0.0, 0.0),
        ]
        bounds = (-10.0, x_size + iso_x + 10.0, -iso_y - 10.0, z_size + 10.0)
        scale, offset = self._view_transform(bounds)
        self._draw_grid(painter, scale, offset)

        front = [
            Point(0.0, 0.0),
            Point(x_size, 0.0),
            Point(x_size, z_size),
            Point(0.0, z_size),
        ]
        side = [
            Point(x_size, 0.0),
            Point(x_size + iso_x, -iso_y),
            Point(x_size + iso_x, z_size - iso_y),
            Point(x_size, z_size),
        ]
        top = [
            Point(0.0, z_size),
            Point(x_size, z_size),
            Point(x_size + iso_x, z_size - iso_y),
            Point(iso_x, z_size - iso_y),
        ]
        self._draw_face(painter, side, scale, offset, QColor("#4a6fa5"))
        self._draw_face(painter, top, scale, offset, QColor("#608b4e"))
        self._draw_face(painter, front, scale, offset, QColor("#b26f3a"))
        painter.setPen(QPen(QColor("#dfe7ef"), 1.2))
        painter.drawPolyline(
            QPolygonF([self._map_point(point, scale, offset) for point in points])
        )
        title = (
            "assembled box" if self._settings.box_kind == "box" else "assembled drawer"
        )
        painter.drawText(16, 26, title)

    def _draw_box_dimensions(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#dfe7ef"), 1.0))
        painter.drawText(
            QRectF(16.0, 44.0, 250.0, 22.0),
            Qt.AlignmentFlag.AlignLeft,
            (
                f"{self._settings.size_x:.1f} x {self._settings.size_y:.1f} x "
                f"{self._settings.size_z:.1f} mm"
            ),
        )

    def _draw_face(
        self,
        painter: QPainter,
        points: list[Point],
        scale: float,
        offset: QPointF,
        color: QColor,
    ) -> None:
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 165))
        painter.setPen(QPen(QColor("#dfe7ef"), 1.2))
        painter.drawPolygon(
            QPolygonF([self._map_point(point, scale, offset) for point in points])
        )

    def _draw_grid(self, painter: QPainter, scale: float, offset: QPointF) -> None:
        painter.setPen(QPen(QColor("#202831"), 1.0))
        spacing = 25.0 * scale
        if spacing < 10.0:
            spacing = 50.0 * scale
        x = offset.x() % spacing
        while x < self.width():
            painter.drawLine(QPointF(x, 0.0), QPointF(x, self.height()))
            x += spacing
        y = offset.y() % spacing
        while y < self.height():
            painter.drawLine(QPointF(0.0, y), QPointF(self.width(), y))
            y += spacing

    def _layout_bounds(self) -> tuple[float, float, float, float]:
        sheet_count = self._stock_sheet_count()
        if not self._panels:
            return 0.0, self._settings.stock_width, 0.0, self._settings.stock_height
        min_x = min(panel.bounds[0] for panel in self._panels)
        max_x = max(panel.bounds[1] for panel in self._panels)
        min_y = min(panel.bounds[2] for panel in self._panels)
        max_y = max(panel.bounds[3] for panel in self._panels)
        if sheet_count:
            max_x = max(
                max_x,
                self._stock_origin_x(sheet_count - 1) + self._settings.stock_width,
            )
            min_x = min(min_x, 0.0)
            min_y = min(min_y, 0.0)
            max_y = max(max_y, self._settings.stock_height)
        return min_x, max_x, min_y, max_y

    def _expanded_bounds(
        self, bounds: tuple[float, float, float, float], margin: float
    ) -> tuple[float, float, float, float]:
        min_x, max_x, min_y, max_y = bounds
        return min_x - margin, max_x + margin, min_y - margin, max_y + margin

    def _view_transform(
        self, bounds: tuple[float, float, float, float]
    ) -> tuple[float, QPointF]:
        min_x, max_x, min_y, max_y = bounds
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        scale = min((self.width() - 42.0) / span_x, (self.height() - 42.0) / span_y)
        scale = max(scale, 0.01)
        offset = QPointF(
            21.0 - min_x * scale + ((self.width() - 42.0) - span_x * scale) * 0.5,
            self.height()
            - 21.0
            + min_y * scale
            - ((self.height() - 42.0) - span_y * scale) * 0.5,
        )
        return scale, offset

    def _map_point(self, point: Point, scale: float, offset: QPointF) -> QPointF:
        return QPointF(offset.x() + point.x * scale, offset.y() - point.y * scale)

    def _panel_named(self, name: str) -> Panel | None:
        for panel in self._panels:
            if panel.name == name:
                return panel
        return None

    def _stock_sheet_count(self) -> int:
        if not self._panels:
            return 1
        return max(panel.stock_index for panel in self._panels) + 1

    def _stock_origin_x(self, stock_index: int) -> float:
        if self._panels:
            for panel in self._panels:
                if panel.stock_index == stock_index:
                    return panel.stock_origin_x
        return stock_index * (
            self._settings.stock_width + self._settings.layout_gap * 2.0
        )

    def _finger_pitch(self, panel: Panel) -> float:
        outline = (
            panel.outline[:-1]
            if panel.outline[-1] == panel.outline[0]
            else panel.outline
        )
        runs = [
            self._segment_length(start, end)
            for start, end in zip(outline, [*outline[1:], outline[0]])
            if self._segment_length(start, end)
            > self._settings.material_thickness * 1.2
        ]
        return min(runs) if runs else self._settings.finger_width

    def _point_is_near_panel_edge(self, panel: Panel, point: Point) -> bool:
        min_x, max_x, min_y, max_y = panel.bounds
        tolerance = 0.001
        return (
            abs(point.x - min_x) < tolerance
            or abs(point.x - max_x) < tolerance
            or abs(point.y - min_y) < tolerance
            or abs(point.y - max_y) < tolerance
        )

    def _is_bounding_edge(self, panel: Panel, start: Point, end: Point) -> bool:
        min_x, max_x, min_y, max_y = panel.bounds
        tolerance = 0.001
        return (
            abs(start.x - end.x) < tolerance
            and (abs(start.x - min_x) < tolerance or abs(start.x - max_x) < tolerance)
        ) or (
            abs(start.y - end.y) < tolerance
            and (abs(start.y - min_y) < tolerance or abs(start.y - max_y) < tolerance)
        )

    def _segment_length(self, start: Point, end: Point) -> float:
        dx = end.x - start.x
        dy = end.y - start.y
        return (dx * dx + dy * dy) ** 0.5

    def _point_along_segment(
        self, start: Point, dx: float, dy: float, length: float, distance: float
    ) -> Point:
        angle = atan2(dy, dx)
        return Point(start.x + cos(angle) * distance, start.y + sin(angle) * distance)
