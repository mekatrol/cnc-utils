from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .theme import AppTheme


class WizardStepBar(QWidget):
    step_selected = Signal(int)

    def __init__(self, titles: list[str], theme: AppTheme, parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._titles = titles
        self._current_step_index = 0
        self._completed_steps: set[int] = set()
        self._enabled_steps: set[int] = {0}
        self._step_paths: list[QPainterPath] = []
        self._step_rects: list[QRectF] = []
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def set_state(
        self, *, current_step_index: int, completed_steps: set[int], is_step_enabled
    ) -> None:
        self._current_step_index = current_step_index
        self._completed_steps = set(completed_steps)
        self._enabled_steps = {
            index for index in range(len(self._titles)) if is_step_enabled(index)
        }
        self.update()

    def sizeHint(self) -> QSize:
        return self.minimumSizeHint()

    def minimumSizeHint(self) -> QSize:
        total_width, _ = self._layout_metrics()
        return QSize(total_width, 64)

    def step_bounds(self, index: int) -> QRect:
        if not 0 <= index < len(self._titles):
            return QRect()
        if len(self._step_rects) != len(self._titles):
            self._refresh_step_geometry()
        if not 0 <= index < len(self._step_rects):
            return QRect()
        return self._step_rects[index].toRect()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        for index, path in enumerate(self._step_paths):
            if index not in self._enabled_steps:
                continue
            if path.contains(position):
                self.step_selected.emit(index)
                event.accept()
                return
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), self._theme.named_color("wizard_step_bar_background"))

        _, arrow = self._layout_metrics()
        self._refresh_step_geometry()
        self._step_paths = []
        step_count = max(len(self._titles), 1)

        for index, title in enumerate(self._titles):
            step_rect = self._step_rects[index]
            path = self._build_step_path(step_rect, arrow, step_count)
            self._step_paths.append(path)

            fill, border, text = self._colors_for_step(index)
            painter.fillPath(path, fill)
            painter.setPen(QPen(border, 1.2))
            painter.drawPath(path)

            text_left_padding = 16.0 if index == 0 else arrow * 0.95
            text_right_padding = 16.0 if index == step_count - 1 else arrow * 0.95
            text_rect = step_rect.adjusted(
                int(text_left_padding), 0, -int(text_right_padding), 0
            )
            painter.setPen(text)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, title)

        painter.end()

    def _refresh_step_geometry(self) -> None:
        rect = self.rect().adjusted(0, 4, 0, -4)
        _, arrow = self._layout_metrics()
        self._step_rects = []

        left = float(rect.left())
        for width in self._step_widths():
            step_rect = QRectF(left, rect.top(), float(width), rect.height())
            self._step_rects.append(step_rect)
            left += width - arrow

    def _layout_metrics(self) -> tuple[int, float]:
        arrow = 24.0
        widths = self._step_widths()
        total_width = max(
            int(sum(widths) - max(0, len(widths) - 1) * arrow),
            1,
        )
        return total_width, arrow

    def _step_widths(self) -> list[int]:
        metrics = QFontMetrics(self.font())
        widths: list[int] = []
        for title in self._titles:
            label_width = metrics.horizontalAdvance(title)
            widths.append(max(170, label_width + 56))
        return widths

    def _build_step_path(
        self, rect: QRectF, arrow: float, step_count: int
    ) -> QPainterPath:
        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()
        middle = rect.center().y()

        path = QPainterPath()
        path.moveTo(left, top)

        path.lineTo(right - arrow, top)
        path.lineTo(right, middle)
        path.lineTo(right - arrow, bottom)

        path.lineTo(left, bottom)
        path.lineTo(left + arrow, middle)
        path.closeSubpath()
        return path

    def _colors_for_step(self, index: int) -> tuple[QColor, QColor, QColor]:
        if index not in self._enabled_steps:
            return (
                self._theme.named_color("wizard_step_disabled_fill"),
                self._theme.named_color("wizard_step_disabled_border"),
                self._theme.named_color("wizard_step_disabled_text"),
            )
        if index == self._current_step_index:
            return (
                self._theme.named_color("wizard_step_current_fill"),
                self._theme.named_color("wizard_step_current_border"),
                self._theme.named_color("wizard_step_current_text"),
            )
        if index in self._completed_steps:
            return (
                self._theme.named_color("wizard_step_completed_fill"),
                self._theme.named_color("wizard_step_completed_border"),
                self._theme.named_color("wizard_step_completed_text"),
            )
        return (
            self._theme.named_color("wizard_step_pending_fill"),
            self._theme.named_color("wizard_step_pending_border"),
            self._theme.named_color("wizard_step_pending_text"),
        )
