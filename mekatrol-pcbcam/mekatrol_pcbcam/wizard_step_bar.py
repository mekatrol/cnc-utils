from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class WizardStepBar(QWidget):
    step_selected = Signal(int)

    def __init__(self, titles: list[str], parent=None) -> None:
        super().__init__(parent)
        self._titles = titles
        self._current_step_index = 0
        self._completed_steps: set[int] = set()
        self._enabled_steps: set[int] = {0}
        self._step_paths: list[QPainterPath] = []
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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

        rect = self.rect().adjusted(0, 4, 0, -4)
        step_count = max(len(self._titles), 1)
        arrow = min(24.0, rect.width() / max(step_count * 5.0, 1.0))
        base_width = rect.width() / step_count

        self._step_paths = []
        for index, title in enumerate(self._titles):
            left = rect.left() + (index * base_width)
            right = rect.left() + ((index + 1) * base_width)
            step_rect = QRectF(left, rect.top(), right - left, rect.height())
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
            painter.drawText(
                text_rect, Qt.AlignmentFlag.AlignCenter, f"{index + 1}. {title}"
            )

        painter.end()

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
            return QColor("#2a313a"), QColor("#414b56"), QColor("#7f8a96")
        if index == self._current_step_index:
            return QColor("#ff7a1a"), QColor("#ffae66"), QColor("#18120d")
        if index in self._completed_steps:
            return QColor("#1ccfd0"), QColor("#7ee7e8"), QColor("#0b1d20")
        return QColor("#233243"), QColor("#425365"), QColor("#c7d2de")
