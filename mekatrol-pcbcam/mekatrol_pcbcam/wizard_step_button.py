from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPainterPath, QPen
from PySide6.QtWidgets import QPushButton

from .theme import AppTheme


class WizardStepButton(QPushButton):
    def __init__(
        self,
        step_number: int,
        title: str,
        theme: AppTheme,
        *,
        is_first: bool,
        is_last: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.theme = theme
        self.step_number = step_number
        self.title = title
        self.is_first = is_first
        self.is_last = is_last
        self.is_current = False
        self.is_completed = False
        self.setMinimumHeight(52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(False)
        self.setFlat(True)
        self.setStyleSheet("background: transparent; border: 0;")

    def set_step_state(
        self,
        *,
        is_current: bool,
        is_completed: bool,
        is_enabled: bool,
    ) -> None:
        self.is_current = is_current
        self.is_completed = is_completed
        self.setEnabled(is_enabled)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 3, -2, -3)
        tip = min(26.0, rect.width() * 0.18)
        notch = 0.0 if self.is_first else tip
        path = QPainterPath()

        start_x = rect.left() if self.is_first else rect.left() + notch
        path.moveTo(start_x, rect.top())
        right_top_x = rect.right() if self.is_last else rect.right() - tip
        path.lineTo(right_top_x, rect.top())
        if not self.is_last:
            path.lineTo(rect.right(), rect.center().y())
        path.lineTo(rect.right() if self.is_last else rect.right() - tip, rect.bottom())
        path.lineTo(start_x, rect.bottom())
        if not self.is_first:
            path.lineTo(rect.left(), rect.center().y())
        path.closeSubpath()

        if not self.isEnabled():
            fill = self.theme.named_color("wizard_step_disabled_fill")
            border = self.theme.named_color("wizard_step_disabled_border")
            text = self.theme.named_color("wizard_step_disabled_text")
        elif self.is_current:
            fill = self.theme.named_color("wizard_step_current_fill")
            border = self.theme.named_color("wizard_step_current_border")
            text = self.theme.named_color("wizard_step_current_text")
        elif self.is_completed:
            fill = self.theme.named_color("wizard_step_completed_fill")
            border = self.theme.named_color("wizard_step_completed_border")
            text = self.theme.named_color("wizard_step_completed_text")
        else:
            fill = self.theme.named_color("wizard_step_pending_fill")
            border = self.theme.named_color("wizard_step_pending_border")
            text = self.theme.named_color("wizard_step_pending_text")

        painter.fillPath(path, fill)
        painter.setPen(QPen(border, 1.2))
        painter.drawPath(path)
        painter.setPen(text)
        label = f"{self.step_number}. {self.title}"
        left_padding = 18 if self.is_first else 30
        right_padding = 18 if self.is_last else 30
        painter.drawText(
            rect.adjusted(left_padding, 0, -right_padding, 0),
            Qt.AlignmentFlag.AlignCenter,
            label,
        )
        painter.end()
