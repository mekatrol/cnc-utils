from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class MirrorPreviewWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._edge = ""
        self.setMinimumHeight(170)

    def set_edge(self, edge: str) -> None:
        self._edge = edge
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#11151c"))

        board_rect = self.rect().adjusted(34, 24, -34, -24)
        board_rect.setWidth(board_rect.width() * 0.72)
        board_rect.moveLeft((self.width() - board_rect.width()) * 0.5)
        painter.setPen(QPen(QColor("#dfe7ef"), 2))
        painter.drawRect(board_rect)

        label = "No mirror required"
        if self._edge:
            label = f"Mirror around {self._edge.replace('_', ' ')} edge"
            painter.setPen(QPen(QColor("#ff7a1a"), 4))
            if self._edge == "left":
                painter.drawLine(board_rect.topLeft(), board_rect.bottomLeft())
                self._draw_arrow(
                    painter,
                    board_rect.center() + QPointF(-36, 0),
                    board_rect.center() + QPointF(36, 0),
                )
            elif self._edge == "right":
                painter.drawLine(board_rect.topRight(), board_rect.bottomRight())
                self._draw_arrow(
                    painter,
                    board_rect.center() + QPointF(36, 0),
                    board_rect.center() + QPointF(-36, 0),
                )
            elif self._edge == "top":
                painter.drawLine(board_rect.topLeft(), board_rect.topRight())
                self._draw_arrow(
                    painter,
                    board_rect.center() + QPointF(0, -30),
                    board_rect.center() + QPointF(0, 30),
                )
            elif self._edge == "bottom":
                painter.drawLine(board_rect.bottomLeft(), board_rect.bottomRight())
                self._draw_arrow(
                    painter,
                    board_rect.center() + QPointF(0, 30),
                    board_rect.center() + QPointF(0, -30),
                )

        painter.setPen(QColor("#dfe7ef"))
        painter.drawText(
            self.rect().adjusted(0, 0, 0, -8),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            label,
        )
        painter.end()

    def _draw_arrow(self, painter: QPainter, start: QPointF, end: QPointF) -> None:
        painter.drawLine(start, end)
        direction = end - start
        if abs(direction.x()) > abs(direction.y()):
            sign = 1 if direction.x() >= 0 else -1
            head = QPolygonF(
                [
                    end,
                    end + QPointF(-12 * sign, -8),
                    end + QPointF(-12 * sign, 8),
                ]
            )
        else:
            sign = 1 if direction.y() >= 0 else -1
            head = QPolygonF(
                [
                    end,
                    end + QPointF(-8, -12 * sign),
                    end + QPointF(8, -12 * sign),
                ]
            )
        painter.setBrush(QColor("#ff7a1a"))
        painter.drawPolygon(head)
