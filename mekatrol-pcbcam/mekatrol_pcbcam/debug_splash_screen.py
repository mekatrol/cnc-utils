from __future__ import annotations

from PySide6.QtCore import QElapsedTimer, QEventLoop, QTimer, Qt
from PySide6.QtGui import QColor, QFontDatabase, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSplashScreen

from . import PROJECT_URL, __version__


class DebugSplashScreen(QSplashScreen):
    def __init__(self, pixmap: QPixmap) -> None:
        super().__init__(pixmap)
        self._debug_click_loop: QEventLoop | None = None
        self._minimum_visible_loop: QEventLoop | None = None
        self._border_width = 2
        self._source_pixmap = pixmap
        self._minimum_visible_ms = 0
        self._startup_complete = False
        self._dismiss_requested = False
        self._visible_timer = QElapsedTimer()
        self.setPixmap(self._composited_pixmap())
        self.setFont(self._metadata_font())

    def begin_startup_timing(self, minimum_visible_ms: int) -> None:
        self._minimum_visible_ms = max(0, minimum_visible_ms)
        self._startup_complete = False
        self._dismiss_requested = False
        self._visible_timer.start()

    def mark_startup_complete(self) -> None:
        self._startup_complete = True

    def wait_until_ready(self) -> None:
        if not self._startup_complete:
            return

        if self._dismiss_requested:
            return

        remaining_ms = self._remaining_visible_ms()
        if remaining_ms <= 0:
            return

        loop = QEventLoop()
        self._minimum_visible_loop = loop
        QTimer.singleShot(remaining_ms, loop.quit)
        loop.exec()
        self._minimum_visible_loop = None

    def wait_for_click(self) -> None:
        loop = QEventLoop()
        self._debug_click_loop = loop
        loop.exec()
        self._debug_click_loop = None

    def _remaining_visible_ms(self) -> int:
        if not self._visible_timer.isValid():
            return 0
        elapsed_ms = self._visible_timer.elapsed()
        return max(0, self._minimum_visible_ms - elapsed_ms)

    def show_status_message(self, message: str) -> None:
        self.showMessage(
            message,
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            color=self.message_color(),
        )

    def _border_color(self) -> QColor:
        window_color = self._theme_window_color()
        luminance = (
            0.2126 * window_color.redF()
            + 0.7152 * window_color.greenF()
            + 0.0722 * window_color.blueF()
        )
        return QColor("#f3f5f7") if luminance < 0.5 else QColor("#1f2328")

    def _background_color(self) -> QColor:
        window_color = self._theme_window_color()
        luminance = (
            0.2126 * window_color.redF()
            + 0.7152 * window_color.greenF()
            + 0.0722 * window_color.blueF()
        )
        return QColor("#16181d") if luminance < 0.5 else QColor("#f5f6f8")

    def _theme_window_color(self) -> QColor:
        return self.palette().window().color()

    def message_color(self) -> QColor:
        background = self._background_color()
        luminance = (
            0.2126 * background.redF()
            + 0.7152 * background.greenF()
            + 0.0722 * background.blueF()
        )
        return QColor("#f3f5f7") if luminance < 0.5 else QColor("#1f2328")

    def _metadata_font(self):
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setPointSize(max(16, fixed_font.pointSize()))
        return fixed_font

    def _composited_pixmap(self) -> QPixmap:
        composed = QPixmap(
            (self._source_pixmap.width() * 3) // 2,
            (self._source_pixmap.height() * 3) // 2,
        )
        composed.fill(self._background_color())

        painter = QPainter(composed)
        painter.drawPixmap(0, 0, self._source_pixmap)
        painter.end()
        return composed

    def drawContents(self, painter: QPainter) -> None:
        super().drawContents(painter)
        painter.save()
        self._draw_metadata(painter)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(self._border_color(), self._border_width))
        inset = self._border_width // 2
        painter.drawRect(self.rect().adjusted(inset, inset, -inset - 1, -inset - 1))
        painter.restore()

    def _draw_metadata(self, painter: QPainter) -> None:
        metadata_lines = [
            f"{'Version':>8}: {__version__}",
            f"{'Website':>8}: {PROJECT_URL}",
        ]
        painter.setFont(self._metadata_font())
        painter.setPen(self.message_color())

        metrics = painter.fontMetrics()
        line_height = metrics.height()
        block_width = max(metrics.horizontalAdvance(line) for line in metadata_lines)
        block_height = line_height * len(metadata_lines)
        x_margin = 150
        y_margin = 50
        x = self.rect().right() - block_width - x_margin
        y = self.rect().bottom() - block_height - y_margin - 28

        for index, line in enumerate(metadata_lines):
            painter.drawText(x, y + ((index + 1) * line_height), line)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._dismiss_requested = True
        if self._debug_click_loop is not None:
            self._debug_click_loop.quit()
            event.accept()
            return
        if self._startup_complete and self._minimum_visible_loop is not None:
            self._minimum_visible_loop.quit()
        event.accept()
