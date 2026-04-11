from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtGui import QColor, QFontDatabase, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from . import PROJECT_URL, __version__
from .main_window import MainWindow


DEBUG_SPLASH_ENV = "MEKATROL_PCBCAM_DEBUG_SPLASH"


class DebugSplashScreen(QSplashScreen):
    def __init__(self, pixmap: QPixmap) -> None:
        super().__init__(pixmap)
        self._debug_click_loop: QEventLoop | None = None
        self._border_width = 2
        self._source_pixmap = pixmap
        self.setPixmap(self._composited_pixmap())

    def wait_for_click(self) -> None:
        loop = QEventLoop()
        self._debug_click_loop = loop
        loop.exec()
        self._debug_click_loop = None

    def show_status_message(self, message: str) -> None:
        self.showMessage(
            message,
            alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            color=self.message_color(),
        )

    def _border_color(self) -> QColor:
        # Pick a border that contrasts with the current desktop theme rather
        # than the splash image, so the splash bounds stay readable.
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

    def _composited_pixmap(self) -> QPixmap:
        composed = QPixmap(self._source_pixmap.size())
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
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setPointSize(max(16, fixed_font.pointSize()))
        painter.setFont(fixed_font)
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
        # Debug-only behavior: when enabled, a click on the splash screen
        # releases the temporary event loop so the app can continue startup.
        if self._debug_click_loop is not None:
            self._debug_click_loop.quit()
            event.accept()
            return
        super().mousePressEvent(event)


def _asset_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / Path(*parts)


def _debug_hold_splash_enabled() -> bool:
    return os.environ.get(DEBUG_SPLASH_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("mekatrol-pcbcam")
    app.setOrganizationName("Mekatrol")

    splash_path = _asset_path("splash.png")
    pixmap = QPixmap(str(splash_path))
    splash = DebugSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    splash.setStyleSheet("font-weight: 600;")
    splash.show()
    app.processEvents()

    for message in (
        "Starting PCBCAM viewer...",
        "Preparing Qt workspace...",
        "Initialising 3D viewport...",
    ):
        splash.show_status_message(message)
        app.processEvents()
        time.sleep(0.12)

    if _debug_hold_splash_enabled():
        # This path is intentionally debug-only. It keeps the splash visible
        # until you click it, which makes layout/styling inspection easier.
        splash.show_status_message(
            f"Debug splash hold enabled. Click splash to continue. ({DEBUG_SPLASH_ENV}=1)"
        )
        app.processEvents()
        splash.wait_for_click()

    window = MainWindow()
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser()
        if candidate.exists():
            window.load_file(str(candidate))
    window.show()
    splash.finish(window)
    return app.exec()
