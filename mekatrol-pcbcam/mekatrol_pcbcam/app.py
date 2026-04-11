from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, QSettings, Qt
from PySide6.QtGui import QColor, QFontDatabase, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import QApplication, QSplashScreen, QWidget

from . import PROJECT_URL, __version__
from .main_window import MainWindow


DEBUG_SPLASH_ENV = "MEKATROL_PCBCAM_DEBUG_SPLASH"
LAST_SCREEN_KEY = "ui/last_screen_name"
WINDOW_STATE_KEY = "ui/window_state"
WINDOW_X_KEY = "ui/window_x"
WINDOW_Y_KEY = "ui/window_y"
WINDOW_WIDTH_KEY = "ui/window_width"
WINDOW_HEIGHT_KEY = "ui/window_height"


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


def _settings() -> QSettings:
    return QSettings()


def _resolve_startup_screen(app: QApplication) -> QScreen:
    settings = _settings()
    saved_name = settings.value(LAST_SCREEN_KEY, "", type=str).strip()

    if saved_name:
        for screen in app.screens():
            if screen.name() == saved_name:
                return screen

    primary = app.primaryScreen()
    if primary is not None:
        return primary

    screens = app.screens()
    if not screens:
        raise RuntimeError("No screens are available for application startup.")
    return screens[0]


def _screen_for_widget(widget: QWidget) -> QScreen | None:
    handle = widget.windowHandle()
    if handle is not None and handle.screen() is not None:
        return handle.screen()
    return widget.screen()


def _save_last_screen(widget: QWidget) -> None:
    screen = _screen_for_widget(widget)
    if screen is None:
        return
    _settings().setValue(LAST_SCREEN_KEY, screen.name())


def _center_widget_on_screen(widget: QWidget, screen: QScreen) -> None:
    available = screen.availableGeometry()
    rect = widget.frameGeometry()
    rect.moveCenter(available.center())
    top_left = rect.topLeft()
    top_left.setX(max(available.left(), min(top_left.x(), available.right() - rect.width() + 1)))
    top_left.setY(max(available.top(), min(top_left.y(), available.bottom() - rect.height() + 1)))
    widget.move(top_left)


def _clamp_window_to_screen(widget: QWidget, screen: QScreen) -> None:
    available = screen.availableGeometry()
    width = min(widget.width(), available.width())
    height = min(widget.height(), available.height())
    widget.resize(width, height)

    x = max(available.left(), min(widget.x(), available.right() - width + 1))
    y = max(available.top(), min(widget.y(), available.bottom() - height + 1))
    widget.move(x, y)


def _apply_saved_window_placement(window: MainWindow, startup_screen: QScreen) -> None:
    settings = _settings()
    saved_state = settings.value(WINDOW_STATE_KEY, "normal", type=str).strip().lower()

    if saved_state == "normal":
        width = settings.value(WINDOW_WIDTH_KEY, 1280, type=int)
        height = settings.value(WINDOW_HEIGHT_KEY, 840, type=int)
        x = settings.value(WINDOW_X_KEY, None)
        y = settings.value(WINDOW_Y_KEY, None)

        window.resize(max(640, width), max(480, height))
        if x is not None and y is not None:
            window.move(int(x), int(y))
            _clamp_window_to_screen(window, startup_screen)
        else:
            _center_widget_on_screen(window, startup_screen)
        return

    _center_widget_on_screen(window, startup_screen)


def _save_window_placement(window: MainWindow) -> None:
    if window.isMinimized():
        return

    settings = _settings()
    _save_last_screen(window)

    if window.isMaximized():
        settings.setValue(WINDOW_STATE_KEY, "maximized")
        return

    settings.setValue(WINDOW_STATE_KEY, "normal")
    settings.setValue(WINDOW_X_KEY, window.x())
    settings.setValue(WINDOW_Y_KEY, window.y())
    settings.setValue(WINDOW_WIDTH_KEY, window.width())
    settings.setValue(WINDOW_HEIGHT_KEY, window.height())


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("mekatrol-pcbcam")
    app.setOrganizationName("Mekatrol")
    startup_screen = _resolve_startup_screen(app)

    splash_path = _asset_path("splash.png")
    pixmap = QPixmap(str(splash_path))
    splash = DebugSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    splash.setStyleSheet("font-weight: 600;")
    _center_widget_on_screen(splash, startup_screen)
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
    _apply_saved_window_placement(window, startup_screen)
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser()
        if candidate.exists():
            window.load_file(str(candidate))
    app.aboutToQuit.connect(lambda: _save_window_placement(window))
    if _settings().value(WINDOW_STATE_KEY, "normal", type=str).strip().lower() == "maximized":
        window.showMaximized()
    else:
        window.show()
    splash.finish(window)
    return app.exec()
