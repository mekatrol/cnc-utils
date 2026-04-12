from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PySide6.QtCore import QElapsedTimer, QEventLoop, QStandardPaths, QTimer, Qt
from PySide6.QtGui import QColor, QFontDatabase, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import QApplication, QSplashScreen, QWidget

from . import PROJECT_URL, __version__
from .main_window import MainWindow


APPLICATION_NAME = "mekatrol-pcbcam"
ORGANIZATION_NAME = "Mekatrol"
DEBUG_SPLASH_ENV = "MEKATROL_PCBCAM_DEBUG_SPLASH"
CONFIG_FILE_NAME = f"{APPLICATION_NAME}.yaml"
DEFAULT_SPLASH_MINIMUM_VISIBLE_MS = 3000
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 840
MINIMUM_WINDOW_WIDTH = 640
MINIMUM_WINDOW_HEIGHT = 480
VALID_WINDOW_STATES = {"normal", "maximized"}


@dataclass
class UiSaveState:
    last_screen_name: str = ""
    window_state: str = "normal"
    window_x: int | None = None
    window_y: int | None = None
    window_width: int = DEFAULT_WINDOW_WIDTH
    window_height: int = DEFAULT_WINDOW_HEIGHT


@dataclass
class AppConfig:
    splash_minimum_visible_ms: int = DEFAULT_SPLASH_MINIMUM_VISIBLE_MS
    ui_save_state: UiSaveState = field(default_factory=UiSaveState)


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

    def _metadata_font(self):
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setPointSize(max(16, fixed_font.pointSize()))
        return fixed_font

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
        # A splash click is treated as a dismissal request. In debug mode it
        # releases the click-held loop. In normal mode it can also release the
        # minimum-visible timer, but only after startup has completed.
        self._dismiss_requested = True
        if self._debug_click_loop is not None:
            self._debug_click_loop.quit()
            event.accept()
            return
        if self._startup_complete and self._minimum_visible_loop is not None:
            self._minimum_visible_loop.quit()
        event.accept()


def _asset_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / Path(*parts)


def _debug_hold_splash_enabled() -> bool:
    return os.environ.get(DEBUG_SPLASH_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _config_path() -> Path:
    config_root = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericConfigLocation
    )
    base_path = Path(config_root) if config_root else Path.home() / ".config"
    return base_path / ORGANIZATION_NAME / CONFIG_FILE_NAME


def _parse_int(
    value: object,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        return default

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default

    if minimum is not None and parsed < minimum:
        return default
    if maximum is not None and parsed > maximum:
        return default
    return parsed


def _parse_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_string(value: object, default: str = "") -> str:
    if not isinstance(value, str):
        return default
    return value.strip()


def _parse_window_state(value: object) -> str:
    parsed = _parse_string(value, "normal").lower()
    if parsed in VALID_WINDOW_STATES:
        return parsed
    return "normal"


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _load_config() -> AppConfig:
    path = _config_path()
    data: object = {}

    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            loaded = {}
        data = loaded if isinstance(loaded, dict) else {}

    app_data = data.get("app", {}) if isinstance(data, dict) else {}
    ui_data = data.get("ui_save_state", {}) if isinstance(data, dict) else {}

    if not isinstance(app_data, dict):
        app_data = {}
    if not isinstance(ui_data, dict):
        ui_data = {}

    return AppConfig(
        splash_minimum_visible_ms=_parse_int(
            app_data.get("splash_minimum_visible_ms"),
            DEFAULT_SPLASH_MINIMUM_VISIBLE_MS,
            minimum=0,
        ),
        ui_save_state=UiSaveState(
            last_screen_name=_parse_string(ui_data.get("last_screen_name")),
            window_state=_parse_window_state(ui_data.get("window_state")),
            window_x=_parse_optional_int(ui_data.get("window_x")),
            window_y=_parse_optional_int(ui_data.get("window_y")),
            window_width=_parse_int(
                ui_data.get("window_width"),
                DEFAULT_WINDOW_WIDTH,
                minimum=MINIMUM_WINDOW_WIDTH,
            ),
            window_height=_parse_int(
                ui_data.get("window_height"),
                DEFAULT_WINDOW_HEIGHT,
                minimum=MINIMUM_WINDOW_HEIGHT,
            ),
        ),
    )


def _save_config(config: AppConfig) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# mekatrol-pcbcam configuration.",
            "app:",
            "  # Minimum time to keep the splash screen visible during startup, in milliseconds.",
            f"  splash_minimum_visible_ms: {config.splash_minimum_visible_ms}",
            "",
            "ui_save_state:",
            "  # Name of the last display used by the main window. Used to pick the startup screen.",
            f"  last_screen_name: {_yaml_scalar(config.ui_save_state.last_screen_name)}",
            "  # Main window mode restored on startup. Allowed values: normal, maximized.",
            f"  window_state: {_yaml_scalar(config.ui_save_state.window_state)}",
            "  # Left edge of the window in global screen coordinates when window_state is normal.",
            f"  window_x: {_yaml_scalar(config.ui_save_state.window_x)}",
            "  # Top edge of the window in global screen coordinates when window_state is normal.",
            f"  window_y: {_yaml_scalar(config.ui_save_state.window_y)}",
            "  # Window width in pixels when window_state is normal.",
            f"  window_width: {config.ui_save_state.window_width}",
            "  # Window height in pixels when window_state is normal.",
            f"  window_height: {config.ui_save_state.window_height}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _resolve_startup_screen(app: QApplication, config: AppConfig) -> QScreen:
    saved_name = config.ui_save_state.last_screen_name

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


def _save_last_screen(widget: QWidget, config: AppConfig) -> None:
    screen = _screen_for_widget(widget)
    if screen is None:
        return
    config.ui_save_state.last_screen_name = screen.name()


def _center_widget_on_screen(widget: QWidget, screen: QScreen) -> None:
    available = screen.availableGeometry()
    rect = widget.frameGeometry()
    rect.moveCenter(available.center())
    top_left = rect.topLeft()
    top_left.setX(
        max(available.left(), min(top_left.x(), available.right() - rect.width() + 1))
    )
    top_left.setY(
        max(available.top(), min(top_left.y(), available.bottom() - rect.height() + 1))
    )
    widget.move(top_left)


def _clamp_window_to_screen(widget: QWidget, screen: QScreen) -> None:
    available = screen.availableGeometry()
    width = min(widget.width(), available.width())
    height = min(widget.height(), available.height())
    widget.resize(width, height)

    x = max(available.left(), min(widget.x(), available.right() - width + 1))
    y = max(available.top(), min(widget.y(), available.bottom() - height + 1))
    widget.move(x, y)


def _apply_saved_window_placement(
    window: MainWindow, startup_screen: QScreen, config: AppConfig
) -> None:
    saved_state = config.ui_save_state.window_state

    if saved_state == "normal":
        width = config.ui_save_state.window_width
        height = config.ui_save_state.window_height
        x = config.ui_save_state.window_x
        y = config.ui_save_state.window_y

        window.resize(max(MINIMUM_WINDOW_WIDTH, width), max(MINIMUM_WINDOW_HEIGHT, height))
        if x is not None and y is not None:
            window.move(x, y)
            _clamp_window_to_screen(window, startup_screen)
        else:
            _center_widget_on_screen(window, startup_screen)
        return

    _center_widget_on_screen(window, startup_screen)


def _save_window_placement(window: MainWindow, config: AppConfig) -> None:
    if window.isMinimized():
        return

    _save_last_screen(window, config)

    if window.isMaximized():
        config.ui_save_state.window_state = "maximized"
        _save_config(config)
        return

    config.ui_save_state.window_state = "normal"
    config.ui_save_state.window_x = window.x()
    config.ui_save_state.window_y = window.y()
    config.ui_save_state.window_width = max(MINIMUM_WINDOW_WIDTH, window.width())
    config.ui_save_state.window_height = max(MINIMUM_WINDOW_HEIGHT, window.height())
    _save_config(config)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APPLICATION_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    config = _load_config()
    _save_config(config)
    startup_screen = _resolve_startup_screen(app, config)

    splash_path = _asset_path("splash.png")
    pixmap = QPixmap(str(splash_path))
    splash = DebugSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    splash.setStyleSheet("font-weight: 600;")
    _center_widget_on_screen(splash, startup_screen)
    splash.show()
    splash.begin_startup_timing(config.splash_minimum_visible_ms)
    app.processEvents()

    for message in ("Starting PCBCAM...",):
        splash.show_status_message(message)
        app.processEvents()

    if _debug_hold_splash_enabled():
        # This path is intentionally debug-only. It keeps the splash visible
        # until you click it, which makes layout/styling inspection easier.
        splash.show_status_message(
            f"Debug splash hold enabled. Click splash to continue. ({DEBUG_SPLASH_ENV}=1)"
        )
        app.processEvents()
        splash.wait_for_click()

    window = MainWindow()
    _apply_saved_window_placement(window, startup_screen, config)
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser()
        if candidate.exists():
            window.load_file(str(candidate))
    splash.mark_startup_complete()
    if not _debug_hold_splash_enabled():
        # Normal startup keeps the splash visible for at least the configured
        # minimum duration, but does not add extra delay after slow loads.
        splash.wait_until_ready()
    app.aboutToQuit.connect(lambda: _save_window_placement(window, config))
    if config.ui_save_state.window_state == "maximized":
        window.showMaximized()
    else:
        window.show()
    splash.finish(window)
    return app.exec()
