from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from PySide6.QtCore import QStandardPaths, Qt
from PySide6.QtGui import QPixmap, QScreen
from PySide6.QtWidgets import QApplication, QWidget

from .app_config import AppConfig
from .app_constants import (
    APPLICATION_NAME,
    CONFIG_FILE_NAME,
    DEBUG_SPLASH_ENV,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_SPLASH_MINIMUM_VISIBLE_MS,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    MINIMUM_WINDOW_HEIGHT,
    MINIMUM_WINDOW_WIDTH,
    ORGANIZATION_NAME,
    VALID_LOG_LEVELS,
    VALID_WINDOW_STATES,
)
from .debug_splash_screen import DebugSplashScreen
from .diagnostics import InMemoryLogHandler, get_log_tracker
from .file_locations import FileLocations
from .logging_config import LoggingConfig
from .main_window import MainWindow
from .ui_save_state import UiSaveState


logger = logging.getLogger(__name__)


def _describe_value(value: object) -> str:
    return repr(value)


def _append_config_warning(
    warnings: list[str], field_name: str, reason: str, default: object
) -> None:
    warnings.append(f"{field_name}: {reason}; using {default!r}.")

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


def _application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.name and str(argv0) not in {"", "-c"}:
        return argv0.resolve().parent
    return Path(__file__).resolve().parent


def _default_log_path() -> str:
    return f"logs/{APPLICATION_NAME}.log"


def _resolve_log_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return _application_directory() / candidate


def _parse_int(
    value: object,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    field_name: str | None = None,
    warnings: list[str] | None = None,
) -> int:
    if isinstance(value, bool):
        if field_name is not None and warnings is not None:
            _append_config_warning(
                warnings,
                field_name,
                f"expected an integer but got boolean {_describe_value(value)}",
                default,
            )
        return default

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        if field_name is not None and warnings is not None and value is not None:
            _append_config_warning(
                warnings,
                field_name,
                f"could not parse integer from {_describe_value(value)}",
                default,
            )
        return default

    if minimum is not None and parsed < minimum:
        if field_name is not None and warnings is not None:
            _append_config_warning(
                warnings,
                field_name,
                f"value {parsed} is below minimum {minimum}",
                default,
            )
        return default
    if maximum is not None and parsed > maximum:
        if field_name is not None and warnings is not None:
            _append_config_warning(
                warnings,
                field_name,
                f"value {parsed} is above maximum {maximum}",
                default,
            )
        return default
    return parsed


def _parse_optional_int(
    value: object, *, field_name: str | None = None, warnings: list[str] | None = None
) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        if field_name is not None and warnings is not None:
            _append_config_warning(
                warnings,
                field_name,
                f"expected an integer or null but got boolean {_describe_value(value)}",
                None,
            )
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        if field_name is not None and warnings is not None:
            _append_config_warning(
                warnings,
                field_name,
                f"could not parse integer from {_describe_value(value)}",
                None,
            )
        return None


def _parse_string(
    value: object,
    default: str = "",
    *,
    field_name: str | None = None,
    warnings: list[str] | None = None,
) -> str:
    if not isinstance(value, str):
        if field_name is not None and warnings is not None and value is not None:
            _append_config_warning(
                warnings,
                field_name,
                f"expected a string but got {_describe_value(value)}",
                default,
            )
        return default
    return value.strip()


def _parse_window_state(value: object, *, warnings: list[str] | None = None) -> str:
    parsed = _parse_string(
        value,
        "normal",
        field_name="ui_save_state.window_state",
        warnings=warnings,
    ).lower()
    if parsed in VALID_WINDOW_STATES:
        return parsed
    if value is not None and warnings is not None:
        _append_config_warning(
            warnings,
            "ui_save_state.window_state",
            f"invalid value {_describe_value(value)}; expected one of {sorted(VALID_WINDOW_STATES)}",
            "normal",
        )
    return "normal"


def _parse_log_level(
    value: object,
    default: str = DEFAULT_LOG_LEVEL,
    *,
    field_name: str | None = None,
    warnings: list[str] | None = None,
) -> str:
    parsed = _parse_string(
        value,
        default,
        field_name=field_name,
        warnings=warnings,
    ).upper()
    if parsed in VALID_LOG_LEVELS:
        return parsed
    if value is not None and field_name is not None and warnings is not None:
        _append_config_warning(
            warnings,
            field_name,
            f"invalid log level {_describe_value(value)}; expected one of {sorted(VALID_LOG_LEVELS)}",
            default,
        )
    return default


def _parse_log_path(value: object, *, warnings: list[str] | None = None) -> str:
    parsed = _parse_string(value, field_name="logging.path", warnings=warnings)
    if parsed:
        return parsed
    if value not in (None, "") and warnings is not None:
        _append_config_warning(
            warnings,
            "logging.path",
            "log path was empty after parsing",
            _default_log_path(),
        )
    return _default_log_path()


def _parse_logger_levels(
    value: object, default_level: str, *, warnings: list[str] | None = None
) -> dict[str, str]:
    if not isinstance(value, dict):
        if value is not None and warnings is not None:
            _append_config_warning(
                warnings,
                "logging.loggers",
                f"expected a mapping but got {_describe_value(value)}",
                {},
            )
        return {}

    parsed: dict[str, str] = {}
    for logger_name, level in value.items():
        if not isinstance(logger_name, str):
            if warnings is not None:
                warnings.append(
                    "logging.loggers: skipped logger override with non-string name "
                    f"{_describe_value(logger_name)}."
                )
            continue
        normalized_name = logger_name.strip()
        if not normalized_name:
            if warnings is not None:
                warnings.append(
                    "logging.loggers: skipped logger override with empty logger name."
                )
            continue
        parsed[normalized_name] = _parse_log_level(
            level,
            default_level,
            field_name=f"logging.loggers.{normalized_name}",
            warnings=warnings,
        )
    return parsed


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _load_config() -> tuple[AppConfig, list[str]]:
    path = _config_path()
    data: object = {}
    warnings: list[str] = []

    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            warnings.append(
                f"Failed to read config file {path}: {exc}. Using default settings."
            )
            loaded = {}
        except yaml.YAMLError as exc:
            warnings.append(
                f"Failed to parse YAML config file {path}: {exc}. Using default settings."
            )
            loaded = {}
        if loaded is None:
            loaded = {}
        elif not isinstance(loaded, dict):
            warnings.append(
                f"Config file {path} must contain a top-level mapping. Using default settings."
            )
            loaded = {}
        data = loaded

    app_data = data.get("app", {}) if isinstance(data, dict) else {}
    logging_data = data.get("logging", {}) if isinstance(data, dict) else {}
    file_locations_data = data.get("file_locations", {}) if isinstance(data, dict) else {}
    ui_data = data.get("ui_save_state", {}) if isinstance(data, dict) else {}

    if not isinstance(app_data, dict):
        warnings.append("app: expected a mapping; using default app settings.")
        app_data = {}
    if not isinstance(logging_data, dict):
        warnings.append("logging: expected a mapping; using default logging settings.")
        logging_data = {}
    if not isinstance(file_locations_data, dict):
        warnings.append(
            "file_locations: expected a mapping; using default file location settings."
        )
        file_locations_data = {}
    if not isinstance(ui_data, dict):
        warnings.append("ui_save_state: expected a mapping; using default UI settings.")
        ui_data = {}

    logging_level = _parse_log_level(
        logging_data.get("level"),
        field_name="logging.level",
        warnings=warnings,
    )

    return AppConfig(
        splash_minimum_visible_ms=_parse_int(
            app_data.get("splash_minimum_visible_ms"),
            DEFAULT_SPLASH_MINIMUM_VISIBLE_MS,
            minimum=0,
            field_name="app.splash_minimum_visible_ms",
            warnings=warnings,
        ),
        logging=LoggingConfig(
            level=logging_level,
            path=_parse_log_path(logging_data.get("path"), warnings=warnings),
            max_bytes=_parse_int(
                logging_data.get("max_bytes"),
                DEFAULT_LOG_MAX_BYTES,
                minimum=1024,
                field_name="logging.max_bytes",
                warnings=warnings,
            ),
            backup_count=_parse_int(
                logging_data.get("backup_count"),
                DEFAULT_LOG_BACKUP_COUNT,
                minimum=1,
                field_name="logging.backup_count",
                warnings=warnings,
            ),
            loggers=_parse_logger_levels(
                logging_data.get("loggers"), logging_level, warnings=warnings
            ),
        ),
        file_locations=FileLocations(
            last_load_directory=_parse_string(
                file_locations_data.get("last_load_directory"),
                field_name="file_locations.last_load_directory",
                warnings=warnings,
            ),
            last_save_directory=_parse_string(
                file_locations_data.get("last_save_directory"),
                field_name="file_locations.last_save_directory",
                warnings=warnings,
            ),
        ),
        ui_save_state=UiSaveState(
            last_screen_name=_parse_string(
                ui_data.get("last_screen_name"),
                field_name="ui_save_state.last_screen_name",
                warnings=warnings,
            ),
            window_state=_parse_window_state(
                ui_data.get("window_state"), warnings=warnings
            ),
            window_x=_parse_optional_int(
                ui_data.get("window_x"),
                field_name="ui_save_state.window_x",
                warnings=warnings,
            ),
            window_y=_parse_optional_int(
                ui_data.get("window_y"),
                field_name="ui_save_state.window_y",
                warnings=warnings,
            ),
            window_width=_parse_int(
                ui_data.get("window_width"),
                DEFAULT_WINDOW_WIDTH,
                minimum=MINIMUM_WINDOW_WIDTH,
                field_name="ui_save_state.window_width",
                warnings=warnings,
            ),
            window_height=_parse_int(
                ui_data.get("window_height"),
                DEFAULT_WINDOW_HEIGHT,
                minimum=MINIMUM_WINDOW_HEIGHT,
                field_name="ui_save_state.window_height",
                warnings=warnings,
            ),
        ),
    ), warnings


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
            "logging:",
            "  # Default log level for the mekatrol_pcbcam application loggers.",
            "  # Common values: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
            f"  level: {_yaml_scalar(config.logging.level)}",
            "  # Log file path. Relative paths are resolved against the running application executable directory.",
            f"  path: {_yaml_scalar(config.logging.path)}",
            "  # Maximum size in bytes for each log file before rollover creates a new file.",
            f"  max_bytes: {config.logging.max_bytes}",
            "  # Number of rotated log files to keep alongside the active log file.",
            f"  backup_count: {config.logging.backup_count}",
            "  # Optional per-logger overrides. These names usually match Python module namespaces.",
            "  # Example: mekatrol_pcbcam.gcode_parser: DEBUG",
            *(
                ["  loggers: {}"]
                if not config.logging.loggers
                else [
                    "  loggers:",
                    *[
                        f"    {logger_name}: {_yaml_scalar(level)}"
                        for logger_name, level in sorted(config.logging.loggers.items())
                    ],
                ]
            ),
            "",
            "file_locations:",
            "  # Most recently used directory for file-open dialogs.",
            f"  last_load_directory: {_yaml_scalar(config.file_locations.last_load_directory)}",
            "",
            "  # Most recently used directory for file-save dialogs.",
            f"  last_save_directory: {_yaml_scalar(config.file_locations.last_save_directory)}",
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


def _configure_logging(config: AppConfig) -> Path:
    configured_path = _resolve_log_path(config.logging.path)
    fallback_path = _resolve_log_path(_default_log_path())
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    handler: logging.Handler
    active_path = configured_path
    try:
        configured_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            configured_path,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding="utf-8",
        )
    except OSError:
        active_path = fallback_path
        try:
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                fallback_path,
                maxBytes=config.logging.max_bytes,
                backupCount=config.logging.backup_count,
                encoding="utf-8",
            )
        except OSError:
            handler = logging.StreamHandler(sys.stderr)
            active_path = Path("<stderr>")

    handler.setFormatter(formatter)
    handler.setLevel(logging.NOTSET)
    memory_handler = InMemoryLogHandler(get_log_tracker())
    memory_handler.setLevel(logging.NOTSET)

    package_logger = logging.getLogger("mekatrol_pcbcam")
    for existing_handler in package_logger.handlers:
        existing_handler.close()
    package_logger.handlers.clear()
    package_logger.setLevel(config.logging.level)
    package_logger.addHandler(handler)
    package_logger.addHandler(memory_handler)
    package_logger.propagate = False

    for logger_name, level in config.logging.loggers.items():
        logging.getLogger(logger_name).setLevel(level)

    logger.info(
        "Logging configured: level=%s path=%s max_bytes=%d backup_count=%d",
        config.logging.level,
        active_path,
        config.logging.max_bytes,
        config.logging.backup_count,
    )
    return active_path


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
    get_log_tracker().clear()
    config, config_warnings = _load_config()
    active_log_path = _configure_logging(config)
    for warning in config_warnings:
        logger.warning("Config warning: %s", warning)
    _save_config(config)
    logger.info("Application startup beginning with config at %s", _config_path())
    logger.debug("Active log output path: %s", active_log_path)
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

    window = MainWindow(config)
    _apply_saved_window_placement(window, startup_screen, config)
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser()
        if candidate.exists():
            logger.info("Loading startup file argument: %s", candidate)
            window.load_file(str(candidate))
        else:
            logger.warning("Startup file argument does not exist: %s", candidate)
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
    logger.info("Application startup complete")
    return app.exec()
