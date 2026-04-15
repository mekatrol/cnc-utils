from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .theme_info import ThemeInfo


DEFAULT_THEME_FILE_NAME = "default_dark.yaml"


@dataclass
class AppTheme:
    theme_info: ThemeInfo = field(default_factory=ThemeInfo)
    main_window_muted_text: str = "#5b6571"
    wizard_step_disabled_fill: str = "#2a313a"
    wizard_step_disabled_border: str = "#414b56"
    wizard_step_disabled_text: str = "#7f8a96"
    wizard_step_current_fill: str = "#ff7a1a"
    wizard_step_current_border: str = "#ffae66"
    wizard_step_current_text: str = "#18120d"
    wizard_step_completed_fill: str = "#2f7d32"
    wizard_step_completed_border: str = "#66bb6a"
    wizard_step_completed_text: str = "#ecf7ed"
    wizard_step_pending_fill: str = "#233243"
    wizard_step_pending_border: str = "#425365"
    wizard_step_pending_text: str = "#c7d2de"
    pcb_preview_background: str = "#11151c"
    pcb_preview_grid_minor: str = "#1e2630"
    pcb_preview_grid_major: str = "#2d3947"
    pcb_preview_outline: str = "#ffe066"
    pcb_preview_drill: str = "#dfe7ef"
    pcb_preview_alignment: str = "#6ee7b7"
    pcb_preview_text: str = "#dfe7ef"
    pcb_preview_gerber_palette: list[str] = field(
        default_factory=lambda: [
            "#ff7f50",
            "#5dd39e",
            "#7aa2ff",
            "#f28482",
            "#ffd166",
            "#a78bfa",
        ]
    )
    mirror_preview_background: str = "#11151c"
    mirror_preview_board_outline: str = "#dfe7ef"
    mirror_preview_highlight: str = "#ff7a1a"
    mirror_preview_text: str = "#dfe7ef"
    toolpath_background: str = "#11151c"
    toolpath_grid_major: str = "#2f3945"
    toolpath_grid_minor: str = "#202831"
    toolpath_axis_x: str = "#d84d4d"
    toolpath_axis_y: str = "#3ecf8e"
    toolpath_axis_z: str = "#4ea1ff"
    toolpath_rapid: str = "#f2c94c"
    toolpath_cut: str = "#f97316"
    toolpath_text: str = "#dfe7ef"
    splash_background: str = "#16181d"
    splash_border: str = "#f3f5f7"
    splash_message_text: str = "#f3f5f7"
    splash_link: str = "#2f81f7"

    def color(self, value: str):
        from PySide6.QtGui import QColor

        return QColor(value)

    def named_color(self, field_name: str):
        from PySide6.QtGui import QColor

        return QColor(getattr(self, field_name))

    def gerber_palette(self):
        from PySide6.QtGui import QColor

        return [QColor(value) for value in self.pcb_preview_gerber_palette]


def default_theme() -> AppTheme:
    return AppTheme()


def load_theme(path: Path) -> tuple[AppTheme, list[str]]:
    warnings: list[str] = []
    loaded: object = {}

    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            warnings.append(
                f"Failed to read theme file {path}: {exc}. Using default theme values."
            )
            loaded = {}
        except yaml.YAMLError as exc:
            warnings.append(
                f"Failed to parse YAML theme file {path}: {exc}. Using default theme values."
            )
            loaded = {}

    if loaded is None:
        loaded = {}
    elif not isinstance(loaded, dict):
        warnings.append(
            f"Theme file {path} must contain a top-level mapping. Using default theme values."
        )
        loaded = {}

    defaults = default_theme()
    theme_info_data = loaded.get("theme_info", {})
    colors_data = loaded.get("colors", {})

    if not isinstance(theme_info_data, dict):
        warnings.append("theme_info: expected a mapping; using default theme metadata.")
        theme_info_data = {}
    if not isinstance(colors_data, dict):
        warnings.append("colors: expected a mapping; using default theme colors.")
        colors_data = {}

    theme = AppTheme(
        theme_info=ThemeInfo(
            name=_parse_text(
                theme_info_data.get("name"),
                defaults.theme_info.name,
                "theme_info.name",
                warnings,
            ),
            description=_parse_text(
                theme_info_data.get("description"),
                defaults.theme_info.description,
                "theme_info.description",
                warnings,
            ),
            author=_parse_text(
                theme_info_data.get("author"),
                defaults.theme_info.author,
                "theme_info.author",
                warnings,
            ),
        ),
        main_window_muted_text=_parse_color(
            colors_data.get("main_window_muted_text"),
            defaults.main_window_muted_text,
            "colors.main_window_muted_text",
            warnings,
        ),
        wizard_step_disabled_fill=_parse_color(
            colors_data.get("wizard_step_disabled_fill"),
            defaults.wizard_step_disabled_fill,
            "colors.wizard_step_disabled_fill",
            warnings,
        ),
        wizard_step_disabled_border=_parse_color(
            colors_data.get("wizard_step_disabled_border"),
            defaults.wizard_step_disabled_border,
            "colors.wizard_step_disabled_border",
            warnings,
        ),
        wizard_step_disabled_text=_parse_color(
            colors_data.get("wizard_step_disabled_text"),
            defaults.wizard_step_disabled_text,
            "colors.wizard_step_disabled_text",
            warnings,
        ),
        wizard_step_current_fill=_parse_color(
            colors_data.get("wizard_step_current_fill"),
            defaults.wizard_step_current_fill,
            "colors.wizard_step_current_fill",
            warnings,
        ),
        wizard_step_current_border=_parse_color(
            colors_data.get("wizard_step_current_border"),
            defaults.wizard_step_current_border,
            "colors.wizard_step_current_border",
            warnings,
        ),
        wizard_step_current_text=_parse_color(
            colors_data.get("wizard_step_current_text"),
            defaults.wizard_step_current_text,
            "colors.wizard_step_current_text",
            warnings,
        ),
        wizard_step_completed_fill=_parse_color(
            colors_data.get("wizard_step_completed_fill"),
            defaults.wizard_step_completed_fill,
            "colors.wizard_step_completed_fill",
            warnings,
        ),
        wizard_step_completed_border=_parse_color(
            colors_data.get("wizard_step_completed_border"),
            defaults.wizard_step_completed_border,
            "colors.wizard_step_completed_border",
            warnings,
        ),
        wizard_step_completed_text=_parse_color(
            colors_data.get("wizard_step_completed_text"),
            defaults.wizard_step_completed_text,
            "colors.wizard_step_completed_text",
            warnings,
        ),
        wizard_step_pending_fill=_parse_color(
            colors_data.get("wizard_step_pending_fill"),
            defaults.wizard_step_pending_fill,
            "colors.wizard_step_pending_fill",
            warnings,
        ),
        wizard_step_pending_border=_parse_color(
            colors_data.get("wizard_step_pending_border"),
            defaults.wizard_step_pending_border,
            "colors.wizard_step_pending_border",
            warnings,
        ),
        wizard_step_pending_text=_parse_color(
            colors_data.get("wizard_step_pending_text"),
            defaults.wizard_step_pending_text,
            "colors.wizard_step_pending_text",
            warnings,
        ),
        pcb_preview_background=_parse_color(
            colors_data.get("pcb_preview_background"),
            defaults.pcb_preview_background,
            "colors.pcb_preview_background",
            warnings,
        ),
        pcb_preview_grid_minor=_parse_color(
            colors_data.get("pcb_preview_grid_minor"),
            defaults.pcb_preview_grid_minor,
            "colors.pcb_preview_grid_minor",
            warnings,
        ),
        pcb_preview_grid_major=_parse_color(
            colors_data.get("pcb_preview_grid_major"),
            defaults.pcb_preview_grid_major,
            "colors.pcb_preview_grid_major",
            warnings,
        ),
        pcb_preview_outline=_parse_color(
            colors_data.get("pcb_preview_outline"),
            defaults.pcb_preview_outline,
            "colors.pcb_preview_outline",
            warnings,
        ),
        pcb_preview_drill=_parse_color(
            colors_data.get("pcb_preview_drill"),
            defaults.pcb_preview_drill,
            "colors.pcb_preview_drill",
            warnings,
        ),
        pcb_preview_alignment=_parse_color(
            colors_data.get("pcb_preview_alignment"),
            defaults.pcb_preview_alignment,
            "colors.pcb_preview_alignment",
            warnings,
        ),
        pcb_preview_text=_parse_color(
            colors_data.get("pcb_preview_text"),
            defaults.pcb_preview_text,
            "colors.pcb_preview_text",
            warnings,
        ),
        pcb_preview_gerber_palette=_parse_palette(
            colors_data.get("pcb_preview_gerber_palette"),
            defaults.pcb_preview_gerber_palette,
            "colors.pcb_preview_gerber_palette",
            warnings,
        ),
        mirror_preview_background=_parse_color(
            colors_data.get("mirror_preview_background"),
            defaults.mirror_preview_background,
            "colors.mirror_preview_background",
            warnings,
        ),
        mirror_preview_board_outline=_parse_color(
            colors_data.get("mirror_preview_board_outline"),
            defaults.mirror_preview_board_outline,
            "colors.mirror_preview_board_outline",
            warnings,
        ),
        mirror_preview_highlight=_parse_color(
            colors_data.get("mirror_preview_highlight"),
            defaults.mirror_preview_highlight,
            "colors.mirror_preview_highlight",
            warnings,
        ),
        mirror_preview_text=_parse_color(
            colors_data.get("mirror_preview_text"),
            defaults.mirror_preview_text,
            "colors.mirror_preview_text",
            warnings,
        ),
        toolpath_background=_parse_color(
            colors_data.get("toolpath_background"),
            defaults.toolpath_background,
            "colors.toolpath_background",
            warnings,
        ),
        toolpath_grid_major=_parse_color(
            colors_data.get("toolpath_grid_major"),
            defaults.toolpath_grid_major,
            "colors.toolpath_grid_major",
            warnings,
        ),
        toolpath_grid_minor=_parse_color(
            colors_data.get("toolpath_grid_minor"),
            defaults.toolpath_grid_minor,
            "colors.toolpath_grid_minor",
            warnings,
        ),
        toolpath_axis_x=_parse_color(
            colors_data.get("toolpath_axis_x"),
            defaults.toolpath_axis_x,
            "colors.toolpath_axis_x",
            warnings,
        ),
        toolpath_axis_y=_parse_color(
            colors_data.get("toolpath_axis_y"),
            defaults.toolpath_axis_y,
            "colors.toolpath_axis_y",
            warnings,
        ),
        toolpath_axis_z=_parse_color(
            colors_data.get("toolpath_axis_z"),
            defaults.toolpath_axis_z,
            "colors.toolpath_axis_z",
            warnings,
        ),
        toolpath_rapid=_parse_color(
            colors_data.get("toolpath_rapid"),
            defaults.toolpath_rapid,
            "colors.toolpath_rapid",
            warnings,
        ),
        toolpath_cut=_parse_color(
            colors_data.get("toolpath_cut"),
            defaults.toolpath_cut,
            "colors.toolpath_cut",
            warnings,
        ),
        toolpath_text=_parse_color(
            colors_data.get("toolpath_text"),
            defaults.toolpath_text,
            "colors.toolpath_text",
            warnings,
        ),
        splash_background=_parse_color(
            colors_data.get("splash_background"),
            defaults.splash_background,
            "colors.splash_background",
            warnings,
        ),
        splash_border=_parse_color(
            colors_data.get("splash_border"),
            defaults.splash_border,
            "colors.splash_border",
            warnings,
        ),
        splash_message_text=_parse_color(
            colors_data.get("splash_message_text"),
            defaults.splash_message_text,
            "colors.splash_message_text",
            warnings,
        ),
        splash_link=_parse_color(
            colors_data.get("splash_link"),
            defaults.splash_link,
            "colors.splash_link",
            warnings,
        ),
    )
    return theme, warnings


def save_theme(path: Path, theme: AppTheme) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# mekatrol-pcbcam theme file.",
            "# Share this file with other users by copying it into the themes folder.",
            "theme_info:",
            "  # Human-readable theme name shown to users.",
            f"  name: {_yaml_scalar(theme.theme_info.name)}",
            "  # Short explanation of the theme's overall look and purpose.",
            f"  description: {_yaml_scalar(theme.theme_info.description)}",
            "  # Theme author or source.",
            f"  author: {_yaml_scalar(theme.theme_info.author)}",
            "",
            "colors:",
            "  # Secondary text used for explanatory hints and muted labels in the main window.",
            f"  main_window_muted_text: {_yaml_scalar(theme.main_window_muted_text)}",
            "  # Wizard step background when a step is locked or unavailable.",
            f"  wizard_step_disabled_fill: {_yaml_scalar(theme.wizard_step_disabled_fill)}",
            "  # Wizard step outline when a step is locked or unavailable.",
            f"  wizard_step_disabled_border: {_yaml_scalar(theme.wizard_step_disabled_border)}",
            "  # Wizard step label color when a step is locked or unavailable.",
            f"  wizard_step_disabled_text: {_yaml_scalar(theme.wizard_step_disabled_text)}",
            "  # Wizard step background for the stage the user is currently editing.",
            f"  wizard_step_current_fill: {_yaml_scalar(theme.wizard_step_current_fill)}",
            "  # Wizard step outline for the current stage.",
            f"  wizard_step_current_border: {_yaml_scalar(theme.wizard_step_current_border)}",
            "  # Wizard step label color for the current stage.",
            f"  wizard_step_current_text: {_yaml_scalar(theme.wizard_step_current_text)}",
            "  # Wizard step background for stages that are already complete.",
            f"  wizard_step_completed_fill: {_yaml_scalar(theme.wizard_step_completed_fill)}",
            "  # Wizard step outline for completed stages.",
            f"  wizard_step_completed_border: {_yaml_scalar(theme.wizard_step_completed_border)}",
            "  # Wizard step label color for completed stages.",
            f"  wizard_step_completed_text: {_yaml_scalar(theme.wizard_step_completed_text)}",
            "  # Wizard step background for available stages that are not current or complete.",
            f"  wizard_step_pending_fill: {_yaml_scalar(theme.wizard_step_pending_fill)}",
            "  # Wizard step outline for available stages that are not current or complete.",
            f"  wizard_step_pending_border: {_yaml_scalar(theme.wizard_step_pending_border)}",
            "  # Wizard step label color for available stages that are not current or complete.",
            f"  wizard_step_pending_text: {_yaml_scalar(theme.wizard_step_pending_text)}",
            "  # Background for the 2D PCB preview canvas.",
            f"  pcb_preview_background: {_yaml_scalar(theme.pcb_preview_background)}",
            "  # Fine grid lines in the 2D PCB preview.",
            f"  pcb_preview_grid_minor: {_yaml_scalar(theme.pcb_preview_grid_minor)}",
            "  # Major grid lines in the 2D PCB preview.",
            f"  pcb_preview_grid_major: {_yaml_scalar(theme.pcb_preview_grid_major)}",
            "  # Board-outline stroke in the 2D PCB preview.",
            f"  pcb_preview_outline: {_yaml_scalar(theme.pcb_preview_outline)}",
            "  # Drill-hole stroke color in the 2D PCB preview.",
            f"  pcb_preview_drill: {_yaml_scalar(theme.pcb_preview_drill)}",
            "  # Alignment-hole highlight color in the 2D PCB preview.",
            f"  pcb_preview_alignment: {_yaml_scalar(theme.pcb_preview_alignment)}",
            "  # Overlay text color in the 2D PCB preview.",
            f"  pcb_preview_text: {_yaml_scalar(theme.pcb_preview_text)}",
            "  # Rotating colors used to distinguish imported Gerber layers in the 2D PCB preview.",
            "  pcb_preview_gerber_palette:",
            *[
                f"    - {_yaml_scalar(value)}"
                for value in theme.pcb_preview_gerber_palette
            ],
            "  # Background for the mirror-setup helper diagram.",
            f"  mirror_preview_background: {_yaml_scalar(theme.mirror_preview_background)}",
            "  # Board rectangle stroke in the mirror-setup helper diagram.",
            f"  mirror_preview_board_outline: {_yaml_scalar(theme.mirror_preview_board_outline)}",
            "  # Active mirror-axis line and arrow color in the mirror-setup helper diagram.",
            f"  mirror_preview_highlight: {_yaml_scalar(theme.mirror_preview_highlight)}",
            "  # Label text under the mirror-setup helper diagram.",
            f"  mirror_preview_text: {_yaml_scalar(theme.mirror_preview_text)}",
            "  # Background for the NC toolpath preview.",
            f"  toolpath_background: {_yaml_scalar(theme.toolpath_background)}",
            "  # Major grid lines in the NC toolpath preview.",
            f"  toolpath_grid_major: {_yaml_scalar(theme.toolpath_grid_major)}",
            "  # Minor grid lines in the NC toolpath preview.",
            f"  toolpath_grid_minor: {_yaml_scalar(theme.toolpath_grid_minor)}",
            "  # X-axis color in the NC toolpath preview.",
            f"  toolpath_axis_x: {_yaml_scalar(theme.toolpath_axis_x)}",
            "  # Y-axis color in the NC toolpath preview.",
            f"  toolpath_axis_y: {_yaml_scalar(theme.toolpath_axis_y)}",
            "  # Z-axis color in the NC toolpath preview.",
            f"  toolpath_axis_z: {_yaml_scalar(theme.toolpath_axis_z)}",
            "  # Rapid-travel motion color in the NC toolpath preview.",
            f"  toolpath_rapid: {_yaml_scalar(theme.toolpath_rapid)}",
            "  # Cutting motion color in the NC toolpath preview.",
            f"  toolpath_cut: {_yaml_scalar(theme.toolpath_cut)}",
            "  # Overlay text color in the NC toolpath preview.",
            f"  toolpath_text: {_yaml_scalar(theme.toolpath_text)}",
            "  # Splash window background behind the artwork.",
            f"  splash_background: {_yaml_scalar(theme.splash_background)}",
            "  # Splash window border line.",
            f"  splash_border: {_yaml_scalar(theme.splash_border)}",
            "  # Splash status message and metadata text color.",
            f"  splash_message_text: {_yaml_scalar(theme.splash_message_text)}",
            "  # Splash website link color.",
            f"  splash_link: {_yaml_scalar(theme.splash_link)}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def ensure_default_theme_file(themes_directory: Path) -> Path:
    path = themes_directory / DEFAULT_THEME_FILE_NAME
    themes_directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    save_theme(path, default_theme())
    return path


def _parse_text(
    value: object,
    default: str,
    field_name: str,
    warnings: list[str],
) -> str:
    if not isinstance(value, str):
        if value is not None:
            warnings.append(
                f"{field_name}: expected a string, got {value!r}; using {default!r}."
            )
        return default
    normalized = value.strip()
    if normalized:
        return normalized
    warnings.append(f"{field_name}: value is empty; using {default!r}.")
    return default


def _parse_color(
    value: object,
    default: str,
    field_name: str,
    warnings: list[str],
) -> str:
    if not isinstance(value, str):
        if value is not None:
            warnings.append(
                f"{field_name}: expected a color string, got {value!r}; using {default!r}."
            )
        return default
    normalized = value.strip()
    if _is_valid_color_string(normalized):
        return normalized
    warnings.append(
        f"{field_name}: invalid color {value!r}; using {default!r}."
    )
    return default


def _parse_palette(
    value: object,
    default: list[str],
    field_name: str,
    warnings: list[str],
) -> list[str]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        warnings.append(
            f"{field_name}: expected a list of color strings; using default palette."
        )
        return list(default)

    parsed: list[str] = []
    for index, item in enumerate(value):
        parsed.append(
            _parse_color(
                item,
                default[index] if index < len(default) else default[-1],
                f"{field_name}[{index}]",
                warnings,
            )
        )

    if not parsed:
        warnings.append(f"{field_name}: palette is empty; using default palette.")
        return list(default)
    return parsed


def _yaml_scalar(value: object) -> str:
    dumped = yaml.safe_dump(value, default_flow_style=True).strip()
    return dumped if dumped != "..." else "null"


def _is_valid_color_string(value: str) -> bool:
    return bool(re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", value))
