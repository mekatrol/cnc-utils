from __future__ import annotations

import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QStandardPaths, Qt
from PySide6.QtGui import QCloseEvent, QDoubleValidator, QIntValidator, QScreen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .app_config import AppConfig
from .box_settings import BoxSettings
from .file_locations import FileLocations
from .gcode_generator import GcodeGenerator
from .layout_generator import LayoutGenerator
from .preview_widget import PreviewWidget
from .ui_save_state import UiSaveState

APPLICATION_NAME = "box-creator"
ORGANIZATION_NAME = "Mekatrol"
CONFIG_FILE_NAME = f"{APPLICATION_NAME}.json"
PROJECT_FILE_FILTER = (
    "Box Creator Project (*.boxcreator.json *.json);;JSON Files (*.json)"
)
MINIMUM_WINDOW_WIDTH = 620
MINIMUM_WINDOW_HEIGHT = 460


class MainWindow(QMainWindow):
    STEP_TITLES = ["Box", "Material", "Joints", "Tabs", "Preview", "Generate"]

    def __init__(
        self, config: AppConfig, save_config: Callable[[AppConfig], None]
    ) -> None:
        super().__init__()
        self.resize(1180, 760)
        self.config = config
        self._save_config = save_config
        self.settings = BoxSettings()
        self.layout_generator = LayoutGenerator()
        self.gcode_generator = GcodeGenerator()
        self.panels = self.layout_generator.generate(self.settings)
        self.current_step = 0
        self.inputs: dict[str, QLineEdit] = {}
        self.project_path: Path | None = None
        self.has_unsaved_changes = False

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)
        file_row = QHBoxLayout()
        new_button = QPushButton("New")
        new_button.clicked.connect(self._new_project)
        open_button = QPushButton("Open")
        open_button.clicked.connect(self._open_project)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_project)
        save_as_button = QPushButton("Save As")
        save_as_button.clicked.connect(self._save_project_as)
        file_row.addWidget(new_button)
        file_row.addWidget(open_button)
        file_row.addWidget(self.save_button)
        file_row.addWidget(save_as_button)
        file_row.addStretch(1)
        root_layout.addLayout(file_row)

        self.step_row = QHBoxLayout()
        self.step_buttons: list[QPushButton] = []
        for index, title in enumerate(self.STEP_TITLES):
            button = QPushButton(title)
            button.clicked.connect(
                lambda checked=False, step=index: self._set_step(step)
            )
            self.step_buttons.append(button)
            self.step_row.addWidget(button)
        root_layout.addLayout(self.step_row)

        body = QHBoxLayout()
        self.pages = QStackedWidget()
        self.pages.addWidget(self._box_page())
        self.pages.addWidget(self._material_page())
        self.pages.addWidget(self._joints_page())
        self.pages.addWidget(self._tabs_page())
        self.pages.addWidget(self._preview_page())
        self.pages.addWidget(self._generate_page())
        body.addWidget(self.pages, 0)

        self.preview = PreviewWidget()
        body.addWidget(self.preview, 1)
        root_layout.addLayout(body, 1)

        nav = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(lambda: self._set_step(self.current_step - 1))
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(lambda: self._set_step(self.current_step + 1))
        nav.addStretch(1)
        nav.addWidget(self.back_button)
        nav.addWidget(self.next_button)
        root_layout.addLayout(nav)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self._apply_style()
        self._update_window_title()
        self._set_step(0)
        self._refresh_model()

    def _box_page(self) -> QWidget:
        page = self._page("Box size")
        form = self._form(page)
        self._add_text(form, "Job name", "job_name")
        form.addRow("Box type", self._box_kind_picker())
        self._add_number(form, "Outside X width", "size_x")
        self._add_number(form, "Outside Y depth", "size_y")
        self._add_number(form, "Outside Z height", "size_z")
        return page

    def _material_page(self) -> QWidget:
        page = self._page("Material and cutter")
        form = self._form(page)
        self._add_number(form, "Material thickness", "material_thickness")
        self._add_number(form, "Stock X width", "stock_width")
        self._add_number(form, "Stock Y height", "stock_height")
        self._add_number(form, "Edge cut bit diameter", "bit_diameter")
        self._add_number(form, "Relief bit diameter", "relief_diameter")
        self._add_number(form, "Cut depth step", "cut_depth_step")
        return page

    def _joints_page(self) -> QWidget:
        page = self._page("Finger joints")
        form = self._form(page)
        self._add_number(form, "Finger width", "finger_width")
        note = QLabel(
            "Finger height is the material thickness. The generated panels alternate tabs and sockets."
        )
        note.setWordWrap(True)
        page.layout().addWidget(note)
        return page

    def _tabs_page(self) -> QWidget:
        page = self._page("Holding tabs")
        form = self._form(page)
        self.include_tabs = QCheckBox("Add tabs on outer panel edges")
        self.include_tabs.setChecked(self.settings.include_tabs)
        self.include_tabs.toggled.connect(self._refresh_model_from_user)
        form.addRow("", self.include_tabs)
        self._add_number(form, "Tab width", "tab_width")
        self._add_number(form, "Tab remaining height", "tab_height")
        return page

    def _preview_page(self) -> QWidget:
        page = self._page("Preview")
        self.preview_mode = QComboBox()
        self.preview_mode.addItem("Flat cutting layout", "flat")
        self.preview_mode.addItem("Assembled box", "assembled")
        self.preview_mode.currentIndexChanged.connect(self._refresh_preview)
        form = self._form(page)
        form.addRow("View", self.preview_mode)
        return page

    def _generate_page(self) -> QWidget:
        page = self._page("Generate NC")
        form = self._form(page)
        self._add_integer(form, "Feed rate", "feed_rate")
        self._add_integer(form, "Plunge rate", "plunge_rate")
        self._add_integer(form, "Spindle speed", "spindle_speed")
        self.output_path = QLineEdit(str(self._default_output_path()))
        self.output_path.editingFinished.connect(self._mark_project_dirty)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_output)
        row = QHBoxLayout()
        row.addWidget(self.output_path)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Output file", holder)
        generate = QPushButton("Generate NC")
        generate.clicked.connect(self._generate_nc)
        page.layout().addWidget(generate)
        return page

    def _page(self, title: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        heading = QLabel(title)
        heading.setObjectName("heading")
        layout.addWidget(heading)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)
        return page

    def _form(self, page: QWidget) -> QFormLayout:
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        page.layout().addLayout(form)
        return form

    def _box_kind_picker(self) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        group = QButtonGroup(holder)
        self.box_radio = QRadioButton("Box with lid")
        self.drawer_radio = QRadioButton("Drawer tray")
        self.box_radio.setChecked(True)
        group.addButton(self.box_radio)
        group.addButton(self.drawer_radio)
        self.box_radio.toggled.connect(self._refresh_model_from_user)
        self.drawer_radio.toggled.connect(self._refresh_model_from_user)
        layout.addWidget(self.box_radio)
        layout.addWidget(self.drawer_radio)
        return holder

    def _add_text(self, form: QFormLayout, label: str, field_name: str) -> None:
        edit = QLineEdit(str(getattr(self.settings, field_name)))
        edit.editingFinished.connect(self._refresh_model_from_user)
        self.inputs[field_name] = edit
        form.addRow(label, edit)

    def _add_number(self, form: QFormLayout, label: str, field_name: str) -> None:
        edit = QLineEdit(f"{getattr(self.settings, field_name):.3f}")
        validator = QDoubleValidator(0.001, 10000.0, 3, edit)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        edit.setValidator(validator)
        edit.editingFinished.connect(self._refresh_model_from_user)
        self.inputs[field_name] = edit
        form.addRow(f"{label} (mm)", edit)

    def _add_integer(self, form: QFormLayout, label: str, field_name: str) -> None:
        edit = QLineEdit(str(getattr(self.settings, field_name)))
        edit.setValidator(QIntValidator(1, 100000, edit))
        edit.editingFinished.connect(self._refresh_model_from_user)
        self.inputs[field_name] = edit
        form.addRow(label, edit)

    def _set_step(self, step: int) -> None:
        self.current_step = max(0, min(len(self.STEP_TITLES) - 1, step))
        self.pages.setCurrentIndex(self.current_step)
        for index, button in enumerate(self.step_buttons):
            button.setProperty("currentStep", index == self.current_step)
            button.style().unpolish(button)
            button.style().polish(button)
        self.back_button.setEnabled(self.current_step > 0)
        self.next_button.setEnabled(self.current_step < len(self.STEP_TITLES) - 1)
        self._refresh_model()

    def _refresh_model_from_user(self) -> None:
        self._refresh_model()
        self._mark_project_dirty()

    def _refresh_model(self) -> None:
        try:
            for field_name, edit in self.inputs.items():
                value = edit.text().strip()
                current = getattr(self.settings, field_name)
                if isinstance(current, int):
                    setattr(self.settings, field_name, int(value))
                elif isinstance(current, float):
                    setattr(self.settings, field_name, float(value))
                else:
                    setattr(self.settings, field_name, value)
            self.settings.box_kind = "box" if self.box_radio.isChecked() else "drawer"
            self.settings.include_tabs = self.include_tabs.isChecked()
            if self.settings.relief_diameter <= 0.0:
                self.settings.relief_diameter = self.settings.bit_diameter
            self.panels = self.layout_generator.generate(self.settings)
            sheet_count = self._stock_sheet_count()
            self.statusBar().showMessage(
                f"{len(self.panels)} panels ready on {sheet_count} stock sheet"
                f"{'' if sheet_count == 1 else 's'}"
            )
            self._refresh_preview()
        except ValueError as error:
            self.statusBar().showMessage(str(error))

    def _refresh_preview(self) -> None:
        mode_by_step = {0: "box", 1: "material", 2: "joints", 3: "tabs", 5: "generate"}
        mode = mode_by_step.get(self.current_step)
        if mode is None:
            mode = (
                self.preview_mode.currentData()
                if hasattr(self, "preview_mode")
                else "flat"
            )
        self.preview.set_preview(self.panels, self.settings, mode or "flat")

    def _stock_sheet_count(self) -> int:
        if not self.panels:
            return 0
        return max(panel.stock_index for panel in self.panels) + 1

    def _new_project(self) -> None:
        if not self._confirm_discard_or_save_changes():
            return
        self.settings = BoxSettings()
        self.project_path = None
        self.has_unsaved_changes = False
        self._sync_inputs_from_settings()
        self.output_path.blockSignals(True)
        self.output_path.setText(str(self._default_output_path()))
        self.output_path.blockSignals(False)
        self._refresh_model()
        self.statusBar().showMessage("Started new project", 3000)
        self._update_window_title()

    def _open_project(self) -> None:
        if not self._confirm_discard_or_save_changes():
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self._load_project_dialog_directory(),
            PROJECT_FILE_FILTER,
        )
        if not selected:
            return
        path = Path(selected)
        self._remember_project_load_path(path)
        self._load_project_path(path)

    def _save_project(self) -> bool:
        if self.project_path is None:
            return self._save_project_as()
        return self._write_project(self.project_path)

    def _save_project_as(self) -> bool:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(
                Path(self._save_dialog_directory()) / self._default_project_file_name()
            ),
            PROJECT_FILE_FILTER,
        )
        if not selected:
            return False
        path = self._project_path_with_suffix(Path(selected))
        self._remember_save_path(path)
        return self._write_project(path)

    def _load_project_path(self, path: Path) -> bool:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Project file must contain a JSON object.")
            settings_data = data.get("settings", data)
            if not isinstance(settings_data, dict):
                raise ValueError("Project settings must be a JSON object.")
            self.settings = self._settings_from_data(settings_data)
            current_step = int(data.get("current_step", 0))
            output_path = str(data.get("output_path", ""))
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
            QMessageBox.critical(self, "Failed to open project", str(error))
            return False

        self.project_path = path.expanduser().resolve()
        self.has_unsaved_changes = False
        self._remember_recent_project(self.project_path)
        self._sync_inputs_from_settings()
        if output_path:
            self.output_path.setText(output_path)
        self._set_step(current_step)
        self._refresh_model()
        self.statusBar().showMessage(f"Opened {self.project_path.name}", 3000)
        self._update_window_title()
        return True

    def _write_project(self, path: Path) -> bool:
        self._refresh_model()
        resolved_path = path.expanduser().resolve()
        payload = {
            "format": "box-creator-project",
            "version": 1,
            "settings": asdict(self.settings),
            "current_step": self.current_step,
            "output_path": self.output_path.text(),
        }
        try:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            QMessageBox.critical(self, "Failed to save project", str(error))
            return False

        self.project_path = resolved_path
        self.has_unsaved_changes = False
        self._remember_recent_project(resolved_path)
        self.statusBar().showMessage(f"Saved {resolved_path.name}", 3000)
        self._update_window_title()
        return True

    def _settings_from_data(self, data: dict) -> BoxSettings:
        settings = BoxSettings()
        known_fields = {field.name: field for field in fields(BoxSettings)}
        for name, field in known_fields.items():
            if name not in data:
                continue
            current = getattr(settings, name)
            value = data[name]
            if isinstance(current, bool):
                setattr(settings, name, bool(value))
            elif isinstance(current, int):
                setattr(settings, name, int(value))
            elif isinstance(current, float):
                setattr(settings, name, float(value))
            else:
                setattr(settings, name, str(value))
        return settings

    def _sync_inputs_from_settings(self) -> None:
        for field_name, edit in self.inputs.items():
            edit.blockSignals(True)
            value = getattr(self.settings, field_name)
            if isinstance(value, float):
                edit.setText(f"{value:.3f}")
            else:
                edit.setText(str(value))
            edit.blockSignals(False)
        self.box_radio.blockSignals(True)
        self.drawer_radio.blockSignals(True)
        self.include_tabs.blockSignals(True)
        self.box_radio.setChecked(self.settings.box_kind == "box")
        self.drawer_radio.setChecked(self.settings.box_kind == "drawer")
        self.include_tabs.setChecked(self.settings.include_tabs)
        self.box_radio.blockSignals(False)
        self.drawer_radio.blockSignals(False)
        self.include_tabs.blockSignals(False)

    def _default_project_file_name(self) -> str:
        stem = self.settings.job_name.strip() or "finger-box"
        return f"{stem}.boxcreator.json"

    def _project_path_with_suffix(self, path: Path) -> Path:
        if path.name.endswith(".boxcreator.json"):
            return path
        if path.suffix.lower() == ".json":
            return path.with_name(f"{path.stem}.boxcreator.json")
        return path.with_suffix(".boxcreator.json")

    def _confirm_discard_or_save_changes(self) -> bool:
        if not self.has_unsaved_changes:
            return True
        response = QMessageBox.question(
            self,
            "Unsaved changes",
            "This project has unsaved changes. Save before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if response == QMessageBox.StandardButton.Save:
            return self._save_project()
        return response == QMessageBox.StandardButton.Discard

    def _mark_project_dirty(self) -> None:
        if self.has_unsaved_changes:
            return
        self.has_unsaved_changes = True
        self._update_window_title()

    def _update_window_title(self) -> None:
        project_name = (
            self.project_path.name
            if self.project_path is not None
            else "Untitled Project"
        )
        dirty_prefix = "*" if self.has_unsaved_changes else ""
        self.setWindowTitle(f"{dirty_prefix}{APPLICATION_NAME} - {project_name}")

    def _load_dialog_directory(self) -> str:
        configured = self.config.file_locations.last_load_directory.strip()
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
        return str(Path.home())

    def _load_project_dialog_directory(self) -> str:
        configured = self.config.file_locations.last_load_project_directory.strip()
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
        return self._load_dialog_directory()

    def _save_dialog_directory(self) -> str:
        configured = self.config.file_locations.last_save_directory.strip()
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
        return self._load_dialog_directory()

    def _remember_load_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        directory = paths[0].expanduser().resolve().parent
        self.config.file_locations.last_load_directory = str(directory)
        self._save_config(self.config)

    def _remember_project_load_path(self, path: Path) -> None:
        directory = path.expanduser().resolve().parent
        self.config.file_locations.last_load_project_directory = str(directory)
        self._save_config(self.config)

    def _remember_save_path(self, path: Path) -> None:
        directory = path.expanduser().resolve().parent
        self.config.file_locations.last_save_directory = str(directory)
        self._save_config(self.config)

    def _remember_recent_project(self, path: Path) -> None:
        resolved_path = path.expanduser().resolve()
        resolved = str(resolved_path)
        self.config.file_locations.last_load_project_directory = str(
            resolved_path.parent
        )
        recent_projects = [
            item
            for item in (self.config.file_locations.recent_projects or [])
            if item != resolved
        ]
        recent_projects.insert(0, resolved)
        limit = max(1, self.config.file_locations.recent_project_count)
        self.config.file_locations.recent_projects = recent_projects[:limit]
        self._save_config(self.config)

    def _browse_output(self) -> None:
        output_path = self._prompt_output_path()
        if output_path is None:
            return
        self.output_path.setText(str(output_path))
        self._mark_project_dirty()

    def _prompt_output_path(self) -> Path | None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save NC file",
            self._default_output_dialog_path(),
            "NC files (*.nc *.gcode);;All files (*)",
        )
        if not selected:
            return None
        path = Path(selected)
        self._remember_save_path(path)
        return path

    def _default_output_dialog_path(self) -> str:
        current = Path(self.output_path.text()).expanduser()
        if current.parent.exists():
            return str(current)
        return str(Path(self._save_dialog_directory()) / current.name)

    def _default_output_path(self) -> Path:
        return Path.cwd() / "output" / "finger-box.nc"

    def _generate_nc(self) -> None:
        self._refresh_model()
        output_path = self._prompt_output_path()
        if output_path is None:
            return
        previous_output_path = self.output_path.text()
        self.output_path.setText(str(output_path))
        if self.output_path.text() != previous_output_path:
            self._mark_project_dirty()
        try:
            self.gcode_generator.write(self.panels, self.settings, output_path)
        except OSError as error:
            QMessageBox.critical(self, "Generate failed", str(error))
            return
        QMessageBox.information(self, "NC generated", f"Wrote {output_path}")
        self.statusBar().showMessage(f"Wrote {output_path}")

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #2f3138; color: #f3f5f7; font-size: 13px; }
            QLabel#heading { font-size: 20px; font-weight: 700; color: #ffffff; }
            QLineEdit, QComboBox { background: #171b22; color: #f3f5f7; border: 1px solid #596270; padding: 6px; }
            QPushButton { background: #233243; color: #f3f5f7; border: 1px solid #596270; padding: 7px 11px; }
            QPushButton:hover { background: #2d4258; }
            QPushButton[currentStep="true"] { background: #ff9f43; color: #1f1509; border-color: #ffd0a3; }
            QPushButton:disabled { color: #8d97a3; background: #242832; }
            QFrame { color: #596270; }
            """
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._confirm_discard_or_save_changes():
            event.accept()
        else:
            event.ignore()


def _config_path() -> Path:
    config_root = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericConfigLocation
    )
    base_path = Path(config_root) if config_root else Path.home() / ".config"
    return base_path / ORGANIZATION_NAME / APPLICATION_NAME / CONFIG_FILE_NAME


def _load_config() -> AppConfig:
    path = _config_path()
    if not path.exists():
        return AppConfig()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return AppConfig()
    if not isinstance(loaded, dict):
        return AppConfig()
    file_locations_data = loaded.get("file_locations", {})
    if not isinstance(file_locations_data, dict):
        file_locations_data = {}
    ui_data = loaded.get("ui_save_state", {})
    if not isinstance(ui_data, dict):
        ui_data = {}
    return AppConfig(
        file_locations=FileLocations(
            recent_project_count=int(
                file_locations_data.get("recent_project_count", 10)
            ),
            last_load_project_directory=str(
                file_locations_data.get("last_load_project_directory", "")
            ),
            last_load_directory=str(file_locations_data.get("last_load_directory", "")),
            last_save_directory=str(file_locations_data.get("last_save_directory", "")),
            recent_projects=[
                str(item)
                for item in file_locations_data.get("recent_projects", [])
                if isinstance(item, str)
            ],
        ),
        ui_save_state=UiSaveState(
            last_screen_name=str(ui_data.get("last_screen_name", "")),
            window_state=str(ui_data.get("window_state", "normal")),
            window_x=_optional_int(ui_data.get("window_x")),
            window_y=_optional_int(ui_data.get("window_y")),
            window_width=_positive_int(ui_data.get("window_width"), 1180),
            window_height=_positive_int(ui_data.get("window_height"), 760),
        ),
    )


def _save_config(config: AppConfig) -> None:
    path = _config_path()
    payload = {
        "file_locations": asdict(config.file_locations),
        "ui_save_state": asdict(config.ui_save_state),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
    except OSError:
        return


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


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

        window.resize(
            max(MINIMUM_WINDOW_WIDTH, width), max(MINIMUM_WINDOW_HEIGHT, height)
        )
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
    window = MainWindow(config, _save_config)
    _apply_saved_window_placement(window, startup_screen, config)
    app.aboutToQuit.connect(lambda: _save_window_placement(window, config))
    if config.ui_save_state.window_state == "maximized":
        window.showMaximized()
    else:
        window.show()
        app.processEvents()
    return app.exec()
