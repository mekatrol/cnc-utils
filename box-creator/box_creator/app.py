from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QIntValidator
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

from .box_settings import BoxSettings
from .gcode_generator import GcodeGenerator
from .layout_generator import LayoutGenerator
from .preview_widget import PreviewWidget


class MainWindow(QMainWindow):
    STEP_TITLES = ["Box", "Material", "Joints", "Tabs", "Preview", "Generate"]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("box-creator")
        self.resize(1180, 760)
        self.settings = BoxSettings()
        self.layout_generator = LayoutGenerator()
        self.gcode_generator = GcodeGenerator()
        self.panels = self.layout_generator.generate(self.settings)
        self.current_step = 0
        self.inputs: dict[str, QLineEdit] = {}

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)
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
        self.include_tabs.toggled.connect(self._refresh_model)
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
        self.output_path = QLineEdit(str(Path.cwd() / "output" / "finger-box.nc"))
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
        self.box_radio.toggled.connect(self._refresh_model)
        self.drawer_radio.toggled.connect(self._refresh_model)
        layout.addWidget(self.box_radio)
        layout.addWidget(self.drawer_radio)
        return holder

    def _add_text(self, form: QFormLayout, label: str, field_name: str) -> None:
        edit = QLineEdit(str(getattr(self.settings, field_name)))
        edit.editingFinished.connect(self._refresh_model)
        self.inputs[field_name] = edit
        form.addRow(label, edit)

    def _add_number(self, form: QFormLayout, label: str, field_name: str) -> None:
        edit = QLineEdit(f"{getattr(self.settings, field_name):.3f}")
        validator = QDoubleValidator(0.001, 10000.0, 3, edit)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        edit.setValidator(validator)
        edit.editingFinished.connect(self._refresh_model)
        self.inputs[field_name] = edit
        form.addRow(f"{label} (mm)", edit)

    def _add_integer(self, form: QFormLayout, label: str, field_name: str) -> None:
        edit = QLineEdit(str(getattr(self.settings, field_name)))
        edit.setValidator(QIntValidator(1, 100000, edit))
        edit.editingFinished.connect(self._refresh_model)
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

    def _browse_output(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save NC file",
            self.output_path.text(),
            "NC files (*.nc *.gcode);;All files (*)",
        )
        if selected:
            self.output_path.setText(selected)

    def _generate_nc(self) -> None:
        self._refresh_model()
        output_path = Path(self.output_path.text()).expanduser()
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


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
