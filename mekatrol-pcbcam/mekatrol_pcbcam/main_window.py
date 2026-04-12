from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .app_config import AppConfig
from .excellon_file_parser import ExcellonFileParser
from .gerber_file_parser import GerberFileParser
from .imported_drill_file import ImportedDrillFile
from .imported_gerber_file import ImportedGerberFile
from .pcb_preview_widget import PcbPreviewWidget
from .pcb_project import PcbProject
from .wizard_step_bar import WizardStepBar


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    STEP_TITLES = [
        "Gerber Import",
        "Drill Import",
        "Tool Selection",
        "Layer Assignment",
        "Mirror Setup",
        "Alignment Holes",
        "Front Isolation",
        "Back Isolation",
        "Drilling",
        "Edge Cuts",
        "NC Preview",
    ]
    IMPLEMENTED_STEP_COUNT = 2

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.project = PcbProject()
        self.gerber_parser = GerberFileParser()
        self.drill_parser = ExcellonFileParser()
        self.imported_gerbers: list[ImportedGerberFile] = []
        self.imported_drills: list[ImportedDrillFile] = []

        self.preview = PcbPreviewWidget()
        self.step_bar = WizardStepBar(self.STEP_TITLES)
        self.step_bar.step_selected.connect(self._handle_step_selected)

        self.setWindowTitle("mekatrol-pcbcam")
        self.resize(1440, 920)
        self._build_ui()
        self._build_menu()
        self._sync_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)
        root_layout.addWidget(self.step_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1020])
        root_layout.addWidget(splitter, 1)

        nav_row = QHBoxLayout()
        nav_row.addStretch(1)
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._go_back)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._go_next)
        nav_row.addWidget(self.back_button)
        nav_row.addWidget(self.next_button)
        root_layout.addLayout(nav_row)

        self.setCentralWidget(root)
        status = QStatusBar(self)
        status.showMessage("Wizard ready")
        self.setStatusBar(status)

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("PCB CAM Wizard")
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        subtitle = QLabel(
            "Stage 1 implements project handling plus Gerber and Excellon import. "
            "Later CAM steps stay visible in the wizard but remain locked until "
            "their stage is implemented."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #5b6571;")

        summary_card = QFrame()
        summary_card.setFrameShape(QFrame.Shape.StyledPanel)
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(14, 14, 14, 14)
        summary_layout.setSpacing(8)
        self.project_value = QLabel("Unsaved project")
        self.gerber_count_value = QLabel("Gerber files: 0")
        self.drill_count_value = QLabel("Drill files: 0")
        self.step_status_value = QLabel("Stage 1 of the wizard is active.")
        self.step_status_value.setWordWrap(True)
        summary_layout.addWidget(self.project_value)
        summary_layout.addWidget(self.gerber_count_value)
        summary_layout.addWidget(self.drill_count_value)
        summary_layout.addWidget(self.step_status_value)

        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(self._build_gerber_page())
        self.page_stack.addWidget(self._build_drill_page())

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(summary_card)
        layout.addWidget(self.page_stack, 1)
        return panel

    def _build_gerber_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 1: Import Gerber")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(
            "Import one or more Gerber files. Typical files include front copper, "
            "back copper, and edge cuts. All imported geometry is shown in the "
            "preview so you can validate the board before moving on."
        )
        body.setWordWrap(True)

        button_row = QHBoxLayout()
        import_button = QPushButton("Import Gerber Files")
        import_button.clicked.connect(self._import_gerber_files)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_gerbers)
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self._clear_gerbers)
        button_row.addWidget(import_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(clear_button)

        self.gerber_list = QListWidget()
        self.gerber_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        self.gerber_hint = QLabel("Import at least one Gerber file to continue.")
        self.gerber_hint.setWordWrap(True)
        self.gerber_hint.setStyleSheet("color: #5b6571;")

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addLayout(button_row)
        layout.addWidget(self.gerber_list, 1)
        layout.addWidget(self.gerber_hint)
        return page

    def _build_drill_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 2: Import Excellon Drill Files")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(
            "Import Excellon drill files for PTH and NPTH holes. Drill import is "
            "optional for Stage 1, but the preview will overlay hole sizes and "
            "positions so the board stack can be checked now."
        )
        body.setWordWrap(True)

        button_row = QHBoxLayout()
        import_button = QPushButton("Import Drill Files")
        import_button.clicked.connect(self._import_drill_files)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_drills)
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self._clear_drills)
        button_row.addWidget(import_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(clear_button)

        self.drill_list = QListWidget()
        self.drill_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        self.drill_hint = QLabel(
            "You can continue without drill files in Stage 1. Later drill-operation "
            "steps will use the project data saved here."
        )
        self.drill_hint.setWordWrap(True)
        self.drill_hint.setStyleSheet("color: #5b6571;")

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addLayout(button_row)
        layout.addWidget(self.drill_list, 1)
        layout.addWidget(self.drill_hint)
        return page

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        new_project_action = QAction("New Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("Open Project...", self)
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self._open_project)
        file_menu.addAction(open_project_action)

        save_project_action = QAction("Save Project", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project As...", self)
        save_project_as_action.setShortcut("Ctrl+Shift+S")
        save_project_as_action.triggered.connect(self._save_project_as)
        file_menu.addAction(save_project_as_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = self.menuBar().addMenu("&View")
        fit_action = QAction("Fit Preview", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self.preview.fit_to_view)
        view_menu.addAction(fit_action)

    def _new_project(self) -> None:
        self.project.reset()
        self.imported_gerbers = []
        self.imported_drills = []
        self.statusBar().showMessage("Started new project", 3000)
        self._sync_ui()

    def _open_project(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self._load_dialog_directory(),
            "PCB CAM Project (*.mpcbcam.yaml *.yaml);;All Files (*)",
        )
        if not selected:
            return
        self._remember_load_paths([Path(selected)])
        try:
            project = PcbProject.load_from_path(Path(selected))
            imported_gerbers = self._parse_gerber_paths(project.gerber_paths)
            imported_drills = self._parse_drill_paths(project.drill_paths)
        except Exception as exc:
            logger.exception("Failed to open project: %s", selected)
            QMessageBox.critical(self, "Failed to open project", str(exc))
            return

        self.project = project
        self.project.set_current_step(
            min(self.project.current_step_index, self.IMPLEMENTED_STEP_COUNT - 1)
        )
        self.imported_gerbers = imported_gerbers
        self.imported_drills = imported_drills
        self.statusBar().showMessage(f"Opened {Path(selected).name}", 3000)
        self._sync_ui()

    def _save_project(self) -> None:
        if self.project.project_path is None:
            self._save_project_as()
            return
        self._write_project(self.project.project_path)

    def _save_project_as(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(Path(self._save_dialog_directory()) / "project.mpcbcam.yaml"),
            "PCB CAM Project (*.mpcbcam.yaml);;YAML Files (*.yaml)",
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() != ".yaml":
            path = path.with_suffix(".mpcbcam.yaml")
        self._remember_save_path(path)
        self._write_project(path)

    def _write_project(self, path: Path) -> None:
        try:
            self.project.save_to_path(path)
        except Exception as exc:
            logger.exception("Failed to save project: %s", path)
            QMessageBox.critical(self, "Failed to save project", str(exc))
            return
        self.statusBar().showMessage(f"Saved {path.name}", 3000)
        self._sync_ui()

    def _handle_step_selected(self, index: int) -> None:
        if not self.project.can_navigate_to(index, self.IMPLEMENTED_STEP_COUNT):
            self.statusBar().showMessage(
                "That wizard step is locked. Move forward sequentially after changes.",
                4000,
            )
            return
        self.project.set_current_step(index)
        self._sync_ui()

    def _go_back(self) -> None:
        if self.project.current_step_index <= 0:
            return
        self.project.set_current_step(self.project.current_step_index - 1)
        self._sync_ui()

    def _go_next(self) -> None:
        current = self.project.current_step_index
        if current + 1 >= self.IMPLEMENTED_STEP_COUNT:
            self.statusBar().showMessage(
                "The next wizard steps are reserved for the next implementation stage.",
                4000,
            )
            return
        if not self._step_is_valid(current):
            QMessageBox.information(
                self,
                "Wizard step incomplete",
                "Import at least one Gerber file before moving to the next step.",
            )
            return
        self.project.completed_steps.add(current)
        self.project.clear_dirty_state_through(current + 1)
        self.project.set_current_step(current + 1)
        self.project.completed_steps.add(1)
        self._sync_ui()

    def _import_gerber_files(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Gerber Files",
            self._load_dialog_directory(),
            "Gerber Files (*.gbr *.ger *.gtl *.gbl *.gko *.gm1);;All Files (*)",
        )
        if not selected:
            return
        paths = [Path(item) for item in selected]
        self._remember_load_paths(paths)
        try:
            imports = self._parse_gerber_paths(paths)
        except Exception as exc:
            logger.exception("Failed to import Gerber files")
            QMessageBox.critical(self, "Failed to import Gerber files", str(exc))
            return

        self.imported_gerbers = imports
        self.project.replace_gerber_paths(paths)
        self.project.set_current_step(0)
        self.statusBar().showMessage(
            f"Imported {len(self.imported_gerbers)} Gerber file(s)",
            3000,
        )
        self._sync_ui()

    def _import_drill_files(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Drill Files",
            self._load_dialog_directory(),
            "Drill Files (*.drl *.xln *.txt);;All Files (*)",
        )
        if not selected:
            return
        paths = [Path(item) for item in selected]
        self._remember_load_paths(paths)
        try:
            imports = self._parse_drill_paths(paths)
        except Exception as exc:
            logger.exception("Failed to import drill files")
            QMessageBox.critical(self, "Failed to import drill files", str(exc))
            return

        self.imported_drills = imports
        self.project.replace_drill_paths(paths)
        self.project.set_current_step(1)
        self.statusBar().showMessage(
            f"Imported {len(self.imported_drills)} drill file(s)",
            3000,
        )
        self._sync_ui()

    def _remove_selected_gerbers(self) -> None:
        selected_paths = {
            item.data(Qt.ItemDataRole.UserRole) for item in self.gerber_list.selectedItems()
        }
        if not selected_paths:
            return
        remaining = [item for item in self.imported_gerbers if str(item.path) not in selected_paths]
        self.imported_gerbers = remaining
        self.project.replace_gerber_paths([item.path for item in remaining])
        self.project.set_current_step(0)
        self._sync_ui()

    def _clear_gerbers(self) -> None:
        self.imported_gerbers = []
        self.project.replace_gerber_paths([])
        self.project.set_current_step(0)
        self._sync_ui()

    def _remove_selected_drills(self) -> None:
        selected_paths = {
            item.data(Qt.ItemDataRole.UserRole) for item in self.drill_list.selectedItems()
        }
        if not selected_paths:
            return
        remaining = [item for item in self.imported_drills if str(item.path) not in selected_paths]
        self.imported_drills = remaining
        self.project.replace_drill_paths([item.path for item in remaining])
        self.project.set_current_step(min(self.project.current_step_index, 1))
        self._sync_ui()

    def _clear_drills(self) -> None:
        self.imported_drills = []
        self.project.replace_drill_paths([])
        self.project.set_current_step(min(self.project.current_step_index, 1))
        self._sync_ui()

    def _parse_gerber_paths(self, paths: list[Path]) -> list[ImportedGerberFile]:
        imports = [self.gerber_parser.parse_file(path) for path in paths]
        return sorted(imports, key=lambda item: item.display_name.lower())

    def _parse_drill_paths(self, paths: list[Path]) -> list[ImportedDrillFile]:
        imports = [self.drill_parser.parse_file(path) for path in paths]
        return sorted(imports, key=lambda item: item.display_name.lower())

    def _step_is_valid(self, index: int) -> bool:
        if index == 0:
            return bool(self.imported_gerbers)
        if index == 1:
            return True
        return False

    def _sync_ui(self) -> None:
        current = min(self.project.current_step_index, self.IMPLEMENTED_STEP_COUNT - 1)
        self.project.current_step_index = current
        self.page_stack.setCurrentIndex(current)
        self.step_bar.set_state(
            current_step_index=current,
            completed_steps=self.project.completed_steps,
            is_step_enabled=lambda index: self.project.can_navigate_to(
                index, self.IMPLEMENTED_STEP_COUNT
            ),
        )
        self.back_button.setEnabled(current > 0)
        self.next_button.setEnabled(current + 1 < self.IMPLEMENTED_STEP_COUNT)
        self.project_value.setText(
            str(self.project.project_path) if self.project.project_path else "Unsaved project"
        )
        self.gerber_count_value.setText(f"Gerber files: {len(self.imported_gerbers)}")
        self.drill_count_value.setText(f"Drill files: {len(self.imported_drills)}")
        self.step_status_value.setText(self._step_status_text())
        self._refresh_list_widgets()
        self.preview.load_project_geometry(self.imported_gerbers, self.imported_drills)
        self._update_window_title()

    def _refresh_list_widgets(self) -> None:
        self.gerber_list.clear()
        for gerber in self.imported_gerbers:
            geometry_summary = []
            if gerber.traces or gerber.regions or gerber.pads:
                geometry_summary.append("copper geometry")
            if gerber.outline:
                geometry_summary.append("outline")
            if not geometry_summary:
                geometry_summary.append("no visible geometry detected")
            item = QListWidgetItem(f"{gerber.display_name} ({', '.join(geometry_summary)})")
            item.setToolTip(str(gerber.path))
            item.setData(Qt.ItemDataRole.UserRole, str(gerber.path))
            self.gerber_list.addItem(item)

        self.drill_list.clear()
        for drill in self.imported_drills:
            item = QListWidgetItem(f"{drill.display_name} ({len(drill.holes)} holes)")
            item.setToolTip(str(drill.path))
            item.setData(Qt.ItemDataRole.UserRole, str(drill.path))
            self.drill_list.addItem(item)

    def _step_status_text(self) -> str:
        if self.project.dirty_from_step is not None:
            return (
                f"Changes were made at step {self.project.dirty_from_step + 1}. "
                "Forward navigation is sequential until the wizard is replayed."
            )
        if self.project.current_step_index == 0 and not self.imported_gerbers:
            return "Step 1 requires at least one Gerber file before the wizard can continue."
        if self.project.current_step_index == 1:
            return "Stage 1 ends after drill import. Tool selection and CAM generation come next."
        return "Stage 1 of the wizard is active."

    def _update_window_title(self) -> None:
        project_name = (
            self.project.project_path.name
            if self.project.project_path is not None
            else "Untitled Project"
        )
        self.setWindowTitle(f"mekatrol-pcbcam - {project_name}")

    def _load_dialog_directory(self) -> str:
        configured = self.config.file_locations.last_load_directory.strip()
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.exists() and candidate.is_dir():
                return str(candidate)
        return str(Path.home())

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

    def _remember_save_path(self, path: Path) -> None:
        directory = path.expanduser().resolve().parent
        self.config.file_locations.last_save_directory = str(directory)
