from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

from PySide6.QtCore import QStandardPaths, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .app_config import AppConfig
from .alignment_hole import AlignmentHole
from .excellon_file_parser import ExcellonFileParser
from .gcode_parser import GCodeParser
from .gerber_file_parser import GerberFileParser
from .imported_drill_file import ImportedDrillFile
from .imported_gerber_file import ImportedGerberFile
from .mirror_preview_widget import MirrorPreviewWidget
from .pcb_preview_widget import PcbPreviewWidget
from .pcb_project import PcbProject
from .theme import AppTheme, load_theme
from .theme_settings_dialog import ThemeSettingsDialog, discover_theme_options
from .tool_library import ToolLibrary
from .viewer import ToolpathViewer
from .wizard_step_bar import WizardStepBar
from .app_constants import APPLICATION_NAME, ORGANIZATION_NAME


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    STEP_TITLES = [
        "Project",
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
    IMPLEMENTED_STEP_COUNT = 12

    def __init__(
        self,
        config: AppConfig,
        *,
        themes_directory: Path,
        save_config: Callable[[AppConfig], None],
    ) -> None:
        super().__init__()
        self.config = config
        self.theme = config.theme
        self._themes_directory = themes_directory
        self._save_config = save_config
        self.project = PcbProject()
        self.gerber_parser = GerberFileParser()
        self.drill_parser = ExcellonFileParser()
        self.gcode_parser = GCodeParser()
        self.imported_gerbers: list[ImportedGerberFile] = []
        self.imported_drills: list[ImportedDrillFile] = []
        self.tool_library: ToolLibrary | None = None
        self.generated_documents = {}
        self.has_unsaved_changes = False
        self._muted_labels: list[QLabel] = []
        self._sidebar_panels: list[QWidget] = []

        self.preview = PcbPreviewWidget(self.theme)
        self.toolpath_viewer = ToolpathViewer(self.theme)
        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.preview)
        self.preview_stack.addWidget(self.toolpath_viewer)
        self.step_bar = WizardStepBar(self.STEP_TITLES, self.theme)
        self.step_bar.step_selected.connect(self._handle_step_selected)
        self.step_bar_scroll = QScrollArea()
        self.step_bar_scroll.setObjectName("stepBarScroll")
        self.step_bar_scroll.setWidget(self.step_bar)
        self.step_bar_scroll.setWidgetResizable(False)
        self.step_bar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.step_bar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.step_bar_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.step_bar_scroll.setViewportMargins(0, 0, 0, 6)
        self.step_bar_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.setWindowTitle("mekatrol-pcbcam")
        self.setObjectName("mainWindow")
        self.resize(1440, 920)
        self._build_ui()
        self._build_menu()
        self.tool_library = self._default_tool_library()
        self._sync_ui()

    def _muted_text_style(self) -> str:
        return f"color: {self.theme.main_window_muted_text};"

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("mainWindowRoot")
        self._root_widget = root
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)
        root_layout.addWidget(self.step_bar_scroll)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self.preview_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1020])
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(root)
        status = QStatusBar(self)
        status.showMessage("Wizard ready")
        self.setStatusBar(status)
        self._apply_window_theme()

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("sidebarPanel")
        self._sidebar_panel = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 18, 0)
        layout.setSpacing(12)

        title = QLabel("PCB CAM Wizard")
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        subtitle = QLabel(
            "Work through the full CAM wizard from Gerber import through NC generation. "
            "Each step saves into the project so you can reopen and continue."
        )
        subtitle.setWordWrap(True)
        self._apply_muted_text_style(subtitle)

        summary_card = QFrame()
        summary_card.setFrameShape(QFrame.Shape.StyledPanel)
        summary_card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(summary_card)
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(14, 14, 14, 14)
        summary_layout.setSpacing(8)
        self.project_value = QLabel("Unsaved project")
        self.gerber_count_value = QLabel("Gerber files: 0")
        self.drill_count_value = QLabel("Drill files: 0")
        self.step_status_value = QLabel("Wizard ready.")
        self.step_status_value.setWordWrap(True)
        summary_layout.addWidget(self.project_value)
        summary_layout.addWidget(self.gerber_count_value)
        summary_layout.addWidget(self.drill_count_value)
        summary_layout.addWidget(self.step_status_value)

        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(self._build_project_page())
        self.page_stack.addWidget(self._build_gerber_page())
        self.page_stack.addWidget(self._build_drill_page())
        self.page_stack.addWidget(self._build_tool_selection_page())
        self.page_stack.addWidget(self._build_layer_assignment_page())
        self.page_stack.addWidget(self._build_mirror_setup_page())
        self.page_stack.addWidget(self._build_alignment_holes_page())
        self.page_stack.addWidget(
            self._build_operation_page(
                "Step 8: Front Isolation",
                "Generate front copper isolation G-code from the assigned front copper layer.",
                "Generate Front Isolation",
                "_generate_front_isolation",
                "front_isolation",
            )
        )
        self.page_stack.addWidget(
            self._build_operation_page(
                "Step 9: Back Isolation",
                "Generate back copper isolation G-code from the assigned back copper layer.",
                "Generate Back Isolation",
                "_generate_back_isolation",
                "back_isolation",
            )
        )
        self.page_stack.addWidget(
            self._build_operation_page(
                "Step 10: Drilling",
                "Generate drilling G-code for imported drill holes and optional alignment holes.",
                "Generate Drill Operations",
                "_generate_drilling_operations",
                "drilling",
            )
        )
        self.page_stack.addWidget(
            self._build_operation_page(
                "Step 11: Edge Cuts",
                "Generate edge cut G-code from the assigned board outline.",
                "Generate Edge Cuts",
                "_generate_edge_cuts",
                "edge_cuts",
            )
        )
        self.page_stack.addWidget(self._build_nc_preview_page())

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(summary_card)
        layout.addWidget(self.page_stack, 1)

        nav_row = QHBoxLayout()
        nav_row.addStretch(1)
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._go_back)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._go_next)
        nav_row.addWidget(self.back_button)
        nav_row.addWidget(self.next_button)
        layout.addLayout(nav_row)
        return panel

    def _build_project_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 1: Project")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(
            "Start a new project, reopen an existing one, or save the current setup. "
            "If the app automatically reopens a recent project on startup, the wizard "
            "resumes at the saved wizard step."
        )
        body.setWordWrap(True)

        project_row = QHBoxLayout()
        new_project_button = QPushButton("New Project")
        new_project_button.clicked.connect(self._new_project)
        open_project_button = QPushButton("Open Project...")
        open_project_button.clicked.connect(self._open_project)
        project_row.addWidget(new_project_button)
        project_row.addWidget(open_project_button)

        hint = QLabel(
            "Use this step to create or reopen a project file. Gerber import happens in the next step."
        )
        hint.setWordWrap(True)
        self._apply_muted_text_style(hint)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addLayout(project_row)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_gerber_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 2: Import Gerber")
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
        self.gerber_list.itemChanged.connect(self._gerber_item_changed)

        self.gerber_hint = QLabel("Import at least one Gerber file to continue.")
        self.gerber_hint.setWordWrap(True)
        self._apply_muted_text_style(self.gerber_hint)

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

        heading = QLabel("Step 3: Import Excellon Drill Files")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(
            "Import Excellon drill files for PTH and NPTH holes. Drill import is "
            "optional, but the preview will overlay hole sizes and "
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
        self.drill_list.itemChanged.connect(self._drill_item_changed)

        self.drill_hint = QLabel(
            "You can continue without drill files. Later drill-operation "
            "steps will use the project data saved here."
        )
        self.drill_hint.setWordWrap(True)
        self._apply_muted_text_style(self.drill_hint)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addLayout(button_row)
        layout.addWidget(self.drill_list, 1)
        layout.addWidget(self.drill_hint)
        return page

    def _build_tool_selection_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 4: Tool Selection")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(
            "Choose one drilling tool, one milling tool, and one V-bit from the "
            "configured tool library. These choices are stored in the project and "
            "will feed the CAM generation stages."
        )
        body.setWordWrap(True)

        button_row = QHBoxLayout()
        load_button = QPushButton("Load tools.yaml")
        load_button.clicked.connect(self._browse_tool_library)
        clear_button = QPushButton("Clear Tool Library")
        clear_button.clicked.connect(self._clear_tool_library)
        button_row.addWidget(load_button)
        button_row.addWidget(clear_button)

        form_card = QFrame()
        form_card.setFrameShape(QFrame.Shape.StyledPanel)
        form_card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(form_card)
        form = QFormLayout(form_card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(10)
        self.tool_library_value = QLabel("No tool library loaded")
        self.tool_library_value.setWordWrap(True)
        self.drilling_tool_combo = QComboBox()
        self.drilling_tool_combo.currentIndexChanged.connect(
            lambda _: self._tool_selection_changed("drilling", self.drilling_tool_combo)
        )
        self.milling_tool_combo = QComboBox()
        self.milling_tool_combo.currentIndexChanged.connect(
            lambda _: self._tool_selection_changed("milling", self.milling_tool_combo)
        )
        self.vbit_tool_combo = QComboBox()
        self.vbit_tool_combo.currentIndexChanged.connect(
            lambda _: self._tool_selection_changed("v_bits", self.vbit_tool_combo)
        )
        form.addRow("Library", self.tool_library_value)
        form.addRow("Drilling", self.drilling_tool_combo)
        form.addRow("Milling", self.milling_tool_combo)
        form.addRow("V-bit", self.vbit_tool_combo)

        self.tool_selection_hint = QLabel("Load a tool library and select all three tool types.")
        self.tool_selection_hint.setWordWrap(True)
        self._apply_muted_text_style(self.tool_selection_hint)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addLayout(button_row)
        layout.addWidget(form_card)
        layout.addWidget(self.tool_selection_hint)
        layout.addStretch(1)
        return page

    def _build_layer_assignment_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 5: Layer Assignment")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(
            "Assign imported Gerber files to front copper, back copper, and edges. "
            "Any assignment is optional, but at least one of the three roles must be "
            "filled before the wizard can continue."
        )
        body.setWordWrap(True)

        form_card = QFrame()
        form_card.setFrameShape(QFrame.Shape.StyledPanel)
        form_card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(form_card)
        form = QFormLayout(form_card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(10)
        self.front_copper_combo = QComboBox()
        self.front_copper_combo.currentIndexChanged.connect(
            lambda _: self._layer_assignment_changed("front_copper", self.front_copper_combo)
        )
        self.back_copper_combo = QComboBox()
        self.back_copper_combo.currentIndexChanged.connect(
            lambda _: self._layer_assignment_changed("back_copper", self.back_copper_combo)
        )
        self.edges_combo = QComboBox()
        self.edges_combo.currentIndexChanged.connect(
            lambda _: self._layer_assignment_changed("edges", self.edges_combo)
        )
        form.addRow("Front copper", self.front_copper_combo)
        form.addRow("Back copper", self.back_copper_combo)
        form.addRow("Edges", self.edges_combo)

        self.layer_assignment_hint = QLabel("At least one of front copper, back copper, or edges is required.")
        self.layer_assignment_hint.setWordWrap(True)
        self._apply_muted_text_style(self.layer_assignment_hint)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(form_card)
        layout.addWidget(self.layer_assignment_hint)
        layout.addStretch(1)
        return page

    def _build_mirror_setup_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 6: Mirror Setup")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(
            "If both front and back copper are assigned, choose the edge used for "
            "mirroring when the board is flipped. This step is skipped automatically "
            "when only one copper side is active."
        )
        body.setWordWrap(True)

        self.mirror_requirement_label = QLabel()
        self.mirror_requirement_label.setWordWrap(True)

        button_row = QHBoxLayout()
        self.mirror_button_group = QButtonGroup(self)
        self.mirror_buttons: dict[str, QRadioButton] = {}
        for edge, label in (
            ("left", "Left"),
            ("top", "Top"),
            ("right", "Right"),
            ("bottom", "Bottom"),
        ):
            button = QRadioButton(label)
            button.toggled.connect(
                lambda checked, selected=edge: self._mirror_edge_changed(selected, checked)
            )
            self.mirror_button_group.addButton(button)
            self.mirror_buttons[edge] = button
            button_row.addWidget(button)

        self.mirror_preview = MirrorPreviewWidget(self.theme)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(self.mirror_requirement_label)
        layout.addLayout(button_row)
        layout.addWidget(self.mirror_preview)
        layout.addStretch(1)
        return page

    def _build_alignment_holes_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 7: Alignment Holes")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(
            "Define optional alignment holes outside the board edge. Each hole uses "
            "an edge reference plus offsets measured along that edge and outward "
            "from the board boundary."
        )
        body.setWordWrap(True)

        form_card = QFrame()
        form_card.setFrameShape(QFrame.Shape.StyledPanel)
        form_card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(form_card)
        form = QFormLayout(form_card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(10)
        self.alignment_edge_combo = QComboBox()
        self.alignment_edge_combo.addItems(["left", "top", "right", "bottom"])
        self.alignment_offset_along_spin = QDoubleSpinBox()
        self.alignment_offset_along_spin.setDecimals(3)
        self.alignment_offset_along_spin.setRange(-10000.0, 10000.0)
        self.alignment_offset_along_spin.setSuffix(" mm")
        self.alignment_offset_from_edge_spin = QDoubleSpinBox()
        self.alignment_offset_from_edge_spin.setDecimals(3)
        self.alignment_offset_from_edge_spin.setRange(0.0, 10000.0)
        self.alignment_offset_from_edge_spin.setValue(2.0)
        self.alignment_offset_from_edge_spin.setSuffix(" mm")
        self.alignment_diameter_spin = QDoubleSpinBox()
        self.alignment_diameter_spin.setDecimals(3)
        self.alignment_diameter_spin.setRange(0.01, 100.0)
        self.alignment_diameter_spin.setValue(1.0)
        self.alignment_diameter_spin.setSuffix(" mm")
        form.addRow("Edge", self.alignment_edge_combo)
        form.addRow("Offset along edge", self.alignment_offset_along_spin)
        form.addRow("Offset from edge", self.alignment_offset_from_edge_spin)
        form.addRow("Hole diameter", self.alignment_diameter_spin)

        button_row = QHBoxLayout()
        add_button = QPushButton("Add Alignment Hole")
        add_button.clicked.connect(self._add_alignment_hole)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_alignment_holes)
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self._clear_alignment_holes)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(clear_button)

        self.alignment_hole_list = QListWidget()
        self.alignment_holes_hint = QLabel(
            "Alignment holes are optional. Added holes are shown in green in the preview."
        )
        self.alignment_holes_hint.setWordWrap(True)
        self._apply_muted_text_style(self.alignment_holes_hint)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(form_card)
        layout.addLayout(button_row)
        layout.addWidget(self.alignment_hole_list, 1)
        layout.addWidget(self.alignment_holes_hint)
        return page

    def _build_operation_page(
        self,
        heading_text: str,
        body_text: str,
        button_text: str,
        handler_name: str,
        operation_key: str,
    ) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel(heading_text)
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(body_text)
        body.setWordWrap(True)

        button = QPushButton(button_text)
        button.clicked.connect(getattr(self, handler_name))

        path_value = QLabel("Not generated yet")
        path_value.setWordWrap(True)
        path_value.setProperty("operation_key", operation_key)
        if operation_key == "front_isolation":
            self.front_isolation_value = path_value
        elif operation_key == "back_isolation":
            self.back_isolation_value = path_value
        elif operation_key == "drilling":
            self.drilling_value = path_value
        elif operation_key == "edge_cuts":
            self.edge_cuts_value = path_value

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(button)
        layout.addWidget(path_value)
        layout.addStretch(1)
        return page

    def _build_nc_preview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 12: NC Preview")
        heading.setStyleSheet("font-size: 20px; font-weight: 700;")
        body = QLabel(
            "Select any generated NC file to inspect it in the 3D toolpath viewer."
        )
        body.setWordWrap(True)

        self.generated_output_list = QListWidget()
        self.generated_output_list.currentRowChanged.connect(self._generated_output_selected)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(self.generated_output_list, 1)
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

        close_project_action = QAction("Close Project", self)
        close_project_action.setShortcut("Ctrl+W")
        close_project_action.triggered.connect(self._close_project)
        file_menu.addAction(close_project_action)

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

        settings_menu = self.menuBar().addMenu("&Settings")
        theme_action = QAction("Theme...", self)
        theme_action.triggered.connect(self._open_theme_settings)
        settings_menu.addAction(theme_action)

    def _apply_muted_text_style(self, label: QLabel) -> None:
        if label not in self._muted_labels:
            self._muted_labels.append(label)
        label.setStyleSheet(self._muted_text_style())

    def _theme_stylesheet(self) -> str:
        return f"""
            QMainWindow#mainWindow {{
                background-color: {self.theme.main_window_background};
                color: {self.theme.main_window_text};
            }}
            #mainWindowRoot {{
                background-color: {self.theme.main_window_background};
                color: {self.theme.main_window_text};
            }}
            QMenuBar {{
                background-color: {self.theme.main_window_background};
                color: {self.theme.main_window_text};
            }}
            QMenuBar::item:selected {{
                background-color: {self.theme.main_window_panel_background};
            }}
            QMenu {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_text};
                border: 1px solid {self.theme.main_window_panel_border};
            }}
            QMenu::item {{
                background-color: transparent;
                color: {self.theme.main_window_text};
            }}
            QMenu::item:selected {{
                background-color: {self.theme.wizard_step_pending_fill};
                color: {self.theme.main_window_text};
            }}
            QMenu::separator {{
                height: 1px;
                margin: 4px 8px;
                background-color: {self.theme.main_window_panel_border};
            }}
            QStatusBar {{
                background-color: {self.theme.main_window_background};
                color: {self.theme.main_window_text};
            }}
            QStatusBar::item {{
                border: none;
            }}
            QDialog,
            QMessageBox {{
                background-color: {self.theme.main_window_background};
                color: {self.theme.main_window_text};
            }}
            QDialog QLabel,
            QMessageBox QLabel {{
                color: {self.theme.main_window_text};
                background: transparent;
            }}
            QDialog QListWidget,
            QDialog QComboBox,
            QDialog QDoubleSpinBox,
            QMessageBox QListWidget,
            QMessageBox QComboBox,
            QMessageBox QDoubleSpinBox {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_text};
                border: 1px solid {self.theme.main_window_panel_border};
            }}
            QComboBox QAbstractItemView,
            QDialog QComboBox QAbstractItemView,
            QMessageBox QComboBox QAbstractItemView {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_text};
                border: 1px solid {self.theme.main_window_panel_border};
                selection-background-color: {self.theme.wizard_step_pending_fill};
                selection-color: {self.theme.main_window_text};
            }}
            QDialog QPushButton,
            QMessageBox QPushButton {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_text};
                border: 1px solid {self.theme.main_window_panel_border};
                padding: 4px 10px;
            }}
            QDialog QPushButton:disabled,
            QMessageBox QPushButton:disabled {{
                color: {self.theme.main_window_muted_text};
            }}
            #stepBarScroll,
            #stepBarScroll > QWidget,
            #stepBarScroll QScrollBar:horizontal,
            #stepBarScroll QScrollBar:vertical,
            #stepBarScroll QWidget#qt_scrollarea_viewport {{
                background-color: {self.theme.main_window_background};
                color: {self.theme.main_window_text};
                border: none;
            }}
            #sidebarPanel {{
                background-color: {self.theme.main_window_sidebar_background};
                color: {self.theme.main_window_text};
            }}
            #sidebarPanel QLabel {{
                color: {self.theme.main_window_text};
                background: transparent;
            }}
            #sidebarPanelCard,
            #sidebarPanel QListWidget,
            #sidebarPanel QComboBox,
            #sidebarPanel QDoubleSpinBox {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_text};
                border: 1px solid {self.theme.main_window_panel_border};
            }}
            #sidebarPanel QPushButton {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_text};
                border: 1px solid {self.theme.main_window_panel_border};
                padding: 4px 10px;
            }}
            #sidebarPanel QPushButton:disabled {{
                color: {self.theme.main_window_muted_text};
            }}
            """

    def _apply_window_theme(self) -> None:
        stylesheet = self._theme_stylesheet()
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet)
            return
        self.setStyleSheet(stylesheet)

    def _open_theme_settings(self) -> None:
        options = discover_theme_options(self._themes_directory)
        if not options:
            QMessageBox.information(
                self,
                "Theme settings",
                f"No theme files were found in {self._themes_directory}.",
            )
            return

        dialog = ThemeSettingsDialog(
            self._themes_directory,
            options,
            self.config.theme_file,
            self,
        )
        if dialog.exec() == 0:
            return

        selected_theme_file = dialog.selected_theme_file()
        if not selected_theme_file or selected_theme_file == self.config.theme_file:
            return

        theme_path = self._themes_directory / selected_theme_file
        theme, warnings = load_theme(theme_path)
        if warnings:
            QMessageBox.warning(
                self,
                "Theme warnings",
                "\n".join(warnings),
            )

        self.config.theme_file = selected_theme_file
        self.config.theme = theme
        self._save_config(self.config)
        self._replace_theme(theme)
        self.statusBar().showMessage(f"Theme changed to {theme.theme_info.name}", 3000)

    def _replace_theme(self, new_theme: AppTheme) -> None:
        for field in fields(AppTheme):
            setattr(self.theme, field.name, copy.deepcopy(getattr(new_theme, field.name)))

        for label in self._muted_labels:
            self._apply_muted_text_style(label)

        self._apply_window_theme()
        self.step_bar.update()
        self.preview.update()
        self.mirror_preview.update()
        self.toolpath_viewer.update()
        self._sync_ui()

    def _new_project(self) -> None:
        if not self._confirm_discard_or_save_changes():
            return
        self.project.reset()
        self.project.set_current_step(1)
        self.imported_gerbers = []
        self.imported_drills = []
        self.tool_library = self._default_tool_library()
        self.generated_documents = {}
        self.has_unsaved_changes = False
        self.statusBar().showMessage("Started new project", 3000)
        self._sync_ui()

    def _open_project(self) -> None:
        if not self._confirm_discard_or_save_changes():
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self._load_project_dialog_directory(),
            "PCB CAM Project (*.mpcbcam.yaml *.yaml);;All Files (*)",
        )
        if not selected:
            return
        self._remember_project_load_path(Path(selected))
        self.load_project_path(
            Path(selected),
            show_message=True,
            show_errors=True,
            auto_resume=False,
        )

    def _close_project(self) -> None:
        if not self._confirm_discard_or_save_changes():
            return
        self.project.reset()
        self.project.set_current_step(0)
        self.imported_gerbers = []
        self.imported_drills = []
        self.tool_library = self._default_tool_library()
        self.generated_documents = {}
        self.has_unsaved_changes = False
        self.statusBar().showMessage("Closed project", 3000)
        self._sync_ui()

    def _save_project(self) -> bool:
        if self.project.project_path is None:
            return self._save_project_as()
        return self._write_project(self.project.project_path)

    def _save_project_as(self) -> bool:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            str(Path(self._save_dialog_directory()) / "project.mpcbcam.yaml"),
            "PCB CAM Project (*.mpcbcam.yaml);;YAML Files (*.yaml)",
        )
        if not selected:
            return False
        path = Path(selected)
        if path.suffix.lower() != ".yaml":
            path = path.with_suffix(".mpcbcam.yaml")
        self._remember_save_path(path)
        return self._write_project(path)

    def _write_project(self, path: Path) -> bool:
        try:
            self.project.save_to_path(path)
        except Exception as exc:
            logger.exception("Failed to save project: %s", path)
            QMessageBox.critical(self, "Failed to save project", str(exc))
            return False
        self._remember_recent_project(path)
        self.has_unsaved_changes = False
        self.statusBar().showMessage(f"Saved {path.name}", 3000)
        self._sync_ui()
        return True

    def load_project_path(
        self,
        path: Path,
        *,
        show_message: bool = False,
        show_errors: bool = False,
        auto_resume: bool = False,
    ) -> bool:
        try:
            project = PcbProject.load_from_path(path)
            imported_gerbers = self._parse_gerber_paths(project.gerber_paths)
            imported_drills = self._parse_drill_paths(project.drill_paths)
        except Exception as exc:
            logger.exception("Failed to open project: %s", path)
            if show_errors:
                QMessageBox.critical(self, "Failed to open project", str(exc))
            return False

        self.project = project
        saved_step = min(
            self.project.current_step_index,
            self.IMPLEMENTED_STEP_COUNT - 1,
        )
        target_step = saved_step if auto_resume else max(1, saved_step)
        self.project.set_current_step(target_step)
        self.imported_gerbers = imported_gerbers
        self.imported_drills = imported_drills
        self._load_tool_library_from_project(show_errors=show_errors)
        self._remember_recent_project(path)
        self.generated_documents = {}
        self.has_unsaved_changes = False
        if show_message:
            self.statusBar().showMessage(f"Opened {Path(path).name}", 3000)
        self._sync_ui()
        return True

    def _handle_step_selected(self, index: int) -> None:
        if not self.project.can_navigate_to(index, self.IMPLEMENTED_STEP_COUNT):
            self.statusBar().showMessage(
                "That wizard step is locked. Move forward sequentially after changes.",
                4000,
            )
            return
        if index == self.project.current_step_index:
            return
        self.project.set_current_step(index)
        self._mark_project_dirty()
        self._sync_ui()

    def _go_back(self) -> None:
        if self.project.current_step_index <= 0:
            return
        self.project.set_current_step(self.project.current_step_index - 1)
        self._mark_project_dirty()
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
                self._validation_message(current),
            )
            return
        self.project.completed_steps.add(current)
        self.project.clear_dirty_state_through(current + 1)
        self.project.set_current_step(current + 1)
        self._mark_project_dirty()
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

        self.imported_gerbers = self._merge_imported_gerbers(imports)
        if self.project.replace_gerber_paths(
            [item.path for item in self.imported_gerbers]
        ):
            self._mark_project_dirty()
        self.project.set_current_step(1)
        self.statusBar().showMessage(
            f"Imported {len(imports)} Gerber file(s)",
            3000,
        )
        self._sync_ui()

    def _merge_imported_gerbers(
        self,
        imports: list[ImportedGerberFile],
    ) -> list[ImportedGerberFile]:
        merged = list(self.imported_gerbers)
        index_by_path = {
            item.path.resolve(): index for index, item in enumerate(merged)
        }

        for imported in imports:
            resolved_path = imported.path.resolve()
            existing_index = index_by_path.get(resolved_path)
            if existing_index is None:
                index_by_path[resolved_path] = len(merged)
                merged.append(imported)
                continue
            merged[existing_index] = imported

        return merged

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
        if self.project.replace_drill_paths(paths):
            self._mark_project_dirty()
        self.project.set_current_step(2)
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
        if self.project.replace_gerber_paths([item.path for item in remaining]):
            self._mark_project_dirty()
        self.project.set_current_step(1)
        self._sync_ui()

    def _clear_gerbers(self) -> None:
        self.imported_gerbers = []
        if self.project.replace_gerber_paths([]):
            self._mark_project_dirty()
        self.project.set_current_step(1)
        self._sync_ui()

    def _remove_selected_drills(self) -> None:
        selected_paths = {
            item.data(Qt.ItemDataRole.UserRole) for item in self.drill_list.selectedItems()
        }
        if not selected_paths:
            return
        remaining = [item for item in self.imported_drills if str(item.path) not in selected_paths]
        self.imported_drills = remaining
        if self.project.replace_drill_paths([item.path for item in remaining]):
            self._mark_project_dirty()
        self.project.set_current_step(min(self.project.current_step_index, 2))
        self._sync_ui()

    def _clear_drills(self) -> None:
        self.imported_drills = []
        if self.project.replace_drill_paths([]):
            self._mark_project_dirty()
        self.project.set_current_step(min(self.project.current_step_index, 2))
        self._sync_ui()

    def _browse_tool_library(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load tools.yaml",
            self._load_dialog_directory(),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not selected:
            return
        path = Path(selected)
        self._remember_load_paths([path])
        self._load_tool_library(path)

    def _clear_tool_library(self) -> None:
        self.tool_library = None
        if self.project.set_tool_library_path(None):
            self._mark_project_dirty()
        for role in list(self.project.selected_tools):
            if self.project.selected_tools[role]:
                self.project.selected_tools[role] = ""
                self._mark_project_dirty()
        self._sync_ui()

    def _tool_selection_changed(self, role: str, combo: QComboBox) -> None:
        tool_id = str(combo.currentData() or "")
        if self.project.set_selected_tool(role, tool_id):
            self._mark_project_dirty()
        self._sync_ui()

    def _layer_assignment_changed(self, role: str, combo: QComboBox) -> None:
        raw_path = combo.currentData()
        path = None if not raw_path else Path(str(raw_path))
        if self.project.set_layer_assignment(role, path):
            self._mark_project_dirty()
        self._sync_ui()

    def _mirror_edge_changed(self, edge: str, checked: bool) -> None:
        if not checked:
            return
        if self.project.set_mirror_flip_edge(edge):
            self._mark_project_dirty()
        self._sync_ui()

    def _add_alignment_hole(self) -> None:
        holes = list(self.project.alignment_holes)
        holes.append(
            AlignmentHole(
                edge=self.alignment_edge_combo.currentText(),
                offset_along_edge=self.alignment_offset_along_spin.value(),
                offset_from_edge=self.alignment_offset_from_edge_spin.value(),
                diameter=self.alignment_diameter_spin.value(),
            )
        )
        if self.project.replace_alignment_holes(holes):
            self._mark_project_dirty()
        self._sync_ui()

    def _remove_selected_alignment_holes(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.alignment_hole_list.selectedIndexes()},
            reverse=True,
        )
        if not selected_rows:
            return
        holes = list(self.project.alignment_holes)
        for row in selected_rows:
            if 0 <= row < len(holes):
                holes.pop(row)
        if self.project.replace_alignment_holes(holes):
            self._mark_project_dirty()
        self._sync_ui()

    def _clear_alignment_holes(self) -> None:
        if self.project.replace_alignment_holes([]):
            self._mark_project_dirty()
        self._sync_ui()

    def _generate_front_isolation(self) -> None:
        gerber = self._assigned_gerber("front_copper")
        if gerber is None:
            QMessageBox.information(self, "Front isolation", "Assign a front copper Gerber first.")
            return
        tool = self._selected_tool("v_bits")
        if tool is None:
            QMessageBox.information(self, "Front isolation", "Select a V-bit first.")
            return
        try:
            output_path = self._cam_generator().generate_front_isolation(
                gerber,
                output_name="front-isolation.nc",
                tool_tip_diameter=tool.numeric_parameter("tip_diameter", tool.numeric_parameter("diameter", 0.2)),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Front isolation failed", str(exc))
            return
        self._register_generated_output("front_isolation", output_path)

    def _generate_back_isolation(self) -> None:
        gerber = self._assigned_gerber("back_copper")
        if gerber is None:
            QMessageBox.information(self, "Back isolation", "Assign a back copper Gerber first.")
            return
        if self.project.requires_mirror_setup() and not self.project.mirror_flip_edge:
            QMessageBox.information(self, "Back isolation", "Choose a mirror edge first.")
            return
        tool = self._selected_tool("v_bits")
        if tool is None:
            QMessageBox.information(self, "Back isolation", "Select a V-bit first.")
            return
        bounds = self._reference_board_bounds()
        if bounds is None:
            QMessageBox.information(self, "Back isolation", "Board bounds are not available.")
            return
        try:
            output_path = self._cam_generator().generate_back_isolation(
                gerber,
                output_name="back-isolation.nc",
                tool_tip_diameter=tool.numeric_parameter("tip_diameter", tool.numeric_parameter("diameter", 0.2)),
                mirror_edge=self.project.mirror_flip_edge or "left",
                board_bounds=bounds,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Back isolation failed", str(exc))
            return
        self._register_generated_output("back_isolation", output_path)

    def _generate_drilling_operations(self) -> None:
        tool = self._selected_tool("drilling")
        mill_tool = self._selected_tool("milling")
        if tool is None or mill_tool is None:
            QMessageBox.information(
                self,
                "Drilling",
                "Select drilling and milling tools first.",
        )
            return
        holes = []
        for drill in self._active_drills():
            holes.extend(drill.holes)
        holes.extend(self._alignment_hole_positions())
        if not holes:
            QMessageBox.information(
                self,
                "Drilling",
                "There are no drill or alignment holes to generate.",
            )
            return
        try:
            output_path = self._cam_generator().generate_drill_operations(
                holes,
                output_name="drilling.nc",
                drill_diameter=tool.numeric_parameter("diameter", 0.1),
                mill_diameter=mill_tool.numeric_parameter("diameter", 0.1),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Drilling failed", str(exc))
            return
        self._register_generated_output("drilling", output_path)

    def _generate_edge_cuts(self) -> None:
        gerber = self._assigned_gerber("edges")
        if gerber is None or not gerber.outline:
            QMessageBox.information(self, "Edge cuts", "Assign an edge-cuts Gerber with an outline first.")
            return
        mill_tool = self._selected_tool("milling")
        if mill_tool is None:
            QMessageBox.information(self, "Edge cuts", "Select a milling tool first.")
            return
        try:
            output_path = self._cam_generator().generate_edge_cuts(
                gerber.outline,
                output_name="edge-cuts.nc",
                mill_diameter=mill_tool.numeric_parameter("diameter", 0.1),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Edge cut generation failed", str(exc))
            return
        self._register_generated_output("edge_cuts", output_path)

    def _generated_output_selected(self, row: int) -> None:
        if row < 0:
            self.toolpath_viewer.load_document(None)
            return
        item = self.generated_output_list.item(row)
        if item is None:
            return
        operation_key = item.data(Qt.ItemDataRole.UserRole)
        path = self.project.generated_outputs.get(str(operation_key))
        if path is None:
            return
        self._load_generated_document(path)

    def _parse_gerber_paths(self, paths: list[Path]) -> list[ImportedGerberFile]:
        imports = [self.gerber_parser.parse_file(path) for path in paths]
        return sorted(imports, key=lambda item: item.display_name.lower())

    def _parse_drill_paths(self, paths: list[Path]) -> list[ImportedDrillFile]:
        imports = [self.drill_parser.parse_file(path) for path in paths]
        return sorted(imports, key=lambda item: item.display_name.lower())

    def _step_is_valid(self, index: int) -> bool:
        if index == 0:
            return True
        if index == 1:
            return bool(self._active_gerbers())
        if index == 2:
            return not self.imported_drills or bool(self._active_drills())
        if index == 3:
            return self.tool_library is not None and all(
                self.project.selected_tools[role]
                for role in ("drilling", "milling", "v_bits")
            )
        if index == 4:
            return any(self.project.layer_assignments.values())
        if index == 5:
            if not self.project.requires_mirror_setup():
                return True
            return bool(self.project.mirror_flip_edge)
        if index == 6:
            return True
        if index == 7:
            return self._operation_optional_or_generated("front_isolation", "front_copper")
        if index == 8:
            return self._operation_optional_or_generated("back_isolation", "back_copper")
        if index == 9:
            return self._drilling_optional_or_generated()
        if index == 10:
            return self._operation_optional_or_generated("edge_cuts", "edges")
        if index == 11:
            return bool(self.project.generated_outputs)
        return False

    def _validation_message(self, index: int) -> str:
        if index == 0:
            return "Create or load a project before moving on."
        if index == 1:
            if not self.imported_gerbers:
                return "Import at least one Gerber file before moving to the next step."
            return "Select at least one Gerber file before moving to the next step."
        if index == 2:
            if not self.imported_drills:
                return "Drill import is optional. Continue without drill files or import at least one drill file."
            return (
                "Select at least one drill file before moving to the next step, "
                "or clear the drill import if you do not want to use drill data."
            )
        if index == 3:
            return "Load tools.yaml and select a drilling tool, a milling tool, and a V-bit."
        if index == 4:
            return "Assign at least one Gerber file to front copper, back copper, or edges."
        if index == 5:
            return "Select the mirror flip edge before moving to the next step."
        if index == 7:
            return "Generate the front isolation NC file before moving to the next step."
        if index == 8:
            return "Generate the back isolation NC file before moving to the next step."
        if index == 9:
            return "Generate the drilling NC file before moving to the next step."
        if index == 10:
            return "Generate the edge cut NC file before moving to the next step."
        return "Complete the current wizard step before continuing."

    def _sync_ui(self) -> None:
        current = min(self.project.current_step_index, self.IMPLEMENTED_STEP_COUNT - 1)
        self.project.current_step_index = current
        self.step_bar.adjustSize()
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
        self.next_button.setText(
            "Next"
            if current + 1 < self.IMPLEMENTED_STEP_COUNT
            else "Complete"
        )
        self.project_value.setText(
            str(self.project.project_path) if self.project.project_path else "Unsaved project"
        )
        self.gerber_count_value.setText(
            f"Gerber files: {len(self._active_gerbers())}/{len(self.imported_gerbers)} active"
        )
        self.drill_count_value.setText(
            f"Drill files: {len(self._active_drills())}/{len(self.imported_drills)} active"
        )
        self.step_status_value.setText(self._step_status_text())
        self._refresh_list_widgets()
        self._sync_tool_selection_page()
        self._sync_layer_assignment_page()
        self._sync_mirror_setup_page()
        self._sync_alignment_holes_page()
        self._sync_generated_outputs()
        self.preview.load_project_geometry(
            self._active_gerbers(),
            self._active_drills(),
            self._alignment_hole_positions(),
            reference_gerber_files=self.imported_gerbers,
            reference_drill_files=self.imported_drills,
        )
        self.preview_stack.setCurrentIndex(1 if current >= 7 else 0)
        self._update_window_title()
        self._scroll_current_step_into_view()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scroll_current_step_into_view()

    def _refresh_list_widgets(self) -> None:
        self.gerber_list.blockSignals(True)
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
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if self.project.is_gerber_selected(gerber.path)
                else Qt.CheckState.Unchecked
            )
            self.gerber_list.addItem(item)
        self.gerber_list.blockSignals(False)

        self.drill_list.blockSignals(True)
        self.drill_list.clear()
        for drill in self.imported_drills:
            item = QListWidgetItem(f"{drill.display_name} ({len(drill.holes)} holes)")
            item.setToolTip(str(drill.path))
            item.setData(Qt.ItemDataRole.UserRole, str(drill.path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if self.project.is_drill_selected(drill.path)
                else Qt.CheckState.Unchecked
            )
            self.drill_list.addItem(item)
        self.drill_list.blockSignals(False)

    def _step_status_text(self) -> str:
        if self.project.dirty_from_step is not None:
            return (
                f"Changes were made at step {self.project.dirty_from_step + 1}. "
                "Forward navigation is sequential until the wizard is replayed."
            )
        if self.project.current_step_index == 0:
            return "Project setup is active. Create, open, or save a project file before continuing."
        if self.project.current_step_index == 1:
            return "Import Gerber files and leave enabled the layers you want to preview and use later."
        if self.project.current_step_index == 2:
            return (
                "Drill import is optional, but if drill files are loaded at least one must remain "
                "enabled to continue."
            )
        if self.project.current_step_index == 3:
            return "Load tools.yaml and choose the tools needed for drilling, milling, and V-bit operations."
        if self.project.current_step_index == 4:
            return "Assign the imported Gerber files to manufacturing roles."
        if self.project.current_step_index == 5:
            return "Choose the mirror edge only when both front and back copper are assigned."
        if self.project.current_step_index == 6:
            return "Add optional alignment holes and confirm they appear outside the board in preview."
        if self.project.current_step_index == 7:
            return "Generate front copper isolation if a front copper layer is assigned."
        if self.project.current_step_index == 8:
            return "Generate back copper isolation if a back copper layer is assigned."
        if self.project.current_step_index == 9:
            return "Generate drilling for imported drill files and alignment holes."
        if self.project.current_step_index == 10:
            return "Generate edge cut operations if an outline is assigned."
        if self.project.current_step_index == 11:
            return "Select a generated NC file to inspect it in the 3D preview."
        return "Stage 4 of the wizard is active."

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

    def _remember_project_load_path(self, path: Path) -> None:
        directory = path.expanduser().resolve().parent
        self.config.file_locations.last_load_project_directory = str(directory)

    def _remember_save_path(self, path: Path) -> None:
        directory = path.expanduser().resolve().parent
        self.config.file_locations.last_save_directory = str(directory)

    def _remember_recent_project(self, path: Path) -> None:
        resolved_path = path.expanduser().resolve()
        resolved = str(resolved_path)
        self.config.file_locations.last_load_project_directory = str(resolved_path.parent)
        recent_projects = [
            item
            for item in (self.config.file_locations.recent_projects or [])
            if item != resolved
        ]
        recent_projects.insert(0, resolved)
        limit = max(1, self.config.file_locations.recent_project_count)
        self.config.file_locations.recent_projects = recent_projects[:limit]

    def _load_tool_library_from_project(self, *, show_errors: bool) -> None:
        if self.project.tool_library_path is None:
            self.tool_library = self._default_tool_library()
            return
        if not self.project.tool_library_path.exists():
            self.tool_library = None
            if show_errors:
                QMessageBox.warning(
                    self,
                    "Tool library missing",
                    f"Tool library file not found:\n{self.project.tool_library_path}",
                )
            return
        self._load_tool_library(self.project.tool_library_path, show_errors=show_errors)

    def _load_tool_library(self, path: Path, *, show_errors: bool = True) -> None:
        try:
            library = ToolLibrary.load(path)
        except Exception as exc:
            logger.exception("Failed to load tool library: %s", path)
            if show_errors:
                QMessageBox.critical(self, "Failed to load tools.yaml", str(exc))
            return
        self.tool_library = library
        if self.project.set_tool_library_path(path):
            self._mark_project_dirty()
        self._prune_invalid_tool_selections()
        self.statusBar().showMessage(f"Loaded tool library {path.name}", 3000)
        self._sync_ui()

    def _default_tool_library(self) -> ToolLibrary | None:
        config_root = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.GenericConfigLocation
        )
        base_path = Path(config_root) if config_root else Path.home() / ".config"
        candidate = base_path / ORGANIZATION_NAME / APPLICATION_NAME / "tools.yaml"
        if not candidate.exists():
            return None
        try:
            library = ToolLibrary.load(candidate)
        except Exception:
            logger.exception("Failed to auto-load default tools.yaml: %s", candidate)
            return None
        self.project.tool_library_path = candidate.resolve()
        self._prune_invalid_tool_selections()
        return library

    def _prune_invalid_tool_selections(self) -> None:
        if self.tool_library is None:
            for role in self.project.selected_tools:
                self.project.selected_tools[role] = ""
            return
        valid_ids = {
            role: {tool.identifier for tool in self.tool_library.tools_by_category[role]}
            for role in self.project.selected_tools
        }
        for role, selected in self.project.selected_tools.items():
            if selected and selected not in valid_ids[role]:
                self.project.selected_tools[role] = ""

    def _sync_tool_selection_page(self) -> None:
        if self.tool_library is None:
            self.tool_library_value.setText("No tool library loaded")
        else:
            self.tool_library_value.setText(str(self.tool_library.path))
        self._populate_tool_combo(
            self.drilling_tool_combo,
            "drilling",
            self.project.selected_tools["drilling"],
        )
        self._populate_tool_combo(
            self.milling_tool_combo,
            "milling",
            self.project.selected_tools["milling"],
        )
        self._populate_tool_combo(
            self.vbit_tool_combo,
            "v_bits",
            self.project.selected_tools["v_bits"],
        )

    def _populate_tool_combo(
        self,
        combo: QComboBox,
        role: str,
        selected_tool_id: str,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        if self.tool_library is None:
            combo.addItem("No tool library available", "")
            combo.setEnabled(False)
        else:
            combo.setEnabled(True)
            combo.addItem("Select a tool...", "")
            for tool in self.tool_library.tools_by_category[role]:
                combo.addItem(tool.label, tool.identifier)
            index = combo.findData(selected_tool_id)
            combo.setCurrentIndex(0 if index < 0 else index)
        combo.blockSignals(False)

    def _sync_layer_assignment_page(self) -> None:
        self._populate_layer_combo(
            self.front_copper_combo,
            self.project.layer_assignments["front_copper"],
        )
        self._populate_layer_combo(
            self.back_copper_combo,
            self.project.layer_assignments["back_copper"],
        )
        self._populate_layer_combo(
            self.edges_combo,
            self.project.layer_assignments["edges"],
        )

    def _populate_layer_combo(
        self,
        combo: QComboBox,
        selected_path: Path | None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Unassigned", "")
        for gerber in self._active_gerbers():
            combo.addItem(gerber.display_name, str(gerber.path))
        target = "" if selected_path is None else str(selected_path)
        index = combo.findData(target)
        combo.setCurrentIndex(0 if index < 0 else index)
        combo.setEnabled(bool(self._active_gerbers()))
        combo.blockSignals(False)

    def _sync_mirror_setup_page(self) -> None:
        requires_mirror = self.project.requires_mirror_setup()
        self.mirror_requirement_label.setText(
            "Front and back copper are both assigned. Choose the mirror edge."
            if requires_mirror
            else "Only one copper side is assigned. Mirror setup is not required for this project."
        )
        for edge, button in self.mirror_buttons.items():
            button.blockSignals(True)
            button.setEnabled(requires_mirror)
            button.setChecked(requires_mirror and self.project.mirror_flip_edge == edge)
            button.blockSignals(False)
        self.mirror_preview.set_edge(
            self.project.mirror_flip_edge if requires_mirror else ""
        )

    def _sync_alignment_holes_page(self) -> None:
        self.alignment_hole_list.clear()
        for hole, position in zip(
            self.project.alignment_holes,
            self._alignment_hole_positions(),
        ):
            item = QListWidgetItem(
                f"{hole.edge} | along {hole.offset_along_edge:.3f} mm | "
                f"out {hole.offset_from_edge:.3f} mm | dia {hole.diameter:.3f} mm | "
                f"at ({position[0]:.3f}, {position[1]:.3f})"
            )
            self.alignment_hole_list.addItem(item)

    def _alignment_hole_positions(self) -> list[tuple[float, float, float]]:
        reference_bounds = self._reference_board_bounds()
        if reference_bounds is None:
            return []
        positions: list[tuple[float, float, float]] = []
        for hole in self.project.alignment_holes:
            positions.append(
                self._alignment_hole_position_for_bounds(hole, reference_bounds)
            )
        return positions

    def _reference_board_bounds(self) -> tuple[float, float, float, float] | None:
        edge_file = self._assigned_gerber("edges")
        if edge_file is not None and not edge_file.bounds.is_empty:
            return (
                edge_file.bounds.x_min,
                edge_file.bounds.x_max,
                edge_file.bounds.y_min,
                edge_file.bounds.y_max,
            )

        bounds = None
        for role in ("front_copper", "back_copper", "edges"):
            gerber = self._assigned_gerber(role)
            if gerber is None or gerber.bounds.is_empty:
                continue
            if bounds is None:
                bounds = [
                    gerber.bounds.x_min,
                    gerber.bounds.x_max,
                    gerber.bounds.y_min,
                    gerber.bounds.y_max,
                ]
            else:
                bounds[0] = min(bounds[0], gerber.bounds.x_min)
                bounds[1] = max(bounds[1], gerber.bounds.x_max)
                bounds[2] = min(bounds[2], gerber.bounds.y_min)
                bounds[3] = max(bounds[3], gerber.bounds.y_max)
        if bounds is None:
            return None
        return bounds[0], bounds[1], bounds[2], bounds[3]

    def _assigned_gerber(self, role: str) -> ImportedGerberFile | None:
        assigned_path = self.project.layer_assignments.get(role)
        if assigned_path is None:
            return None
        for gerber in self._active_gerbers():
            if gerber.path == assigned_path:
                return gerber
        return None

    def _alignment_hole_position_for_bounds(
        self,
        hole: AlignmentHole,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        x_min, x_max, y_min, y_max = bounds
        if hole.edge == "left":
            return (
                x_min - hole.offset_from_edge,
                y_min + hole.offset_along_edge,
                hole.diameter,
            )
        if hole.edge == "right":
            return (
                x_max + hole.offset_from_edge,
                y_min + hole.offset_along_edge,
                hole.diameter,
            )
        if hole.edge == "top":
            return (
                x_min + hole.offset_along_edge,
                y_max + hole.offset_from_edge,
                hole.diameter,
            )
        return (
            x_min + hole.offset_along_edge,
            y_min - hole.offset_from_edge,
            hole.diameter,
        )

    def _selected_tool(self, role: str):
        if self.tool_library is None:
            return None
        selected_id = self.project.selected_tools.get(role, "")
        for tool in self.tool_library.tools_by_category[role]:
            if tool.identifier == selected_id:
                return tool
        return None

    def _cam_generator(self):
        if self.project.project_path is None:
            raise ValueError("Save the project before generating NC files.")
        try:
            from .cam_generator import CamGenerator
        except ModuleNotFoundError as exc:
            if exc.name == "shapely":
                raise RuntimeError(
                    "CAM generation requires the 'Shapely' package. "
                    "Install dependencies from requirements.txt and restart the application."
                ) from exc
            raise
        return CamGenerator(self.project.project_path.parent / "nc")

    def _register_generated_output(self, operation_key: str, path: Path) -> None:
        self.project.generated_outputs[operation_key] = path.resolve()
        self._mark_project_dirty()
        self._load_generated_document(path)
        self._sync_ui()

    def _load_generated_document(self, path: Path) -> None:
        document = self.gcode_parser.parse_file(path)
        self.generated_documents[str(path.resolve())] = document
        self.toolpath_viewer.load_document(document)

    def _sync_generated_outputs(self) -> None:
        generated_map = {
            "front_isolation": getattr(self, "front_isolation_value", None),
            "back_isolation": getattr(self, "back_isolation_value", None),
            "drilling": getattr(self, "drilling_value", None),
            "edge_cuts": getattr(self, "edge_cuts_value", None),
        }
        for key, label in generated_map.items():
            if label is None:
                continue
            path = self.project.generated_outputs.get(key)
            label.setText(str(path) if path is not None else "Not generated yet")

        self.generated_output_list.blockSignals(True)
        current_key = None
        current_item = self.generated_output_list.currentItem()
        if current_item is not None:
            current_key = current_item.data(Qt.ItemDataRole.UserRole)
        self.generated_output_list.clear()
        for key, title in (
            ("front_isolation", "Front Isolation"),
            ("back_isolation", "Back Isolation"),
            ("drilling", "Drilling"),
            ("edge_cuts", "Edge Cuts"),
        ):
            path = self.project.generated_outputs.get(key)
            if path is None:
                continue
            item = QListWidgetItem(f"{title}: {path.name}")
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.generated_output_list.addItem(item)
            if current_key == key:
                self.generated_output_list.setCurrentItem(item)
        self.generated_output_list.blockSignals(False)
        if self.generated_output_list.count() > 0 and self.generated_output_list.currentRow() < 0:
            self.generated_output_list.setCurrentRow(0)

    def _scroll_current_step_into_view(self) -> None:
        current_rect = self.step_bar.step_bounds(self.project.current_step_index)
        if current_rect.isNull():
            return

        viewport_width = self.step_bar_scroll.viewport().width()
        if viewport_width <= 0:
            return

        horizontal_scroll = self.step_bar_scroll.horizontalScrollBar()
        viewport_left = horizontal_scroll.value()
        viewport_right = viewport_left + viewport_width
        step_left = current_rect.left()
        step_right = current_rect.right()

        if step_left >= viewport_left and step_right <= viewport_right:
            return

        margin = max(24, viewport_width // 10)
        if step_left < viewport_left:
            horizontal_scroll.setValue(max(0, step_left - margin))
            return
        horizontal_scroll.setValue(max(0, step_right - viewport_width + margin))

    def _operation_optional_or_generated(self, operation_key: str, layer_role: str) -> bool:
        if self.project.layer_assignments.get(layer_role) is None:
            return True
        return operation_key in self.project.generated_outputs

    def _drilling_optional_or_generated(self) -> bool:
        if not self._active_drills() and not self.project.alignment_holes:
            return True
        return "drilling" in self.project.generated_outputs

    def _active_gerbers(self) -> list[ImportedGerberFile]:
        return [
            gerber for gerber in self.imported_gerbers if self.project.is_gerber_selected(gerber.path)
        ]

    def _active_drills(self) -> list[ImportedDrillFile]:
        return [
            drill for drill in self.imported_drills if self.project.is_drill_selected(drill.path)
        ]

    def _gerber_item_changed(self, item: QListWidgetItem) -> None:
        raw_path = item.data(Qt.ItemDataRole.UserRole)
        if not raw_path:
            return
        path = Path(str(raw_path))
        selected = item.checkState() == Qt.CheckState.Checked
        if self.project.set_gerber_selected(path, selected):
            self._mark_project_dirty()
            self._sync_ui()

    def _drill_item_changed(self, item: QListWidgetItem) -> None:
        raw_path = item.data(Qt.ItemDataRole.UserRole)
        if not raw_path:
            return
        path = Path(str(raw_path))
        selected = item.checkState() == Qt.CheckState.Checked
        if self.project.set_drill_selected(path, selected):
            self._mark_project_dirty()
            self._sync_ui()

    def _mark_project_dirty(self) -> None:
        self.has_unsaved_changes = True

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
        if response == QMessageBox.StandardButton.Discard:
            return True
        return False

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_discard_or_save_changes():
            event.ignore()
            return
        super().closeEvent(event)
