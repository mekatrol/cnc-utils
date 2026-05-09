from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

from PySide6.QtCore import QSize, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

from .alignment_hole import AlignmentHole
from .app_config import AppConfig
from .app_constants import APPLICATION_NAME, ORGANIZATION_NAME
from .edge_cut_profile import EdgeCutPath
from .edge_cut_validator import EdgeCutValidationResult, validate_edge_segments
from .excellon_file_parser import ExcellonFileParser
from .gcode_parser import GCodeParser
from .gerber_file_parser import GerberFileParser
from .imported_drill_file import ImportedDrillFile
from .imported_gerber_file import ImportedGerberFile
from .mirror_preview_widget import MirrorPreviewWidget
from .nc_origin import (
    NC_ORIGIN_LABELS,
    format_origin_point,
    legacy_origin_point_for_bounds,
    normalize_nc_origin,
)
from .pcb_preview_widget import PcbPreviewWidget
from .pcb_project import PcbProject
from .point_3d import Point3D
from .segment_3d import Segment3D
from .theme import AppTheme, load_theme
from .theme_settings_dialog import ThemeSettingsDialog, discover_theme_options
from .toolpath_document import ToolpathDocument
from .toolpath_stats import ToolpathStats
from .tool_library import ToolLibrary
from .tool_settings_dialog import ToolSettingsDialog
from .viewer import ToolpathViewer
from .wizard_step_bar import WizardStepBar

logger = logging.getLogger(__name__)


class CurrentPageStackedWidget(QStackedWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.currentChanged.connect(lambda _: self.updateGeometry())

    def sizeHint(self) -> QSize:
        current = self.currentWidget()
        if current is not None:
            return current.minimumSizeHint()
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        current = self.currentWidget()
        if current is not None:
            return current.minimumSizeHint()
        return super().minimumSizeHint()


class ResponsiveButtonGrid(QWidget):
    def __init__(self, *, min_column_width: int = 180, parent=None) -> None:
        super().__init__(parent)
        self._min_column_width = min_column_width
        self._buttons: list[QPushButton] = []
        self._column_count = 0
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

    def addButton(self, button: QPushButton) -> None:
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._buttons.append(button)
        self._relayout()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        if not self._buttons:
            return

        available_width = self.width() if self.width() > 0 else self.sizeHint().width()
        column_count = max(1, available_width // self._min_column_width)
        column_count = min(column_count, len(self._buttons))
        if column_count == self._column_count and self._layout.count() == len(
            self._buttons
        ):
            return

        while self._layout.count():
            self._layout.takeAt(0)

        previous_column_count = self._column_count
        self._column_count = column_count
        for index, button in enumerate(self._buttons):
            row = index // column_count
            column = index % column_count
            self._layout.addWidget(button, row, column)
            button.show()
        for column in range(max(previous_column_count, column_count)):
            self._layout.setColumnStretch(column, 1 if column < column_count else 0)
        self._layout.invalidate()
        self.updateGeometry()


class MainWindow(QMainWindow):
    STEP_TITLES = [
        "Project",
        "Stock Definition",
        "Gerber Import",
        "Drill Import",
        "Alignment",
        "Front Isolation",
        "Back Isolation",
        "Drilling",
        "Edge Cuts",
        "NC Preview",
    ]
    FILE_ALIGNMENT_LABELS = {
        "top_left": "Top Left",
        "top_center": "Top Mid",
        "top_right": "Top Right",
        "center_left": "Center Left",
        "center": "Center Mid",
        "center_right": "Center Right",
        "bottom_left": "Bottom Left",
        "bottom_center": "Bottom Mid",
        "bottom_right": "Bottom Right",
    }
    IMPLEMENTED_STEP_COUNT = 10

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
        self._apply_default_project_settings()
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
        self._last_sidebar_page_index: int | None = None
        self._edge_cut_validation_result = EdgeCutValidationResult()
        self._selected_edge_cut_polygon_indices: set[int] = set()
        self._selected_edge_cut_profile_index: int | None = None
        self._generated_edge_cut_preview_paths: list[list[tuple[float, float]]] = []
        self._alignment_preview_row_map: list[int] = []
        self.operation_tool_combos: dict[str, tuple[QComboBox, str]] = {}
        self.tool_library_value_labels: list[QLabel] = []
        self._hidden_generated_output_keys: set[str] = set()
        self._loaded_generated_output_keys: tuple[str, ...] = ()
        self._loaded_generated_output_paths: tuple[str, ...] = ()

        self.preview = PcbPreviewWidget(self.theme)
        self.preview.origin_selected.connect(self._set_origin_location)
        self.preview.edge_polygon_selected.connect(self._select_edge_cut_polygon)
        self.preview.alignment_hole_selected.connect(self._select_alignment_hole)
        self.preview.alignment_hole_position_selected.connect(
            self._add_alignment_hole_at_position
        )
        self.toolpath_viewer = ToolpathViewer(self.theme)
        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.preview)
        self.preview_stack.addWidget(self.toolpath_viewer)
        self.preview_panel = QWidget()
        self.preview_panel.setObjectName("previewPanel")
        preview_layout = QVBoxLayout(self.preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        self.preview_toolbar = QWidget()
        self.preview_toolbar.setObjectName("previewToolbar")
        toolbar_layout = QHBoxLayout(self.preview_toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        self.preview_mode_label = QLabel("Mirror View")
        self.preview_mode_label.setObjectName("previewModeLabel")
        self.mirror_preview_mode_combo = QComboBox()
        self.mirror_preview_mode_combo.addItem("Overlay", "overlay")
        self.mirror_preview_mode_combo.addItem("Side by side", "side_by_side")
        self.mirror_preview_mode_combo.currentIndexChanged.connect(
            self._mirror_preview_mode_changed
        )
        self.preview_side_label = QLabel("View Side")
        self.preview_side_label.setObjectName("previewModeLabel")
        self.preview_side_group = QButtonGroup(self)
        self.preview_side_front_radio = QRadioButton("Front")
        self.preview_side_back_radio = QRadioButton("Back mirror")
        self.preview_side_group.addButton(self.preview_side_front_radio)
        self.preview_side_group.addButton(self.preview_side_back_radio)
        self.preview_side_front_radio.toggled.connect(
            lambda checked: self._preview_side_changed("front", checked)
        )
        self.preview_side_back_radio.toggled.connect(
            lambda checked: self._preview_side_changed("back", checked)
        )
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.preview_mode_label)
        toolbar_layout.addWidget(self.mirror_preview_mode_combo)
        toolbar_layout.addWidget(self.preview_side_label)
        toolbar_layout.addWidget(self.preview_side_front_radio)
        toolbar_layout.addWidget(self.preview_side_back_radio)
        preview_layout.addWidget(self.preview_toolbar)
        preview_layout.addWidget(self.preview_stack, 1)
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
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
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

    def _apply_default_project_settings(self) -> None:
        self.project.stock_origin = normalize_nc_origin(self.config.default_nc_origin)
        self.project.file_alignment_horizontal_offset = max(
            0.0, self.config.default_file_alignment_horizontal_offset
        )
        self.project.file_alignment_vertical_offset = max(
            0.0, self.config.default_file_alignment_vertical_offset
        )

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
        splitter.addWidget(self.preview_panel)
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

        self.page_stack = CurrentPageStackedWidget()
        self.page_stack.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        self.page_stack.addWidget(self._build_project_page())
        self.page_stack.addWidget(self._build_stock_definition_page())
        self.page_stack.addWidget(self._build_gerber_page())
        self.page_stack.addWidget(self._build_drill_page())
        self.page_stack.addWidget(self._build_alignment_holes_page())
        self.page_stack.addWidget(
            self._build_operation_page(
                "Step 6: Front Isolation",
                "Generate front copper isolation G-code from the assigned front copper layer.",
                "Generate Front Isolation",
                "_generate_front_isolation",
                "front_isolation",
                [("V-bit", "front_isolation", "v_bits")],
            )
        )
        self.page_stack.addWidget(
            self._build_operation_page(
                "Step 7: Back Isolation",
                "Generate back copper isolation G-code from the assigned back copper layer.",
                "Generate Back Isolation",
                "_generate_back_isolation",
                "back_isolation",
                [("V-bit", "back_isolation", "v_bits")],
            )
        )
        self.page_stack.addWidget(
            self._build_operation_page(
                "Step 8: Drilling",
                "Generate drilling G-code for imported drill holes and optional alignment holes.",
                "Generate Drill Operations",
                "_generate_drilling_operations",
                "drilling",
                [
                    ("Drilling tool", "drilling_drill", "drilling"),
                    ("Milling tool", "drilling_mill", "milling"),
                ],
            )
        )
        self.page_stack.addWidget(self._build_edge_cuts_page())
        self.page_stack.addWidget(self._build_nc_preview_page())
        if self.page_stack.layout() is not None:
            self.page_stack.layout().setAlignment(Qt.AlignmentFlag.AlignTop)

        self.page_scroll = QScrollArea()
        self.page_scroll.setObjectName("pageScroll")
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.page_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        page_scroll_content = QWidget()
        page_scroll_content.setObjectName("pageScrollContent")
        page_scroll_layout = QVBoxLayout(page_scroll_content)
        page_scroll_layout.setContentsMargins(0, 0, 12, 0)
        page_scroll_layout.setSpacing(0)
        page_scroll_layout.addWidget(self.page_stack, 0, Qt.AlignmentFlag.AlignTop)
        page_scroll_layout.addStretch(1)
        self.page_scroll.setWidget(page_scroll_content)
        layout.addWidget(self.page_scroll, 1)

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
        heading.setObjectName("pageHeading")
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
            "Use this step to create or reopen a project file. Stock definition happens in the next step."
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

        heading = QLabel("Step 3: Import Gerber")
        heading.setObjectName("pageHeading")
        body = QLabel(
            "Import Gerber files, assign front/back/edge roles, and set mirroring "
            "here so the rest of the wizard uses the same board view."
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
        self.gerber_list.setMinimumHeight(150)
        self.gerber_list.setMaximumHeight(210)

        self.gerber_hint = QLabel("Import at least one Gerber file to continue.")
        self.gerber_hint.setWordWrap(True)
        self._apply_muted_text_style(self.gerber_hint)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addLayout(button_row)
        layout.addWidget(self.gerber_list, 1)
        layout.addWidget(self.gerber_hint)
        layout.addWidget(self._build_layer_assignment_section())
        layout.addWidget(self._build_mirror_setup_section())
        return page

    def _build_drill_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 4: Import Excellon Drill Files")
        heading.setObjectName("pageHeading")
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
        self.drill_list.setMinimumHeight(150)
        self.drill_list.setMaximumHeight(210)

        self.drill_hint = QLabel(
            "You can continue without drill files. Later drill-operation "
            "steps will use the project data saved here."
        )
        self.drill_hint.setWordWrap(True)
        self._apply_muted_text_style(self.drill_hint)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addLayout(button_row)
        layout.addWidget(self.drill_list)
        layout.addWidget(self.drill_hint)
        layout.addStretch(1)
        return page

    def _build_layer_assignment_section(self) -> QWidget:
        form_card = QFrame()
        form_card.setFrameShape(QFrame.Shape.StyledPanel)
        form_card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(form_card)
        layout = QVBoxLayout(form_card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        heading = QLabel("Layer Assignment")
        heading.setObjectName("sectionHeading")
        body = QLabel("Each role must use a different Gerber file.")
        body.setWordWrap(True)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self.front_copper_combo = QComboBox()
        self.front_copper_combo.currentIndexChanged.connect(
            lambda _: self._layer_assignment_changed(
                "front_copper", self.front_copper_combo
            )
        )
        self.back_copper_combo = QComboBox()
        self.back_copper_combo.currentIndexChanged.connect(
            lambda _: self._layer_assignment_changed(
                "back_copper", self.back_copper_combo
            )
        )
        self.edges_combo = QComboBox()
        self.edges_combo.currentIndexChanged.connect(
            lambda _: self._layer_assignment_changed("edges", self.edges_combo)
        )
        form.addRow("Front copper", self.front_copper_combo)
        form.addRow("Back copper", self.back_copper_combo)
        form.addRow("Edges", self.edges_combo)

        self.layer_assignment_hint = QLabel(
            "At least one of front copper, back copper, or edges is required."
        )
        self.layer_assignment_hint.setWordWrap(True)
        self._apply_muted_text_style(self.layer_assignment_hint)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addLayout(form)
        layout.addWidget(self.layer_assignment_hint)
        return form_card

    def _build_mirror_setup_section(self) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        heading = QLabel("Mirror Setup")
        heading.setObjectName("sectionHeading")
        body = QLabel("Choose the board-flip edge when both copper sides are assigned.")
        body.setWordWrap(True)
        self.mirror_requirement_label = QLabel()
        self.mirror_requirement_label.setWordWrap(True)

        button_row = QHBoxLayout()
        self.mirror_button_group = QButtonGroup(self)
        self.mirror_buttons: dict[str, QRadioButton] = {}
        for edge, label in (
            ("", "None"),
            ("left", "Left"),
            ("top", "Top"),
            ("right", "Right"),
            ("bottom", "Bottom"),
        ):
            button = QRadioButton(label)
            button.toggled.connect(
                lambda checked, selected=edge: self._mirror_edge_changed(
                    selected, checked
                )
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
        return card

    def _build_alignment_holes_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 5: Alignment")
        heading.setObjectName("pageHeading")
        body = QLabel(
            "Click a stock reference point in the preview to align imported files as "
            "a set, then click a grid intersection to add mirrored alignment holes."
        )
        body.setWordWrap(True)

        self.file_alignment_value = QLabel()
        self.file_alignment_value.setWordWrap(True)
        self._apply_muted_text_style(self.file_alignment_value)

        form_card = QFrame()
        form_card.setFrameShape(QFrame.Shape.StyledPanel)
        form_card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(form_card)
        form = QFormLayout(form_card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(10)
        offset_validator = QDoubleValidator(0.0, 10000.0, 3, self)
        offset_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.file_alignment_horizontal_offset_input = QLineEdit()
        self.file_alignment_horizontal_offset_input.setValidator(offset_validator)
        self.file_alignment_horizontal_offset_input.editingFinished.connect(
            lambda: self._file_alignment_offset_changed(
                "horizontal", self.file_alignment_horizontal_offset_input
            )
        )
        self.file_alignment_vertical_offset_input = QLineEdit()
        self.file_alignment_vertical_offset_input.setValidator(offset_validator)
        self.file_alignment_vertical_offset_input.editingFinished.connect(
            lambda: self._file_alignment_offset_changed(
                "vertical", self.file_alignment_vertical_offset_input
            )
        )
        grid_validator = QDoubleValidator(0.1, 10000.0, 3, self)
        grid_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.alignment_grid_size_input = QLineEdit()
        self.alignment_grid_size_input.setValidator(grid_validator)
        self.alignment_grid_size_input.editingFinished.connect(
            self._alignment_grid_size_changed
        )
        self.alignment_mirror_combo = QComboBox()
        self.alignment_mirror_combo.addItem("Horizontal (left/right)", "horizontal")
        self.alignment_mirror_combo.addItem("Vertical (top/bottom)", "vertical")
        self.alignment_mirror_combo.currentIndexChanged.connect(
            self._selected_alignment_hole_mirror_changed
        )
        diameter_validator = QDoubleValidator(0.01, 100.0, 3, self)
        diameter_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.alignment_diameter_input = QLineEdit()
        self.alignment_diameter_input.setValidator(diameter_validator)
        self.alignment_diameter_input.setText("1.000")
        form.addRow(
            "Horizontal edge offset (mm)", self.file_alignment_horizontal_offset_input
        )
        form.addRow(
            "Vertical edge offset (mm)", self.file_alignment_vertical_offset_input
        )
        form.addRow("Grid size (mm)", self.alignment_grid_size_input)
        form.addRow("Mirror direction", self.alignment_mirror_combo)
        form.addRow("Hole diameter (mm)", self.alignment_diameter_input)

        button_row = QHBoxLayout()
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected_alignment_holes)
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self._clear_alignment_holes)
        button_row.addWidget(remove_button)
        button_row.addWidget(clear_button)

        self.alignment_hole_list = QListWidget()
        self.alignment_hole_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )
        self.alignment_hole_list.itemChanged.connect(self._alignment_hole_item_changed)
        self.alignment_hole_list.itemSelectionChanged.connect(
            self._alignment_hole_selection_changed
        )
        self.alignment_holes_hint = QLabel(
            "Alignment holes are optional. Checked rows are shown in green "
            "and included in NC output."
        )
        self.alignment_holes_hint.setWordWrap(True)
        self._apply_muted_text_style(self.alignment_holes_hint)
        self.alignment_drill_value = QLabel("Alignment drill path: not generated yet")
        self.alignment_drill_value.setWordWrap(True)
        self.alignment_mill_value = QLabel("Alignment mill path: not generated yet")
        self.alignment_mill_value.setWordWrap(True)

        generate_drill_button = QPushButton("Generate Alignment Drill")
        generate_drill_button.clicked.connect(self._generate_alignment_drill_operations)
        generate_mill_button = QPushButton("Generate Alignment Mill")
        generate_mill_button.clicked.connect(self._generate_alignment_mill_operations)
        generate_both_button = QPushButton("Generate Both")
        generate_both_button.clicked.connect(self._generate_alignment_hole_operations)
        generate_row = QHBoxLayout()
        generate_row.addWidget(generate_drill_button)
        generate_row.addWidget(generate_mill_button)
        generate_row.addWidget(generate_both_button)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(self.file_alignment_value)
        layout.addWidget(form_card)
        layout.addLayout(button_row)
        layout.addWidget(self.alignment_hole_list, 1)
        layout.addWidget(self.alignment_holes_hint)
        layout.addWidget(
            self._build_operation_tool_card(
                [
                    ("Alignment drill", "alignment_drill", "drilling"),
                    ("Alignment mill", "alignment_mill", "milling"),
                ]
            )
        )
        layout.addLayout(generate_row)
        layout.addWidget(self.alignment_drill_value)
        layout.addWidget(self.alignment_mill_value)
        return page

    def _build_stock_definition_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 2: Stock Definition")
        heading.setObjectName("pageHeading")
        body = QLabel(
            "Define the PCB stock width, height, and thickness, then click a stock "
            "reference point in the preview to set the NC work origin."
        )
        body.setWordWrap(True)

        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(card)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        stock_form = QFormLayout()
        stock_form.setContentsMargins(0, 0, 0, 0)
        stock_form.setSpacing(10)
        self.stock_width_input = QLineEdit()
        self.stock_width_input.setPlaceholderText("mm")
        self.stock_width_input.setValidator(QDoubleValidator(0.0, 10000.0, 3, self))
        self.stock_width_input.editingFinished.connect(
            lambda: self._stock_dimension_changed("width", self.stock_width_input)
        )
        self.stock_height_input = QLineEdit()
        self.stock_height_input.setPlaceholderText("mm")
        self.stock_height_input.setValidator(QDoubleValidator(0.0, 10000.0, 3, self))
        self.stock_height_input.editingFinished.connect(
            lambda: self._stock_dimension_changed("height", self.stock_height_input)
        )
        self.stock_thickness_input = QLineEdit()
        self.stock_thickness_input.setPlaceholderText("mm")
        self.stock_thickness_input.setValidator(QDoubleValidator(0.0, 1000.0, 3, self))
        self.stock_thickness_input.editingFinished.connect(
            lambda: self._stock_dimension_changed(
                "thickness", self.stock_thickness_input
            )
        )
        stock_form.addRow("Width", self.stock_width_input)
        stock_form.addRow("Height", self.stock_height_input)
        stock_form.addRow("Thickness", self.stock_thickness_input)
        card_layout.addLayout(stock_form)

        self.stock_origin_value = QLabel()
        self.stock_origin_value.setWordWrap(True)
        card_layout.addWidget(self.stock_origin_value)

        self.stock_hint = QLabel()
        self.stock_hint.setWordWrap(True)
        self._apply_muted_text_style(self.stock_hint)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(card)
        layout.addWidget(self.stock_hint)
        layout.addStretch(1)
        return page

    def _build_operation_page(
        self,
        heading_text: str,
        body_text: str,
        button_text: str,
        handler_name: str,
        operation_key: str,
        tool_rows: list[tuple[str, str, str]],
    ) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel(heading_text)
        heading.setObjectName("pageHeading")
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
        layout.addWidget(self._build_operation_tool_card(tool_rows))
        layout.addWidget(button)
        layout.addWidget(path_value)
        layout.addStretch(1)
        return page

    def _build_operation_tool_card(
        self, tool_rows: list[tuple[str, str, str]]
    ) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(card)
        form = QFormLayout(card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(10)

        library_value = QLabel("No tool library loaded")
        library_value.setWordWrap(True)
        self.tool_library_value_labels.append(library_value)

        button_row = QHBoxLayout()
        load_button = QPushButton("Load tools.yaml")
        load_button.clicked.connect(self._browse_tool_library)
        clear_button = QPushButton("Clear Tool Library")
        clear_button.clicked.connect(self._clear_tool_library)
        button_row.addWidget(load_button)
        button_row.addWidget(clear_button)

        form.addRow("Library", library_value)
        form.addRow(button_row)
        for label, operation_tool_key, role in tool_rows:
            combo = QComboBox()
            combo.currentIndexChanged.connect(
                lambda _, key=operation_tool_key, widget=combo: (
                    self._operation_tool_changed(key, widget)
                )
            )
            self.operation_tool_combos[operation_tool_key] = (combo, role)
            form.addRow(label, combo)
        return card

    def _build_edge_cuts_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 9: Edge Cuts")
        heading.setObjectName("pageHeading")
        body = QLabel(
            "Each validated contour becomes a path entry. Select a path from the list, then choose "
            "the milling tool, cut depth, step-down, and contour side. Generate one path or all paths, "
            "and show or hide generated paths in the preview."
        )
        body.setWordWrap(True)

        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setObjectName("sidebarPanelCard")
        self._sidebar_panels.append(card)
        form = QFormLayout(card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(10)

        self.edge_cut_selection_value = QLabel("No path selected")
        self.edge_cut_selection_value.setWordWrap(True)
        self.edge_cut_tool_library_value = QLabel("No tool library loaded")
        self.edge_cut_tool_library_value.setWordWrap(True)
        self.tool_library_value_labels.append(self.edge_cut_tool_library_value)
        self.edge_cut_tool_combo = QComboBox()
        self.edge_cut_tool_combo.currentIndexChanged.connect(
            self._edge_cut_tool_changed
        )
        self.edge_cut_depth_input = QLineEdit()
        self.edge_cut_depth_input.setValidator(
            QDoubleValidator(0.0, 1_000_000.0, 3, self)
        )
        self.edge_cut_depth_input.editingFinished.connect(self._edge_cut_depth_changed)
        self.edge_cut_step_down_input = QLineEdit()
        self.edge_cut_step_down_input.setValidator(
            QDoubleValidator(0.0, 1_000_000.0, 3, self)
        )
        self.edge_cut_step_down_input.editingFinished.connect(
            self._edge_cut_step_down_changed
        )
        self.edge_cut_mode_combo = QComboBox()
        self.edge_cut_mode_combo.addItem("None", "none")
        self.edge_cut_mode_combo.addItem("Outside profile", "outside_profile")
        self.edge_cut_mode_combo.addItem("On contour", "on_contour")
        self.edge_cut_mode_combo.addItem("Inside profile", "inside_profile")
        self.edge_cut_mode_combo.currentIndexChanged.connect(
            self._edge_cut_mode_changed
        )
        form.addRow("Selected path", self.edge_cut_selection_value)
        form.addRow("Library", self.edge_cut_tool_library_value)
        form.addRow("Tool bit", self.edge_cut_tool_combo)
        form.addRow("Cut depth", self.edge_cut_depth_input)
        form.addRow("Step-down", self.edge_cut_step_down_input)
        form.addRow("Contour side", self.edge_cut_mode_combo)

        path_list_label = QLabel("Path List")
        path_list_label.setObjectName("sectionHeading")

        path_action_grid = ResponsiveButtonGrid(min_column_width=168)
        self.edge_cut_generate_selected_button = QPushButton("Generate Selected Path")
        self.edge_cut_generate_selected_button.clicked.connect(
            self._generate_selected_edge_cut_path
        )
        generate_all_button = QPushButton("Generate All Paths")
        generate_all_button.clicked.connect(self._generate_edge_cuts)
        self.edge_cut_delete_selected_button = QPushButton("Delete Selected Path")
        self.edge_cut_delete_selected_button.clicked.connect(
            self._delete_selected_edge_cut_path
        )
        delete_all_button = QPushButton("Delete All Paths")
        delete_all_button.clicked.connect(self._delete_all_edge_cut_paths)
        path_action_grid.addButton(self.edge_cut_generate_selected_button)
        path_action_grid.addButton(generate_all_button)
        path_action_grid.addButton(self.edge_cut_delete_selected_button)
        path_action_grid.addButton(delete_all_button)

        self.edge_cut_profile_list = QListWidget()
        self.edge_cut_profile_list.itemChanged.connect(
            self._edge_cut_profile_item_changed
        )
        self.edge_cut_profile_list.currentRowChanged.connect(
            self._edge_cut_profile_selected
        )

        self.edge_cut_hint = QLabel(
            "Select a path from the list to edit it. The preview shows only generated paths as red dotted lines."
        )
        self.edge_cut_hint.setWordWrap(True)
        self._apply_muted_text_style(self.edge_cut_hint)

        self.edge_cuts_value = QLabel("Not generated yet")
        self.edge_cuts_value.setWordWrap(True)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(card)
        layout.addWidget(path_list_label)
        layout.addWidget(path_action_grid)
        layout.addWidget(self.edge_cut_profile_list, 1)
        layout.addWidget(self.edge_cut_hint)
        layout.addWidget(self.edge_cuts_value)
        layout.addStretch(1)
        return page

    def _build_nc_preview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        heading = QLabel("Step 10: NC Preview")
        heading.setObjectName("pageHeading")
        body = QLabel(
            "Generated NC files are shown together in the 3D toolpath viewer. Toggle files in the list to show or hide them."
        )
        body.setWordWrap(True)

        self.generated_output_list = QListWidget()
        self.generated_output_list.itemChanged.connect(
            self._generated_output_item_changed
        )

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
        maximize_action = QAction("Maximize Window", self)
        maximize_action.setShortcut("Ctrl+Shift+M")
        maximize_action.triggered.connect(self.showMaximized)
        view_menu.addAction(maximize_action)
        restore_action = QAction("Restore Window", self)
        restore_action.triggered.connect(self.showNormal)
        view_menu.addAction(restore_action)

        settings_menu = self.menuBar().addMenu("&Settings")
        theme_action = QAction("Theme...", self)
        theme_action.triggered.connect(self._open_theme_settings)
        settings_menu.addAction(theme_action)
        tools_action = QAction("Tools...", self)
        tools_action.triggered.connect(self._open_tool_settings)
        settings_menu.addAction(tools_action)

    def _apply_muted_text_style(self, label: QLabel) -> None:
        if label not in self._muted_labels:
            self._muted_labels.append(label)
        label.setStyleSheet(self._muted_text_style())

    def _apply_error_text_style(self, label: QLabel) -> None:
        label.setStyleSheet(f"color: {self.theme.main_window_error_text};")

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
                color: {self.theme.main_window_input_text};
                border: 1px solid {self.theme.main_window_panel_border};
            }}
            QDialog QRadioButton,
            QMessageBox QRadioButton {{
                color: {self.theme.main_window_text};
            }}
            QDialog QRadioButton::indicator,
            QMessageBox QRadioButton::indicator,
            #sidebarPanel QRadioButton::indicator {{
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 2px solid {self.theme.radio_indicator_border};
                background-color: {self.theme.radio_indicator_fill};
            }}
            QDialog QRadioButton::indicator:checked,
            QMessageBox QRadioButton::indicator:checked,
            #sidebarPanel QRadioButton::indicator:checked {{
                border: 2px solid {self.theme.radio_indicator_checked_border};
                background-color: {self.theme.radio_indicator_checked_fill};
            }}
            QDialog QRadioButton::indicator:disabled,
            QMessageBox QRadioButton::indicator:disabled,
            #sidebarPanel QRadioButton::indicator:disabled {{
                border: 2px solid {self.theme.radio_indicator_disabled_border};
                background-color: {self.theme.radio_indicator_disabled_fill};
            }}
            QComboBox QAbstractItemView,
            QDialog QComboBox QAbstractItemView,
            QMessageBox QComboBox QAbstractItemView {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_input_text};
                border: 1px solid {self.theme.main_window_panel_border};
                selection-background-color: {self.theme.wizard_step_pending_fill};
                selection-color: {self.theme.main_window_input_text};
            }}
            QDialog QPushButton,
            QMessageBox QPushButton {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_button_text};
                border: 1px solid {self.theme.main_window_panel_border};
                padding: 4px 10px;
            }}
            QDialog QPushButton:disabled,
            QDialog QRadioButton:disabled,
            QMessageBox QPushButton:disabled,
            QMessageBox QRadioButton:disabled {{
                color: {self.theme.main_window_disabled_text};
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
            #pageScroll,
            #pageScroll > QWidget,
            #pageScroll QWidget#qt_scrollarea_viewport {{
                background-color: {self.theme.main_window_sidebar_background};
                color: {self.theme.main_window_text};
                border: none;
            }}
            #previewPanel,
            #previewToolbar {{
                background-color: {self.theme.main_window_background};
                color: {self.theme.main_window_text};
            }}
            #previewModeLabel {{
                color: {self.theme.main_window_text};
                font-weight: 600;
            }}
            #sidebarPanel {{
                background-color: {self.theme.main_window_sidebar_background};
                color: {self.theme.main_window_text};
            }}
            #sidebarPanel QLabel {{
                color: {self.theme.main_window_text};
                background: transparent;
            }}
            #sidebarTitle {{
                color: {self.theme.main_window_heading_text};
                font-size: 26px;
                font-weight: 700;
            }}
            #pageHeading {{
                color: {self.theme.main_window_heading_text};
                font-size: 20px;
                font-weight: 700;
            }}
            #sidebarPanelCard,
            #sidebarPanel QListWidget,
            #sidebarPanel QComboBox,
            #sidebarPanel QDoubleSpinBox {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_input_text};
                border: 1px solid {self.theme.main_window_panel_border};
            }}
            #sidebarPanel QRadioButton {{
                color: {self.theme.main_window_text};
            }}
            #sidebarPanel QRadioButton:disabled {{
                color: {self.theme.main_window_disabled_text};
            }}
            #sidebarPanel QComboBox QAbstractItemView {{
                background-color: {self.theme.main_window_panel_background};
                border: 1px solid {self.theme.main_window_panel_border};
                color: {self.theme.main_window_input_text};
                selection-background-color: {self.theme.wizard_step_pending_fill};
                selection-color: {self.theme.main_window_input_text};
            }}
            #previewToolbar QComboBox,
            #previewToolbar QComboBox QAbstractItemView {{
                background-color: {self.theme.main_window_panel_background};
                border: 1px solid {self.theme.main_window_panel_border};
                color: {self.theme.main_window_input_text};
                selection-background-color: {self.theme.wizard_step_pending_fill};
                selection-color: {self.theme.main_window_input_text};
            }}
            #sidebarPanel QPushButton {{
                background-color: {self.theme.main_window_panel_background};
                color: {self.theme.main_window_button_text};
                border: 1px solid {self.theme.main_window_panel_border};
                padding: 4px 10px;
            }}
            #sidebarPanel QPushButton:disabled {{
                color: {self.theme.main_window_disabled_text};
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
            self._themes_directory, options, self.config.theme_file, self
        )
        if dialog.exec() == 0:
            return

        selected_theme_file = dialog.selected_theme_file()
        if not selected_theme_file or selected_theme_file == self.config.theme_file:
            return

        theme_path = self._themes_directory / selected_theme_file
        theme, warnings = load_theme(theme_path)
        if warnings:
            QMessageBox.warning(self, "Theme warnings", "\n".join(warnings))

        self.config.theme_file = selected_theme_file
        self.config.theme = theme
        self._save_config(self.config)
        self._replace_theme(theme)
        self.statusBar().showMessage(f"Theme changed to {theme.theme_info.name}", 3000)

    def _open_tool_settings(self) -> None:
        path = self._editable_tool_library_path()
        if not path.exists():
            tool_library = None
        elif self.tool_library is None or self.tool_library.path != path:
            try:
                tool_library = ToolLibrary.load(path)
            except Exception as exc:
                logger.exception("Failed to load tool library for editing: %s", path)
                QMessageBox.warning(
                    self,
                    "Tool settings",
                    f"{path} is not in the current tool format.\n"
                    f"{exc}\n\nSaving will replace it with the current format.",
                )
                tool_library = None
        else:
            tool_library = self.tool_library

        dialog = ToolSettingsDialog(path, tool_library, self)
        if dialog.exec() == 0 or not dialog.saved():
            return

        self._load_tool_library(path, show_errors=True)
        self.statusBar().showMessage(f"Saved tool library {path.name}", 3000)

    def _replace_theme(self, new_theme: AppTheme) -> None:
        for field in fields(AppTheme):
            setattr(
                self.theme, field.name, copy.deepcopy(getattr(new_theme, field.name))
            )

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
        self._apply_default_project_settings()
        self.project.set_current_step(PcbProject.STEP_STOCK_DEFINITION)
        self.imported_gerbers = []
        self.imported_drills = []
        self.tool_library = self._default_tool_library()
        self.generated_documents = {}
        self._hidden_generated_output_keys = set()
        self._loaded_generated_output_keys = ()
        self._loaded_generated_output_paths = ()
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
            Path(selected), show_message=True, show_errors=True, auto_resume=False
        )

    def _close_project(self) -> None:
        if not self._confirm_discard_or_save_changes():
            return
        self.project.reset()
        self._apply_default_project_settings()
        self.project.set_current_step(0)
        self.imported_gerbers = []
        self.imported_drills = []
        self.tool_library = self._default_tool_library()
        self.generated_documents = {}
        self._hidden_generated_output_keys = set()
        self._loaded_generated_output_keys = ()
        self._loaded_generated_output_paths = ()
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
            self.project.current_step_index, self.IMPLEMENTED_STEP_COUNT - 1
        )
        target_step = saved_step if auto_resume else max(1, saved_step)
        self.project.set_current_step(target_step)
        self.imported_gerbers = imported_gerbers
        self.imported_drills = imported_drills
        self._load_tool_library_from_project(show_errors=show_errors)
        self._remember_recent_project(path)
        self.generated_documents = {}
        self._hidden_generated_output_keys = set()
        self._loaded_generated_output_keys = ()
        self._loaded_generated_output_paths = ()
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
                self, "Wizard step incomplete", self._validation_message(current)
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
        self.project.set_current_step(PcbProject.STEP_GERBER_IMPORT)
        self.statusBar().showMessage(f"Imported {len(imports)} Gerber file(s)", 3000)
        self._sync_ui()

    def _merge_imported_gerbers(
        self, imports: list[ImportedGerberFile]
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

        self.imported_drills = self._merge_imported_drills(imports)
        if self.project.replace_drill_paths(
            [item.path for item in self.imported_drills]
        ):
            self._mark_project_dirty()
        self.project.set_current_step(PcbProject.STEP_DRILL_IMPORT)
        self.statusBar().showMessage(f"Imported {len(imports)} drill file(s)", 3000)
        self._sync_ui()

    def _merge_imported_drills(
        self, imports: list[ImportedDrillFile]
    ) -> list[ImportedDrillFile]:
        merged = list(self.imported_drills)
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

    def _remove_selected_gerbers(self) -> None:
        selected_paths = {
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.gerber_list.selectedItems()
        }
        if not selected_paths:
            return
        remaining = [
            item
            for item in self.imported_gerbers
            if str(item.path) not in selected_paths
        ]
        self.imported_gerbers = remaining
        if self.project.replace_gerber_paths([item.path for item in remaining]):
            self._mark_project_dirty()
        self.project.set_current_step(PcbProject.STEP_GERBER_IMPORT)
        self._sync_ui()

    def _clear_gerbers(self) -> None:
        self.imported_gerbers = []
        if self.project.replace_gerber_paths([]):
            self._mark_project_dirty()
        self.project.set_current_step(PcbProject.STEP_GERBER_IMPORT)
        self._sync_ui()

    def _remove_selected_drills(self) -> None:
        selected_paths = {
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.drill_list.selectedItems()
        }
        if not selected_paths:
            return
        remaining = [
            item
            for item in self.imported_drills
            if str(item.path) not in selected_paths
        ]
        self.imported_drills = remaining
        if self.project.replace_drill_paths([item.path for item in remaining]):
            self._mark_project_dirty()
        self.project.set_current_step(
            min(self.project.current_step_index, PcbProject.STEP_DRILL_IMPORT)
        )
        self._sync_ui()

    def _clear_drills(self) -> None:
        self.imported_drills = []
        if self.project.replace_drill_paths([]):
            self._mark_project_dirty()
        self.project.set_current_step(
            min(self.project.current_step_index, PcbProject.STEP_DRILL_IMPORT)
        )
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
        for operation_key in list(self.project.operation_tools):
            if self.project.operation_tools[operation_key]:
                self.project.operation_tools[operation_key] = ""
                self._mark_project_dirty()
        self._sync_ui()

    def _operation_tool_changed(self, operation_key: str, combo: QComboBox) -> None:
        tool_id = str(combo.currentData() or "")
        if self.project.set_operation_tool(operation_key, tool_id):
            self._mark_project_dirty()
        self._sync_ui()

    def _stock_dimension_changed(self, field_name: str, widget: QLineEdit) -> None:
        text = widget.text().strip()
        if not text:
            value = 0.0
        else:
            try:
                value = float(text)
            except ValueError:
                self._sync_stock_definition_page()
                return
        changed = self.project.set_stock_dimensions(**{field_name: value})
        if not changed:
            self._sync_stock_definition_page()
            return
        self.toolpath_viewer.load_document(None)
        self._mark_project_dirty()
        self._sync_ui()

    def _layer_assignment_changed(self, role: str, combo: QComboBox) -> None:
        raw_path = combo.currentData()
        path = None if not raw_path else Path(str(raw_path))
        if self.project.set_layer_assignment(role, path):
            self._mark_project_dirty()
        if role == "edges":
            self._refresh_edge_cut_validation()
        self._sync_ui()

    def _mirror_edge_changed(self, edge: str, checked: bool) -> None:
        if not checked:
            return
        if self.project.set_mirror_flip_edge(edge):
            self._mark_project_dirty()
        self._sync_ui()

    def _mirror_preview_mode_changed(self, index: int) -> None:
        mode = str(self.mirror_preview_mode_combo.itemData(index) or "side_by_side")
        if self.project.set_mirror_preview_mode(mode):
            self._mark_project_dirty()
        self._sync_ui()

    def _preview_side_changed(self, side: str, checked: bool) -> None:
        if not checked:
            return
        if self.project.set_mirror_view_side(side):
            self._mark_project_dirty()
            self._loaded_generated_output_keys = ()
            self._loaded_generated_output_paths = ()
        self._sync_ui()

    def _file_alignment_offset_changed(self, axis: str, widget: QLineEdit) -> None:
        text = widget.text().strip()
        if not text:
            value = 0.0
        else:
            try:
                value = float(text)
            except ValueError:
                self._sync_alignment_holes_page()
                return
        kwargs = {axis: max(0.0, value)}
        if self.project.set_file_alignment_offsets(
            horizontal=kwargs.get("horizontal"), vertical=kwargs.get("vertical")
        ):
            self.toolpath_viewer.load_document(None)
            self._mark_project_dirty()
        if axis == "horizontal":
            self.config.default_file_alignment_horizontal_offset = kwargs["horizontal"]
        else:
            self.config.default_file_alignment_vertical_offset = kwargs["vertical"]
        self._save_config(self.config)
        self._sync_ui()

    def _alignment_grid_size_changed(self) -> None:
        grid_size = self._alignment_hole_numeric_input(
            self.alignment_grid_size_input, fallback=5.0, minimum=0.1
        )
        if grid_size is None:
            return
        if self.project.set_alignment_grid_size(grid_size):
            self._mark_project_dirty()
        self._sync_ui()

    def _add_alignment_hole_at_position(self, x_pos: float, y_pos: float) -> None:
        if self.project.current_step_index != PcbProject.STEP_ALIGNMENT_HOLES:
            return
        stock_bounds = self._stock_bounds()
        if stock_bounds is None:
            return
        diameter = self._alignment_hole_numeric_input(
            self.alignment_diameter_input, fallback=1.0, minimum=0.01
        )
        if diameter is None:
            return
        x_min, x_max, y_min, y_max = stock_bounds
        mirror_direction = str(
            self.alignment_mirror_combo.currentData() or "horizontal"
        )
        holes = list(self.project.alignment_holes)
        holes.append(
            AlignmentHole(
                x_offset=min(max(x_pos, x_min), x_max),
                y_offset=min(max(y_pos, y_min), y_max),
                diameter=diameter,
                mirror_direction=mirror_direction,
                enabled=True,
            )
        )
        if self.project.replace_alignment_holes(holes):
            self._mark_project_dirty()
        self._sync_ui()
        self.alignment_hole_list.setCurrentRow(len(holes) - 1)

    def _alignment_hole_numeric_input(
        self, widget: QLineEdit, *, fallback: float, minimum: float
    ) -> float | None:
        text = widget.text().strip()
        if not text:
            value = fallback
        else:
            try:
                value = float(text)
            except ValueError:
                widget.setText(f"{fallback:.3f}")
                return None
        value = max(minimum, value)
        widget.setText(f"{value:.3f}")
        return value

    def _current_alignment_hole_diameter(self) -> float:
        text = self.alignment_diameter_input.text().strip()
        if not text:
            return 1.0
        try:
            value = float(text)
        except ValueError:
            return 1.0
        return max(0.01, value)

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

    def _alignment_hole_item_changed(self, item: QListWidgetItem) -> None:
        row = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(row, int) or not 0 <= row < len(self.project.alignment_holes):
            return
        holes = list(self.project.alignment_holes)
        hole = copy.copy(holes[row])
        hole.enabled = item.checkState() == Qt.CheckState.Checked
        holes[row] = hole
        if self.project.replace_alignment_holes(holes):
            self._mark_project_dirty()
        self._sync_ui()

    def _alignment_hole_selection_changed(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.alignment_hole_list.selectedIndexes()}
        )
        if not selected_rows:
            self.alignment_mirror_combo.setEnabled(True)
            self._sync_alignment_preview_selection()
            return
        first_row = selected_rows[0]
        if 0 <= first_row < len(self.project.alignment_holes):
            hole = self.project.alignment_holes[first_row]
            index = self.alignment_mirror_combo.findData(hole.mirror_direction)
            self.alignment_mirror_combo.blockSignals(True)
            self.alignment_mirror_combo.setCurrentIndex(0 if index < 0 else index)
            self.alignment_mirror_combo.blockSignals(False)
        self._sync_alignment_preview_selection()

    def _selected_alignment_preview_index(self) -> int | None:
        selected_rows = sorted(
            {index.row() for index in self.alignment_hole_list.selectedIndexes()}
        )
        if not selected_rows:
            return None
        selected_row = selected_rows[0]
        for preview_index, row in enumerate(self._alignment_preview_row_map):
            if row == selected_row:
                return preview_index
        return None

    def _sync_alignment_preview_selection(self) -> None:
        if self.project.current_step_index != PcbProject.STEP_ALIGNMENT_HOLES:
            return
        stock_bounds = self._stock_bounds()
        self.preview.set_alignment_hole_selection(
            stock_bounds,
            selection_enabled=stock_bounds is not None,
            selected_hole_index=self._selected_alignment_preview_index(),
            grid_spacing=self.project.alignment_grid_size,
            hover_diameter=self._current_alignment_hole_diameter(),
        )

    def _selected_alignment_hole_mirror_changed(self, index: int) -> None:
        direction = str(self.alignment_mirror_combo.itemData(index) or "horizontal")
        selected_rows = sorted(
            {item.row() for item in self.alignment_hole_list.selectedIndexes()}
        )
        if not selected_rows:
            return
        holes = list(self.project.alignment_holes)
        changed = False
        for row in selected_rows:
            if not 0 <= row < len(holes):
                continue
            hole = copy.copy(holes[row])
            if hole.mirror_direction == direction:
                continue
            hole.mirror_direction = direction
            holes[row] = hole
            changed = True
        if changed and self.project.replace_alignment_holes(holes):
            self._mark_project_dirty()
        self._sync_ui()

    def _select_alignment_hole(self, preview_index: int) -> None:
        if not 0 <= preview_index < len(self._alignment_preview_row_map):
            return
        row = self._alignment_preview_row_map[preview_index]
        if 0 <= row < self.alignment_hole_list.count():
            self.alignment_hole_list.setCurrentRow(row)

    def _generate_alignment_hole_operations(self) -> None:
        drill_path = self._generate_alignment_drill_operations(sync_ui=False)
        if drill_path is None:
            self._sync_ui()
            return
        mill_path = self._generate_alignment_mill_operations(sync_ui=False)
        if mill_path is None:
            self._register_generated_output("alignment_drill", drill_path)
            return
        self.project.generated_outputs["alignment_drill"] = drill_path.resolve()
        self.project.generated_outputs["alignment_mill"] = mill_path.resolve()
        self._hidden_generated_output_keys.discard("alignment_drill")
        self._hidden_generated_output_keys.discard("alignment_mill")
        self._loaded_generated_output_keys = ()
        self._loaded_generated_output_paths = ()
        self._mark_project_dirty()
        self._load_generated_document("alignment_mill", mill_path)
        self._sync_ui()

    def _generate_alignment_drill_operations(
        self, *, sync_ui: bool = True
    ) -> Path | None:
        tool = self._operation_tool("alignment_drill", "drilling")
        if tool is None:
            QMessageBox.information(self, "Alignment drill", "Select a drill first.")
            return None
        holes = self._defined_alignment_hole_positions()
        if not holes:
            QMessageBox.information(
                self,
                "Alignment drill",
                "There are no enabled alignment holes to generate.",
            )
            return None
        try:
            output_path = self._cam_generator().generate_alignment_drill_operations(
                holes,
                output_name="alignment-drill.nc",
                drill_diameter=tool.numeric_parameter("diameter", 0.1),
                origin_point=self._current_origin_point_required(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Alignment drill failed", str(exc))
            return None
        if sync_ui:
            self._register_generated_output("alignment_drill", output_path)
        return output_path

    def _generate_alignment_mill_operations(
        self, *, sync_ui: bool = True
    ) -> Path | None:
        tool = self._operation_tool("alignment_mill", "milling")
        if tool is None:
            QMessageBox.information(
                self, "Alignment mill", "Select a milling tool first."
            )
            return None
        holes = self._defined_alignment_hole_positions()
        if not holes:
            QMessageBox.information(
                self,
                "Alignment mill",
                "There are no enabled alignment holes to generate.",
            )
            return None
        try:
            output_path = self._cam_generator().generate_alignment_mill_operations(
                holes,
                output_name="alignment-mill.nc",
                mill_diameter=tool.numeric_parameter("diameter", 0.1),
                origin_point=self._current_origin_point_required(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Alignment mill failed", str(exc))
            return None
        if sync_ui:
            self._register_generated_output("alignment_mill", output_path)
        return output_path

    def _generate_front_isolation(self) -> None:
        gerber = self._assigned_gerber("front_copper")
        if gerber is None:
            QMessageBox.information(
                self, "Front isolation", "Assign a front copper Gerber first."
            )
            return
        tool = self._operation_tool("front_isolation", "v_bits")
        if tool is None:
            QMessageBox.information(self, "Front isolation", "Select a V-bit first.")
            return
        try:
            output_path = self._cam_generator().generate_front_isolation(
                gerber,
                output_name="front-isolation.nc",
                tool_tip_diameter=tool.numeric_parameter(
                    "tip_diameter", tool.numeric_parameter("diameter", 0.2)
                ),
                origin_point=self._current_origin_point_required(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Front isolation failed", str(exc))
            return
        self._register_generated_output("front_isolation", output_path)

    def _generate_back_isolation(self) -> None:
        gerber = self._assigned_gerber("back_copper")
        if gerber is None:
            QMessageBox.information(
                self, "Back isolation", "Assign a back copper Gerber first."
            )
            return
        tool = self._operation_tool("back_isolation", "v_bits")
        if tool is None:
            QMessageBox.information(self, "Back isolation", "Select a V-bit first.")
            return
        bounds = self._reference_board_bounds()
        if bounds is None:
            QMessageBox.information(
                self, "Back isolation", "Board bounds are not available."
            )
            return
        try:
            output_path = self._cam_generator().generate_back_isolation(
                gerber,
                output_name="back-isolation.nc",
                tool_tip_diameter=tool.numeric_parameter(
                    "tip_diameter", tool.numeric_parameter("diameter", 0.2)
                ),
                mirror_edge=self.project.mirror_flip_edge,
                board_bounds=bounds,
                origin_point=self._current_origin_point_required(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Back isolation failed", str(exc))
            return
        self._register_generated_output("back_isolation", output_path)

    def _generate_drilling_operations(self) -> None:
        tool = self._operation_tool("drilling_drill", "drilling")
        mill_tool = self._operation_tool("drilling_mill", "milling")
        if tool is None or mill_tool is None:
            QMessageBox.information(
                self, "Drilling", "Select drilling and milling tools first."
            )
            return
        holes = []
        for drill in self._active_drills():
            holes.extend(drill.holes)
        if not holes:
            QMessageBox.information(
                self, "Drilling", "There are no drill holes to generate."
            )
            return
        try:
            output_path = self._cam_generator().generate_drill_operations(
                holes,
                output_name="drilling.nc",
                drill_diameter=tool.numeric_parameter("diameter", 0.1),
                mill_diameter=mill_tool.numeric_parameter("diameter", 0.1),
                origin_point=self._current_origin_point_required(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Drilling failed", str(exc))
            return
        self._register_generated_output("drilling", output_path)

    def _generate_edge_cuts(self) -> None:
        self._generate_edge_cut_paths(generate_all=True)

    def _generate_selected_edge_cut_path(self) -> None:
        self._generate_edge_cut_paths(generate_all=False)

    def _generate_edge_cut_paths(self, *, generate_all: bool) -> None:
        gerber = self._assigned_gerber("edges")
        if gerber is None:
            QMessageBox.information(
                self, "Edge cuts", "Assign an edge-cuts Gerber first."
            )
            return
        self._refresh_edge_cut_validation()
        if not self._edge_cut_validation_result.is_valid:
            QMessageBox.warning(self, "Edge cuts", self._edge_validation_message())
            return
        if (
            self.tool_library is None
            or not self.tool_library.tools_by_category["milling"]
        ):
            QMessageBox.information(
                self, "Edge cuts", "Load a tool library with milling tools first."
            )
            return
        path_indices = None if generate_all else self._selected_edge_cut_path_indices()
        if not generate_all and not path_indices:
            QMessageBox.information(self, "Edge cuts", "Select a path first.")
            return
        paths = self._resolved_edge_cut_paths(path_indices=path_indices)
        if not paths:
            QMessageBox.information(
                self,
                "Edge cuts",
                "Configure at least one path with a tool and contour side before generating NC.",
            )
            return
        try:
            self._cam_generator().generate_edge_cuts(
                paths,
                output_name=(
                    "edge-cuts.nc"
                    if generate_all
                    else f"edge-cut-path-{paths[0]['path_index'] + 1}.nc"
                ),
                origin_point=self._current_origin_point_required(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Edge cut generation failed", str(exc))
            return
        updated_paths = list(self.project.edge_cut_profiles)
        for path_index in {int(path["path_index"]) for path in paths}:
            current_path = updated_paths[path_index]
            updated_paths[path_index] = EdgeCutPath(
                polygon_keys=list(current_path.polygon_keys),
                mode=current_path.mode,
                tool_id=current_path.tool_id,
                cut_depth=current_path.cut_depth,
                step_down=current_path.step_down,
                generated=True,
                visible=True,
            )
        if self.project.replace_edge_cut_profiles(updated_paths):
            self._mark_project_dirty()
        self._generated_edge_cut_preview_paths = (
            self._visible_generated_edge_cut_preview_paths()
        )
        try:
            generated_path_indices = {
                index
                for index, profile in enumerate(self.project.edge_cut_profiles)
                if profile.generated and profile.mode != "none"
            }
            output_path = self._cam_generator().generate_edge_cuts(
                self._resolved_edge_cut_paths(path_indices=generated_path_indices),
                output_name="edge-cuts.nc",
                origin_point=self._current_origin_point_required(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Edge cut generation failed", str(exc))
            return
        self._register_generated_output("edge_cuts", output_path)

    def _generated_output_item_changed(self, item: QListWidgetItem) -> None:
        operation_key = item.data(Qt.ItemDataRole.UserRole)
        if not operation_key:
            return
        key = str(operation_key)
        if item.checkState() == Qt.CheckState.Checked:
            self._hidden_generated_output_keys.discard(key)
        else:
            self._hidden_generated_output_keys.add(key)
        self._load_selected_generated_documents()

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
            return self._stock_definition_is_valid()
        if index == 2:
            return (
                bool(self._active_gerbers())
                and any(self.project.layer_assignments.values())
                and self._edge_cut_validation_is_valid()
            )
        if index == 3:
            return not self.imported_drills or bool(self._active_drills())
        if index == 4:
            return True
        if index == 5:
            return self._operation_optional_or_generated(
                "front_isolation", "front_copper"
            )
        if index == 6:
            return self._operation_optional_or_generated(
                "back_isolation", "back_copper"
            )
        if index == 7:
            return self._drilling_optional_or_generated()
        if index == 8:
            return (
                self._edge_cut_validation_is_valid()
                and self._operation_optional_or_generated("edge_cuts", "edges")
            )
        if index == 9:
            return bool(self.project.generated_outputs)
        return False

    def _validation_message(self, index: int) -> str:
        if index == 0:
            return "Create or load a project before moving on."
        if index == 1:
            return "Set stock width, height, thickness, and choose an origin point before moving on."
        if index == 2:
            if not self.imported_gerbers:
                return "Import at least one Gerber file before moving to the next step."
            if not self._active_gerbers():
                return "Select at least one Gerber file before moving to the next step."
            if not self._edge_cut_validation_is_valid():
                return self._edge_validation_message()
            return "Assign at least one active Gerber file to front copper, back copper, or edges."
        if index == 3:
            if not self.imported_drills:
                return "Drill import is optional. Continue without drill files or import at least one drill file."
            return (
                "Select at least one drill file before moving to the next step, "
                "or clear the drill import if you do not want to use drill data."
            )
        if index == 5:
            return (
                "Generate the front isolation NC file before moving to the next step."
            )
        if index == 6:
            return "Generate the back isolation NC file before moving to the next step."
        if index == 7:
            return "Generate the drilling NC file before moving to the next step."
        if index == 8:
            if not self._edge_cut_validation_is_valid():
                return self._edge_validation_message()
            return "Generate the edge cut NC file before moving to the next step."
        return "Complete the current wizard step before continuing."

    def _sync_ui(self) -> None:
        current = min(self.project.current_step_index, self.IMPLEMENTED_STEP_COUNT - 1)
        self.project.current_step_index = current
        self._refresh_edge_cut_validation()
        self.step_bar.adjustSize()
        page_changed = self._last_sidebar_page_index != current
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
            "Next" if current + 1 < self.IMPLEMENTED_STEP_COUNT else "Complete"
        )
        self._refresh_list_widgets()
        self._sync_stock_definition_page()
        self._sync_tool_controls()
        self._sync_layer_assignment_page()
        self._sync_mirror_setup_page()
        self._sync_alignment_holes_page()
        self._sync_edge_cut_page()
        self._sync_edge_cut_action_buttons()
        self._sync_generated_outputs()
        assigned_edges = self.project.layer_assignments["edges"]
        stock_bounds = self._stock_bounds()
        show_stock = (
            current >= PcbProject.STEP_STOCK_DEFINITION and stock_bounds is not None
        )
        fit_preview_view = page_changed or current not in {
            PcbProject.STEP_ALIGNMENT_HOLES,
            PcbProject.STEP_EDGE_CUTS,
        }
        self.preview.load_project_geometry(
            self._active_gerbers(),
            self._active_drills(),
            self._alignment_hole_positions(),
            reference_gerber_files=self._aligned_gerbers(self.imported_gerbers),
            reference_drill_files=self._aligned_drills(self.imported_drills),
            fit_view=fit_preview_view,
        )
        self.preview.set_mirror_setup(
            front_copper_path=(
                None
                if current == PcbProject.STEP_STOCK_DEFINITION
                else self.project.layer_assignments["front_copper"]
            ),
            back_copper_path=(
                None
                if current == PcbProject.STEP_STOCK_DEFINITION
                else self.project.layer_assignments["back_copper"]
            ),
            edges_path=(
                None
                if current == PcbProject.STEP_STOCK_DEFINITION
                else self.project.layer_assignments["edges"]
            ),
            board_bounds=stock_bounds or self._reference_board_bounds(),
            mirror_edge=(
                ""
                if current == PcbProject.STEP_STOCK_DEFINITION
                else self.project.mirror_flip_edge
            ),
            preview_mode=(
                "overlay"
                if current == PcbProject.STEP_STOCK_DEFINITION
                else self.project.mirror_preview_mode
            ),
            view_side=self.project.mirror_view_side,
            fit_view=fit_preview_view,
        )
        self.preview.set_edge_validation(
            assigned_edges,
            polygons=self._edge_cut_validation_result.polygons,
            error_segments=self._edge_cut_validation_result.error_segments,
            selection_enabled=current == PcbProject.STEP_EDGE_CUTS,
            selected_polygon_indices=self._selected_edge_cut_polygon_indices,
            polygon_modes={},
            suppress_source_geometry=False,
        )
        self.preview.set_edge_cut_preview_paths(
            self._visible_generated_edge_cut_preview_paths()
            if current == PcbProject.STEP_EDGE_CUTS
            else []
        )
        self.preview.set_origin_marker(
            stock_bounds if show_stock else None,
            (
                self._file_alignment_point()
                if current == PcbProject.STEP_ALIGNMENT_HOLES
                else self._current_origin_point()
            )
            if show_stock
            else None,
            hotspot_points=(
                self._origin_hotspot_points()
                if current == PcbProject.STEP_STOCK_DEFINITION
                else self._file_alignment_hotspot_points()
                if current == PcbProject.STEP_ALIGNMENT_HOLES
                else None
            ),
            selection_enabled=current
            in {PcbProject.STEP_STOCK_DEFINITION, PcbProject.STEP_ALIGNMENT_HOLES},
            marker_label=(
                "Alignment" if current == PcbProject.STEP_ALIGNMENT_HOLES else "(0, 0)"
            ),
            fit_view=fit_preview_view,
        )
        self.preview.set_alignment_hole_selection(
            stock_bounds,
            selection_enabled=current == PcbProject.STEP_ALIGNMENT_HOLES
            and stock_bounds is not None,
            selected_hole_index=(
                self._selected_alignment_preview_index()
                if current == PcbProject.STEP_ALIGNMENT_HOLES
                else None
            ),
            grid_spacing=self.project.alignment_grid_size,
            hover_diameter=self._current_alignment_hole_diameter(),
        )
        self.toolpath_viewer.set_stock_overlay(
            self._toolpath_stock_bounds() if show_stock else None,
            (0.0, 0.0) if show_stock and self._current_origin_point() else None,
            stock_thickness=self.project.stock_thickness if show_stock else 0.0,
            board_bounds=self._toolpath_board_bounds() if show_stock else None,
            alignment_holes=(
                self._toolpath_defined_alignment_holes() if show_stock else []
            ),
        )
        showing_toolpath = (
            (
                current == PcbProject.STEP_ALIGNMENT_HOLES
                and any(
                    key in self.project.generated_outputs
                    for key in ("alignment_drill", "alignment_mill")
                )
            )
            or current >= PcbProject.STEP_FRONT_ISOLATION
            and current != PcbProject.STEP_EDGE_CUTS
        )
        self.preview_stack.setCurrentIndex(1 if showing_toolpath else 0)
        self.preview_toolbar.setVisible(current != PcbProject.STEP_STOCK_DEFINITION)
        self.preview_mode_label.setVisible(not showing_toolpath)
        self.mirror_preview_mode_combo.setVisible(not showing_toolpath)
        self.preview_side_label.setVisible(
            current >= PcbProject.STEP_FRONT_ISOLATION
            or current == PcbProject.STEP_EDGE_CUTS
        )
        self.preview_side_front_radio.setVisible(
            current >= PcbProject.STEP_FRONT_ISOLATION
            or current == PcbProject.STEP_EDGE_CUTS
        )
        self.preview_side_back_radio.setVisible(
            current >= PcbProject.STEP_FRONT_ISOLATION
            or current == PcbProject.STEP_EDGE_CUTS
        )
        self._update_window_title()
        if page_changed:
            QTimer.singleShot(
                0, lambda: self.page_scroll.verticalScrollBar().setValue(0)
            )
        self._last_sidebar_page_index = current
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
            if gerber.segments:
                geometry_summary.append("edge segments")
            if gerber.outline:
                geometry_summary.append("outline")
            if not geometry_summary:
                geometry_summary.append("no visible geometry detected")
            item = QListWidgetItem(
                f"{gerber.display_name} ({', '.join(geometry_summary)})"
            )
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

    def _update_window_title(self) -> None:
        project_name = (
            self.project.project_path.name
            if self.project.project_path is not None
            else "Untitled Project"
        )
        dirty_prefix = "*" if self.has_unsaved_changes else ""
        self.setWindowTitle(f"{dirty_prefix}mekatrol-pcbcam - {project_name}")

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
        candidate = self._default_tool_library_path()
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

    def _default_tool_library_path(self) -> Path:
        config_root = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.GenericConfigLocation
        )
        base_path = Path(config_root) if config_root else Path.home() / ".config"
        return base_path / ORGANIZATION_NAME / APPLICATION_NAME / "tools.yaml"

    def _editable_tool_library_path(self) -> Path:
        if self.tool_library is not None:
            return self.tool_library.path
        if self.project.tool_library_path is not None:
            return self.project.tool_library_path
        return self._default_tool_library_path()

    def _prune_invalid_tool_selections(self) -> None:
        if self.tool_library is None:
            for operation_key in self.project.operation_tools:
                self.project.operation_tools[operation_key] = ""
            return
        valid_ids = {
            role: {
                tool.identifier for tool in self.tool_library.tools_by_category[role]
            }
            for role in ("drilling", "milling", "v_bits")
        }
        operation_roles = {
            "front_isolation": "v_bits",
            "back_isolation": "v_bits",
            "alignment_drill": "drilling",
            "alignment_mill": "milling",
            "drilling_drill": "drilling",
            "drilling_mill": "milling",
        }
        for operation_key, role in operation_roles.items():
            selected = self.project.operation_tools.get(operation_key, "")
            if selected and selected not in valid_ids[role]:
                self.project.operation_tools[operation_key] = ""

    def _sync_tool_controls(self) -> None:
        if self.tool_library is None:
            library_text = "No tool library loaded"
        else:
            library_text = str(self.tool_library.path)
        for label in self.tool_library_value_labels:
            label.setText(library_text)
        for operation_key, (combo, role) in self.operation_tool_combos.items():
            self._populate_tool_combo(
                combo, role, self.project.operation_tools.get(operation_key, "")
            )

    def _populate_tool_combo(
        self, combo: QComboBox, role: str, selected_tool_id: str
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
            "front_copper",
            self.project.layer_assignments["front_copper"],
        )
        self._populate_layer_combo(
            self.back_copper_combo,
            "back_copper",
            self.project.layer_assignments["back_copper"],
        )
        self._populate_layer_combo(
            self.edges_combo, "edges", self.project.layer_assignments["edges"]
        )
        default_hint = (
            "At least one of front copper, back copper, or edges is required."
        )
        if self._edge_cut_validation_is_valid():
            self.layer_assignment_hint.setText(default_hint)
            self._apply_muted_text_style(self.layer_assignment_hint)
            return
        self.layer_assignment_hint.setText(self._edge_validation_message())
        self._apply_error_text_style(self.layer_assignment_hint)

    def _populate_layer_combo(
        self, combo: QComboBox, role: str, selected_path: Path | None
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Unassigned", "")
        assigned_elsewhere = {
            path
            for other_role, path in self.project.layer_assignments.items()
            if other_role != role and path is not None
        }
        for gerber in self._active_gerbers():
            if gerber.path in assigned_elsewhere:
                continue
            combo.addItem(gerber.display_name, str(gerber.path))
        if selected_path is not None and combo.findData(str(selected_path)) < 0:
            assigned_gerber = self._imported_gerber_by_path(selected_path)
            if assigned_gerber is not None:
                label = assigned_gerber.display_name
            else:
                label = f"{selected_path.name} (missing)"
            combo.addItem(label, str(selected_path))
        target = "" if selected_path is None else str(selected_path)
        index = combo.findData(target)
        combo.setCurrentIndex(0 if index < 0 else index)
        combo.setEnabled(bool(self.imported_gerbers))
        combo.blockSignals(False)

    def _refresh_edge_cut_validation(self) -> None:
        gerber = self._assigned_gerber("edges")
        if gerber is None:
            self._edge_cut_validation_result = EdgeCutValidationResult()
            self._selected_edge_cut_polygon_indices = set()
            self._selected_edge_cut_profile_index = None
            return
        try:
            refreshed_gerber = self.gerber_parser.parse_file(gerber.path)
        except Exception:
            refreshed_gerber = gerber
        else:
            self._replace_imported_gerber(refreshed_gerber)
            x_offset, y_offset = self._file_alignment_offset()
            gerber = self._translate_gerber(refreshed_gerber, x_offset, y_offset)
        try:
            self._edge_cut_validation_result = validate_edge_segments(gerber.segments)
        except ModuleNotFoundError as exc:
            if exc.name != "shapely":
                raise
            self._edge_cut_validation_result = EdgeCutValidationResult(
                issues=[
                    "Edge validation requires the 'Shapely' package. Install dependencies from requirements.txt and restart the application."
                ]
            )
        self._ensure_edge_cut_mode_defaults()

    def _replace_imported_gerber(self, updated: ImportedGerberFile) -> None:
        for index, gerber in enumerate(self.imported_gerbers):
            if gerber.path == updated.path:
                self.imported_gerbers[index] = updated
                return

    def _edge_cut_validation_is_valid(self) -> bool:
        if self.project.layer_assignments.get("edges") is None:
            return True
        return self._edge_cut_validation_result.is_valid

    def _sync_edge_cut_page(self) -> None:
        if not hasattr(self, "edge_cut_mode_combo"):
            return
        polygons = self._edge_cut_validation_result.polygons
        if not polygons:
            self._selected_edge_cut_polygon_indices = set()
            self._selected_edge_cut_profile_index = None
            self.edge_cut_selection_value.setText("No valid paths available")
            self.edge_cut_profile_list.blockSignals(True)
            self.edge_cut_profile_list.clear()
            self.edge_cut_profile_list.blockSignals(False)
            self._populate_edge_cut_tool_combo("")
            self.edge_cut_mode_combo.blockSignals(True)
            self.edge_cut_mode_combo.setCurrentIndex(0)
            self.edge_cut_mode_combo.blockSignals(False)
            self.edge_cut_depth_input.blockSignals(True)
            self.edge_cut_depth_input.setText(
                self._format_edge_cut_numeric(self._default_edge_cut_depth())
            )
            self.edge_cut_depth_input.blockSignals(False)
            self.edge_cut_step_down_input.blockSignals(True)
            self.edge_cut_step_down_input.setText(
                self._format_edge_cut_numeric(self._default_edge_cut_step_down())
            )
            self.edge_cut_step_down_input.blockSignals(False)
            self.edge_cut_tool_combo.setEnabled(False)
            self.edge_cut_depth_input.setEnabled(False)
            self.edge_cut_step_down_input.setEnabled(False)
            self.edge_cut_mode_combo.setEnabled(False)
            return
        self._selected_edge_cut_polygon_indices = {
            index
            for index in self._selected_edge_cut_polygon_indices
            if 0 <= index < len(polygons)
        }
        profile_display_rows = self._edge_cut_profile_display_rows()
        self.edge_cut_profile_list.blockSignals(True)
        current_profile_index = self._selected_edge_cut_profile_index
        self.edge_cut_profile_list.clear()
        for profile_index, label in profile_display_rows:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, profile_index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if self.project.edge_cut_profiles[profile_index].visible
                else Qt.CheckState.Unchecked
            )
            self.edge_cut_profile_list.addItem(item)
        if current_profile_index is not None:
            profile_found = False
            for row in range(self.edge_cut_profile_list.count()):
                item = self.edge_cut_profile_list.item(row)
                if (
                    item is not None
                    and item.data(Qt.ItemDataRole.UserRole) == current_profile_index
                ):
                    self.edge_cut_profile_list.setCurrentRow(row)
                    profile_found = True
                    break
            if not profile_found:
                self._selected_edge_cut_profile_index = None
        self.edge_cut_profile_list.blockSignals(False)
        selected_indices = sorted(self._selected_edge_cut_polygon_indices)
        if not selected_indices:
            self.edge_cut_selection_value.setText("No path selected")
            self._populate_edge_cut_tool_combo("")
            self.edge_cut_mode_combo.blockSignals(True)
            self.edge_cut_mode_combo.setCurrentIndex(0)
            self.edge_cut_mode_combo.blockSignals(False)
            self.edge_cut_depth_input.blockSignals(True)
            self.edge_cut_depth_input.setText(
                self._format_edge_cut_numeric(self._default_edge_cut_depth())
            )
            self.edge_cut_depth_input.blockSignals(False)
            self.edge_cut_step_down_input.blockSignals(True)
            self.edge_cut_step_down_input.setText(
                self._format_edge_cut_numeric(self._default_edge_cut_step_down())
            )
            self.edge_cut_step_down_input.blockSignals(False)
            self.edge_cut_tool_combo.setEnabled(bool(self._available_edge_cut_tools()))
            self.edge_cut_depth_input.setEnabled(False)
            self.edge_cut_step_down_input.setEnabled(False)
            self.edge_cut_mode_combo.setEnabled(False)
            return
        selected_path = self.project.edge_cut_profiles[selected_indices[0]]
        self.edge_cut_selection_value.setText(
            f"Path {selected_indices[0] + 1} of {len(polygons)} selected"
        )
        self._populate_edge_cut_tool_combo(selected_path.tool_id)
        self.edge_cut_mode_combo.blockSignals(True)
        selected_profile_mode = self._selected_edge_cut_profile_mode()
        self.edge_cut_mode_combo.setCurrentIndex(
            max(self.edge_cut_mode_combo.findData(selected_profile_mode), 0)
        )
        self.edge_cut_mode_combo.blockSignals(False)
        self.edge_cut_depth_input.blockSignals(True)
        self.edge_cut_depth_input.setText(
            self._format_edge_cut_numeric(selected_path.cut_depth)
        )
        self.edge_cut_depth_input.blockSignals(False)
        self.edge_cut_step_down_input.blockSignals(True)
        self.edge_cut_step_down_input.setText(
            self._format_edge_cut_numeric(selected_path.step_down)
        )
        self.edge_cut_step_down_input.blockSignals(False)
        self.edge_cut_tool_combo.setEnabled(bool(self._available_edge_cut_tools()))
        self.edge_cut_depth_input.setEnabled(True)
        self.edge_cut_step_down_input.setEnabled(True)
        self.edge_cut_mode_combo.setEnabled(True)

    def _select_edge_cut_polygon(self, index: int, ctrl_pressed: bool) -> None:
        if index < 0:
            self._selected_edge_cut_polygon_indices = set()
            self._selected_edge_cut_profile_index = None
            if hasattr(self, "edge_cut_profile_list"):
                self.edge_cut_profile_list.blockSignals(True)
                self.edge_cut_profile_list.setCurrentRow(-1)
                self.edge_cut_profile_list.blockSignals(False)
            self.statusBar().showMessage("No edge contours selected", 2000)
            self._sync_ui()
            return
        if not (0 <= index < len(self._edge_cut_validation_result.polygons)):
            return
        del ctrl_pressed
        self._selected_edge_cut_polygon_indices = {index}
        self._selected_edge_cut_profile_index = index
        if hasattr(self, "edge_cut_profile_list"):
            self.edge_cut_profile_list.blockSignals(True)
            self.edge_cut_profile_list.setCurrentRow(index)
            self.edge_cut_profile_list.blockSignals(False)
        self.statusBar().showMessage(f"Selected edge path {index + 1}", 2000)
        self._sync_ui()

    def _edge_cut_mode_changed(self, row: int) -> None:
        if self._selected_edge_cut_profile_index is None:
            return
        if not (
            0
            <= self._selected_edge_cut_profile_index
            < len(self.project.edge_cut_profiles)
        ):
            return
        mode = str(self.edge_cut_mode_combo.itemData(row) or "none")
        updated_profiles = list(self.project.edge_cut_profiles)
        selected_profile = updated_profiles[self._selected_edge_cut_profile_index]
        if selected_profile.mode == mode:
            return
        updated_profiles[self._selected_edge_cut_profile_index] = EdgeCutPath(
            polygon_keys=list(selected_profile.polygon_keys),
            mode=mode,
            tool_id=selected_profile.tool_id,
            cut_depth=selected_profile.cut_depth,
            step_down=selected_profile.step_down,
            generated=selected_profile.generated,
            visible=selected_profile.visible,
        )
        if self.project.replace_edge_cut_profiles(updated_profiles):
            self.toolpath_viewer.load_document(None)
            self._mark_project_dirty()
        self._sync_ui()

    def _edge_cut_tool_changed(self, row: int) -> None:
        if self._selected_edge_cut_profile_index is None:
            return
        if not (
            0
            <= self._selected_edge_cut_profile_index
            < len(self.project.edge_cut_profiles)
        ):
            return
        tool_id = str(self.edge_cut_tool_combo.itemData(row) or "")
        updated_profiles = list(self.project.edge_cut_profiles)
        selected_path = updated_profiles[self._selected_edge_cut_profile_index]
        if selected_path.tool_id == tool_id:
            return
        updated_profiles[self._selected_edge_cut_profile_index] = EdgeCutPath(
            polygon_keys=list(selected_path.polygon_keys),
            mode=selected_path.mode,
            tool_id=tool_id,
            cut_depth=selected_path.cut_depth,
            step_down=selected_path.step_down,
            generated=selected_path.generated,
            visible=selected_path.visible,
        )
        if self.project.replace_edge_cut_profiles(updated_profiles):
            self.toolpath_viewer.load_document(None)
            self._mark_project_dirty()
        self._sync_ui()

    def _edge_cut_depth_changed(self) -> None:
        value = self._parse_edge_cut_numeric_input(
            self.edge_cut_depth_input, fallback=self._default_edge_cut_depth()
        )
        if value is None:
            self._sync_ui()
            return
        self.edge_cut_depth_input.setText(self._format_edge_cut_numeric(value))
        self._update_selected_edge_cut_path(cut_depth=value)

    def _edge_cut_step_down_changed(self) -> None:
        value = self._parse_edge_cut_numeric_input(
            self.edge_cut_step_down_input, fallback=self._default_edge_cut_step_down()
        )
        if value is None:
            self._sync_ui()
            return
        self.edge_cut_step_down_input.setText(self._format_edge_cut_numeric(value))
        self._update_selected_edge_cut_path(step_down=value)

    def _update_selected_edge_cut_path(
        self, *, cut_depth: float | None = None, step_down: float | None = None
    ) -> None:
        if self._selected_edge_cut_profile_index is None:
            return
        if not (
            0
            <= self._selected_edge_cut_profile_index
            < len(self.project.edge_cut_profiles)
        ):
            return
        updated_profiles = list(self.project.edge_cut_profiles)
        selected_path = updated_profiles[self._selected_edge_cut_profile_index]
        next_cut_depth = (
            selected_path.cut_depth if cut_depth is None else float(cut_depth)
        )
        next_step_down = (
            selected_path.step_down if step_down is None else float(step_down)
        )
        if (
            abs(selected_path.cut_depth - next_cut_depth) < 1e-9
            and abs(selected_path.step_down - next_step_down) < 1e-9
        ):
            return
        updated_profiles[self._selected_edge_cut_profile_index] = EdgeCutPath(
            polygon_keys=list(selected_path.polygon_keys),
            mode=selected_path.mode,
            tool_id=selected_path.tool_id,
            cut_depth=next_cut_depth,
            step_down=next_step_down,
            generated=selected_path.generated,
            visible=selected_path.visible,
        )
        if self.project.replace_edge_cut_profiles(updated_profiles):
            self.toolpath_viewer.load_document(None)
            self._mark_project_dirty()
        self._sync_ui()

    def _edge_cut_mode_label(self, mode: str) -> str:
        if mode == "none":
            return "None"
        if mode == "inside_profile":
            return "Inside profile"
        if mode == "on_contour":
            return "On contour"
        return "Outside profile"

    def _ensure_edge_cut_mode_defaults(self) -> None:
        polygons = self._edge_cut_validation_result.polygons
        if not polygons:
            self._selected_edge_cut_polygon_indices = set()
            self._selected_edge_cut_profile_index = None
            return
        self._selected_edge_cut_polygon_indices = {
            index
            for index in self._selected_edge_cut_polygon_indices
            if 0 <= index < len(polygons)
        }
        polygon_config_map: dict[str, EdgeCutPath] = {}
        for profile in self.project.edge_cut_profiles:
            for polygon_key in profile.polygon_keys:
                polygon_config_map[polygon_key] = profile
        self.project.edge_cut_profiles = [
            EdgeCutPath(
                polygon_keys=[self._polygon_key(polygon)],
                mode=polygon_config_map.get(
                    self._polygon_key(polygon), EdgeCutPath()
                ).mode,
                tool_id=polygon_config_map.get(
                    self._polygon_key(polygon), EdgeCutPath(tool_id="")
                ).tool_id
                or "",
                cut_depth=polygon_config_map.get(
                    self._polygon_key(polygon),
                    EdgeCutPath(cut_depth=self._default_edge_cut_depth()),
                ).cut_depth,
                step_down=polygon_config_map.get(
                    self._polygon_key(polygon),
                    EdgeCutPath(step_down=self._default_edge_cut_step_down()),
                ).step_down,
                generated=polygon_config_map.get(
                    self._polygon_key(polygon), EdgeCutPath()
                ).generated,
                visible=polygon_config_map.get(
                    self._polygon_key(polygon), EdgeCutPath(visible=True)
                ).visible,
            )
            for polygon in polygons
        ]
        if self._selected_edge_cut_profile_index is not None and not (
            0
            <= self._selected_edge_cut_profile_index
            < len(self.project.edge_cut_profiles)
        ):
            self._selected_edge_cut_profile_index = None

    def _edge_cut_profile_selected(self, row: int) -> None:
        if row < 0:
            self._selected_edge_cut_profile_index = None
            self._selected_edge_cut_polygon_indices = set()
            self._sync_ui()
            return
        item = self.edge_cut_profile_list.item(row)
        if item is None:
            self._selected_edge_cut_profile_index = None
            return
        profile_index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(profile_index, int):
            self._selected_edge_cut_profile_index = None
            return
        if not (0 <= profile_index < len(self.project.edge_cut_profiles)):
            self._selected_edge_cut_profile_index = None
            return
        self._selected_edge_cut_profile_index = profile_index
        profile = self.project.edge_cut_profiles[profile_index]
        self._selected_edge_cut_polygon_indices = self._profile_polygon_indices(profile)
        self.statusBar().showMessage(f"Highlighted edge path {profile_index + 1}", 2000)
        self._sync_ui()

    def _delete_selected_edge_cut_path(self) -> None:
        selected_indices = self._selected_edge_cut_path_indices()
        if not selected_indices:
            QMessageBox.information(self, "Edge cuts", "Select a path first.")
            return
        updated_paths = list(self.project.edge_cut_profiles)
        for path_index in selected_indices:
            current_path = updated_paths[path_index]
            updated_paths[path_index] = EdgeCutPath(
                polygon_keys=list(current_path.polygon_keys),
                mode="none",
                tool_id="",
                cut_depth=self._default_edge_cut_depth(),
                step_down=self._default_edge_cut_step_down(),
                generated=False,
                visible=True,
            )
        if self.project.replace_edge_cut_profiles(updated_paths):
            self.toolpath_viewer.load_document(None)
            self._mark_project_dirty()
        self._generated_edge_cut_preview_paths = (
            self._visible_generated_edge_cut_preview_paths()
        )
        self._sync_ui()

    def _delete_all_edge_cut_paths(self) -> None:
        if not self.project.edge_cut_profiles:
            return
        updated_paths = [
            EdgeCutPath(
                polygon_keys=list(path.polygon_keys),
                mode="none",
                tool_id="",
                cut_depth=self._default_edge_cut_depth(),
                step_down=self._default_edge_cut_step_down(),
                generated=False,
                visible=True,
            )
            for path in self.project.edge_cut_profiles
        ]
        if self.project.replace_edge_cut_profiles(updated_paths):
            self.toolpath_viewer.load_document(None)
            self._mark_project_dirty()
        self._generated_edge_cut_preview_paths = []
        self._sync_ui()

    def _edge_cut_profile_item_changed(self, item: QListWidgetItem) -> None:
        profile_index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(profile_index, int):
            return
        if not (0 <= profile_index < len(self.project.edge_cut_profiles)):
            return
        current_path = self.project.edge_cut_profiles[profile_index]
        visible = item.checkState() == Qt.CheckState.Checked
        if current_path.visible == visible:
            return
        updated_paths = list(self.project.edge_cut_profiles)
        updated_paths[profile_index] = EdgeCutPath(
            polygon_keys=list(current_path.polygon_keys),
            mode=current_path.mode,
            tool_id=current_path.tool_id,
            cut_depth=current_path.cut_depth,
            step_down=current_path.step_down,
            generated=current_path.generated,
            visible=visible,
        )
        if self.project.replace_edge_cut_profiles(updated_paths):
            self._mark_project_dirty()
        self._generated_edge_cut_preview_paths = (
            self._visible_generated_edge_cut_preview_paths()
        )
        self._sync_ui()

    def _resolved_edge_cut_paths(
        self, *, path_indices: set[int] | None = None
    ) -> list[dict[str, object]]:
        polygon_map = {
            self._polygon_key(polygon): polygon
            for polygon in self._edge_cut_validation_result.polygons
        }
        resolved: list[dict[str, object]] = []
        for profile_index, profile in enumerate(self.project.edge_cut_profiles):
            if path_indices is not None and profile_index not in path_indices:
                continue
            if profile.mode == "none":
                continue
            tool = self._edge_cut_tool_by_id(profile.tool_id)
            if tool is None:
                raise ValueError(
                    f"Path {profile_index + 1} does not have a valid milling tool."
                )
            for polygon_key in profile.polygon_keys:
                polygon = polygon_map.get(polygon_key)
                if polygon is None:
                    continue
                toolpath = self._cam_generator().edge_cut_paths(
                    [polygon],
                    cut_modes=[profile.mode],
                    mill_diameter=tool.numeric_parameter("diameter", 0.1),
                )[0]
                resolved.append(
                    {
                        "path_index": profile_index,
                        "outline": polygon,
                        "cut_mode": profile.mode,
                        "mill_diameter": tool.numeric_parameter("diameter", 0.1),
                        "cut_depth": max(0.001, profile.cut_depth),
                        "step_down": max(0.001, profile.step_down),
                        "tool_label": tool.label,
                        "tool_id": tool.identifier,
                        "toolpath": toolpath,
                    }
                )
        return resolved

    def _selected_edge_cut_path_indices(self) -> set[int]:
        if self._selected_edge_cut_profile_index is None:
            return set()
        if not (
            0
            <= self._selected_edge_cut_profile_index
            < len(self.project.edge_cut_profiles)
        ):
            return set()
        return {self._selected_edge_cut_profile_index}

    def _sync_edge_cut_action_buttons(self) -> None:
        if not hasattr(self, "edge_cut_generate_selected_button"):
            return
        has_selected_path = bool(self._selected_edge_cut_path_indices())
        self.edge_cut_delete_selected_button.setEnabled(has_selected_path)
        self.edge_cut_generate_selected_button.setEnabled(
            self._selected_edge_cut_path_can_generate()
        )

    def _selected_edge_cut_path_can_generate(self) -> bool:
        selected_indices = self._selected_edge_cut_path_indices()
        if len(selected_indices) != 1:
            return False
        if self._assigned_gerber("edges") is None:
            return False
        if not self._edge_cut_validation_result.is_valid:
            return False
        profile = self.project.edge_cut_profiles[next(iter(selected_indices))]
        if profile.mode == "none":
            return False
        if self._edge_cut_tool_by_id(profile.tool_id) is None:
            return False
        polygon_keys = {
            self._polygon_key(polygon)
            for polygon in self._edge_cut_validation_result.polygons
        }
        return any(key in polygon_keys for key in profile.polygon_keys)

    def _visible_generated_edge_cut_preview_paths(
        self,
    ) -> list[list[tuple[float, float]]]:
        visible_paths: list[list[tuple[float, float]]] = []
        polygon_map = {
            self._polygon_key(polygon): polygon
            for polygon in self._edge_cut_validation_result.polygons
        }
        for profile in self.project.edge_cut_profiles:
            if not profile.generated or not profile.visible or profile.mode == "none":
                continue
            tool = self._edge_cut_tool_by_id(profile.tool_id)
            if tool is None:
                continue
            for polygon_key in profile.polygon_keys:
                polygon = polygon_map.get(polygon_key)
                if polygon is None:
                    continue
                try:
                    toolpath = self._cam_generator().edge_cut_paths(
                        [polygon],
                        cut_modes=[profile.mode],
                        mill_diameter=tool.numeric_parameter("diameter", 0.1),
                    )[0]
                except Exception:
                    continue
                visible_paths.append(toolpath)
        return visible_paths

    def _profile_polygon_indices(self, profile: EdgeCutPath) -> set[int]:
        polygon_indices: set[int] = set()
        polygon_key_to_index = {
            self._polygon_key(polygon): index
            for index, polygon in enumerate(self._edge_cut_validation_result.polygons)
        }
        for polygon_key in profile.polygon_keys:
            index = polygon_key_to_index.get(polygon_key)
            if index is not None:
                polygon_indices.add(index)
        return polygon_indices

    def _edge_cut_profile_display_rows(self) -> list[tuple[int, str]]:
        rows: list[tuple[int, str]] = []
        for index, profile in enumerate(self.project.edge_cut_profiles):
            tool = self._edge_cut_tool_by_id(profile.tool_id)
            tool_label = tool.label if tool is not None else "No tool"
            status = "generated" if profile.generated else "not generated"
            if profile.generated and not profile.visible:
                status = "hidden"
            rows.append(
                (
                    index,
                    f"Path {index + 1}: {self._edge_cut_mode_label(profile.mode)} | {tool_label} | depth {profile.cut_depth:.3f} mm | step {profile.step_down:.3f} mm | {status}",
                )
            )
        return rows

    def _selected_edge_cut_profile_mode(self) -> str:
        if self._selected_edge_cut_profile_index is not None and (
            0
            <= self._selected_edge_cut_profile_index
            < len(self.project.edge_cut_profiles)
        ):
            return self.project.edge_cut_profiles[
                self._selected_edge_cut_profile_index
            ].mode
        return str(self.edge_cut_mode_combo.currentData() or "none")

    def _edge_cut_polygon_labels(self) -> dict[int, str]:
        return {
            index: profile.mode
            for index, profile in enumerate(self.project.edge_cut_profiles)
        }

    def _available_edge_cut_tools(self):
        if self.tool_library is None:
            return []
        return list(self.tool_library.tools_by_category["milling"])

    def _populate_edge_cut_tool_combo(self, selected_tool_id: str) -> None:
        self.edge_cut_tool_combo.blockSignals(True)
        self.edge_cut_tool_combo.clear()
        tools = self._available_edge_cut_tools()
        if not tools:
            self.edge_cut_tool_combo.addItem("No milling tools available", "")
            self.edge_cut_tool_combo.setEnabled(False)
        else:
            self.edge_cut_tool_combo.addItem("Select a tool...", "")
            for tool in tools:
                self.edge_cut_tool_combo.addItem(tool.label, tool.identifier)
            index = self.edge_cut_tool_combo.findData(selected_tool_id)
            self.edge_cut_tool_combo.setCurrentIndex(0 if index < 0 else index)
            self.edge_cut_tool_combo.setEnabled(True)
        self.edge_cut_tool_combo.blockSignals(False)

    def _edge_cut_tool_by_id(self, tool_id: str):
        for tool in self._available_edge_cut_tools():
            if tool.identifier == tool_id:
                return tool
        return None

    def _default_edge_cut_depth(self) -> float:
        return 1.65

    def _default_edge_cut_step_down(self) -> float:
        return 0.2

    def _format_edge_cut_numeric(self, value: float) -> str:
        return f"{float(value):.3f}"

    def _parse_edge_cut_numeric_input(
        self, field: QLineEdit, *, fallback: float
    ) -> float | None:
        text = field.text().strip()
        if not text:
            return fallback
        try:
            value = float(text)
        except ValueError:
            return None
        return max(0.0, min(1_000_000.0, value))

    def _polygon_key(self, polygon: list[tuple[float, float]]) -> str:
        return "|".join(f"{x_pos:.6f},{y_pos:.6f}" for x_pos, y_pos in polygon)

    def _edge_validation_message(self) -> str:
        assigned_path = self.project.layer_assignments.get("edges")
        if assigned_path is None:
            return ""
        if self._edge_cut_validation_result.message:
            return (
                f"Edge cut layer '{assigned_path.name}' is invalid. "
                f"{self._edge_cut_validation_result.message}"
            )
        return f"Edge cut layer '{assigned_path.name}' is invalid."

    def _sync_mirror_setup_page(self) -> None:
        requires_mirror = self.project.requires_mirror_setup()
        self.mirror_requirement_label.setText(
            "Front and back copper are both assigned. Choose a mirror edge or None for no mirroring, then select Overlay or Side by side preview."
            if requires_mirror
            else "Only one copper side is assigned. Mirror setup is not required for this project."
        )
        for edge, button in self.mirror_buttons.items():
            button.blockSignals(True)
            button.setEnabled(requires_mirror)
            button.setChecked(
                requires_mirror
                and (
                    self.project.mirror_flip_edge == edge
                    or (not self.project.mirror_flip_edge and edge == "")
                )
            )
            button.blockSignals(False)
        self.mirror_preview_mode_combo.blockSignals(True)
        index = self.mirror_preview_mode_combo.findData(
            self.project.mirror_preview_mode
        )
        self.mirror_preview_mode_combo.setCurrentIndex(0 if index < 0 else index)
        self.mirror_preview_mode_combo.setEnabled(requires_mirror)
        self.mirror_preview_mode_combo.blockSignals(False)
        can_mirror_side = bool(
            self.project.mirror_flip_edge and self._preview_mirror_bounds() is not None
        )
        self.preview_side_front_radio.blockSignals(True)
        self.preview_side_back_radio.blockSignals(True)
        self.preview_side_front_radio.setChecked(
            self.project.mirror_view_side != "back"
        )
        self.preview_side_back_radio.setChecked(self.project.mirror_view_side == "back")
        self.preview_side_front_radio.setEnabled(can_mirror_side)
        self.preview_side_back_radio.setEnabled(can_mirror_side)
        self.preview_side_front_radio.blockSignals(False)
        self.preview_side_back_radio.blockSignals(False)
        self.mirror_preview.set_edge(
            self.project.mirror_flip_edge if requires_mirror else ""
        )

    def _sync_alignment_holes_page(self) -> None:
        self.file_alignment_horizontal_offset_input.blockSignals(True)
        self.file_alignment_horizontal_offset_input.setText(
            f"{self.project.file_alignment_horizontal_offset:.3f}"
        )
        self.file_alignment_horizontal_offset_input.blockSignals(False)
        self.file_alignment_vertical_offset_input.blockSignals(True)
        self.file_alignment_vertical_offset_input.setText(
            f"{self.project.file_alignment_vertical_offset:.3f}"
        )
        self.file_alignment_vertical_offset_input.blockSignals(False)
        self.alignment_grid_size_input.blockSignals(True)
        self.alignment_grid_size_input.setText(
            f"{self.project.alignment_grid_size:.3f}"
        )
        self.alignment_grid_size_input.blockSignals(False)
        selected_point = self._file_alignment_point()
        alignment_label = self.FILE_ALIGNMENT_LABELS.get(
            self.project.file_alignment,
            self.project.file_alignment.replace("_", " ").title(),
        )
        self.file_alignment_value.setText(
            "Imported files alignment: "
            f"{alignment_label} at {format_origin_point(selected_point)}"
            if selected_point is not None
            else (
                "Imported files alignment: set the stock dimensions, then click "
                "a stock hotspot"
            )
        )
        selected_rows = {
            index.row() for index in self.alignment_hole_list.selectedIndexes()
        }
        current_row = self.alignment_hole_list.currentRow()
        self.alignment_hole_list.blockSignals(True)
        self.alignment_hole_list.clear()
        for row, hole in enumerate(self.project.alignment_holes):
            base_position, mirrored_position = self._alignment_hole_pair_positions(hole)
            if base_position is None or mirrored_position is None:
                position_label = "position unavailable"
            else:
                position_label = (
                    f"at ({base_position[0]:.3f}, {base_position[1]:.3f}) mm -> "
                    f"({mirrored_position[0]:.3f}, {mirrored_position[1]:.3f}) mm"
                )
            mirror_label = (
                "horizontal" if hole.mirror_direction == "horizontal" else "vertical"
            )
            item = QListWidgetItem(
                f"{position_label} | {mirror_label} mirror | dia {hole.diameter:.3f} mm"
            )
            item.setData(Qt.ItemDataRole.UserRole, row)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if hole.enabled else Qt.CheckState.Unchecked
            )
            self.alignment_hole_list.addItem(item)
            if row in selected_rows:
                item.setSelected(True)
        if current_row in selected_rows:
            self.alignment_hole_list.setCurrentRow(current_row)
        elif selected_rows:
            self.alignment_hole_list.setCurrentRow(selected_rows[0])
        else:
            self.alignment_hole_list.setCurrentRow(-1)
        self.alignment_hole_list.blockSignals(False)
        self._alignment_hole_selection_changed()

    def _sync_stock_definition_page(self) -> None:
        for widget, value in (
            (self.stock_width_input, self.project.stock_width),
            (self.stock_height_input, self.project.stock_height),
            (self.stock_thickness_input, self.project.stock_thickness),
        ):
            widget.blockSignals(True)
            widget.setText("" if value == 0.0 else f"{value:.3f}")
            widget.blockSignals(False)
        selected_origin = self._current_origin_point()
        origin_label = NC_ORIGIN_LABELS.get(
            self.project.stock_origin,
            self.project.stock_origin.replace("_", " ").title(),
        )
        self.stock_origin_value.setText(
            f"Current origin: {origin_label} at {format_origin_point(selected_origin)}"
            if selected_origin is not None
            else "Current origin: set the stock dimensions, then click a stock hotspot"
        )
        self.stock_hint.setText(
            "Hover over the stock rectangle in preview and click a corner, edge midpoint, or the center to set XY work zero."
            if self._stock_bounds() is not None
            else "Enter stock width, height, and thickness to activate the stock preview and origin hotspots."
        )

    def _alignment_hole_positions(self) -> list[tuple[float, float, float]]:
        self._alignment_preview_row_map = []
        reference_bounds = self._stock_bounds()
        if reference_bounds is None:
            return []
        positions: list[tuple[float, float, float]] = []
        for row, hole in enumerate(self.project.alignment_holes):
            if not hole.enabled:
                continue
            base_position = self._alignment_hole_position_for_bounds(
                hole, reference_bounds
            )
            mirrored_position = self._mirrored_alignment_hole_position(
                base_position, hole.mirror_direction, reference_bounds
            )
            positions.append(base_position)
            self._alignment_preview_row_map.append(row)
            if mirrored_position[:2] != base_position[:2]:
                positions.append(mirrored_position)
                self._alignment_preview_row_map.append(row)
        return positions

    def _alignment_hole_pair_positions(
        self, hole: AlignmentHole
    ) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
        reference_bounds = self._stock_bounds()
        if reference_bounds is None:
            return None, None
        base_position = self._alignment_hole_position_for_bounds(hole, reference_bounds)
        mirrored_position = self._mirrored_alignment_hole_position(
            base_position, hole.mirror_direction, reference_bounds
        )
        return base_position, mirrored_position

    def _visible_alignment_hole_positions(self) -> list[tuple[float, float, float]]:
        reference_bounds = self._stock_bounds()
        if reference_bounds is None:
            return []
        positions: list[tuple[float, float, float]] = []
        for hole in self.project.alignment_holes:
            if not hole.enabled:
                continue
            base_position = self._alignment_hole_position_for_bounds(
                hole, reference_bounds
            )
            mirrored_position = self._mirrored_alignment_hole_position(
                base_position, hole.mirror_direction, reference_bounds
            )
            positions.append(base_position)
            if mirrored_position[:2] != base_position[:2]:
                positions.append(mirrored_position)
        return positions

    def _defined_alignment_hole_positions(self) -> list[tuple[float, float, float]]:
        reference_bounds = self._stock_bounds()
        if reference_bounds is None:
            return []
        return [
            self._alignment_hole_position_for_bounds(hole, reference_bounds)
            for hole in self.project.alignment_holes
            if hole.enabled
        ]

    def _reference_board_bounds(self) -> tuple[float, float, float, float] | None:
        edge_file = self._assigned_gerber("edges")
        if edge_file is not None and not edge_file.bounds.is_empty:
            return (
                edge_file.bounds.x_min,
                edge_file.bounds.x_max,
                edge_file.bounds.y_min,
                edge_file.bounds.y_max,
            )

        for gerber in self._active_gerbers():
            if not gerber.outline or gerber.bounds.is_empty:
                continue
            return (
                gerber.bounds.x_min,
                gerber.bounds.x_max,
                gerber.bounds.y_min,
                gerber.bounds.y_max,
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

    def _reference_board_bounds_required(self) -> tuple[float, float, float, float]:
        bounds = self._reference_board_bounds()
        if bounds is None:
            raise ValueError("Board bounds are not available.")
        return bounds

    def _stock_definition_is_valid(self) -> bool:
        return (
            self.project.stock_width > 0.0
            and self.project.stock_height > 0.0
            and self.project.stock_thickness > 0.0
            and self._current_origin_point() is not None
        )

    def _stock_bounds(self) -> tuple[float, float, float, float] | None:
        if self.project.stock_width <= 0.0 or self.project.stock_height <= 0.0:
            return None
        return (0.0, self.project.stock_width, 0.0, self.project.stock_height)

    def _current_origin_point(self) -> tuple[float, float] | None:
        bounds = self._stock_bounds()
        if bounds is None:
            return None
        return legacy_origin_point_for_bounds(bounds, self.project.stock_origin)

    def _current_origin_point_required(self) -> tuple[float, float]:
        point = self._current_origin_point()
        if point is None:
            raise ValueError("NC origin point is not available.")
        return point

    def _toolpath_stock_bounds(self) -> tuple[float, float, float, float] | None:
        bounds = self._stock_bounds()
        origin_point = self._current_origin_point()
        if bounds is None or origin_point is None:
            return None
        return self._bounds_relative_to_origin(bounds, origin_point)

    def _toolpath_board_bounds(self) -> tuple[float, float, float, float] | None:
        bounds = self._reference_board_bounds()
        origin_point = self._current_origin_point()
        if bounds is None or origin_point is None:
            return None
        return self._bounds_relative_to_origin(bounds, origin_point)

    def _toolpath_preview_bounds(self) -> tuple[float, float, float, float] | None:
        return self._toolpath_board_bounds() or self._toolpath_stock_bounds()

    def _toolpath_alignment_holes(self) -> list[tuple[float, float, float]]:
        origin_point = self._current_origin_point()
        if origin_point is None:
            return []
        origin_x, origin_y = origin_point
        holes = [
            (x_pos - origin_x, y_pos - origin_y, diameter)
            for x_pos, y_pos, diameter in self._visible_alignment_hole_positions()
        ]
        if self.project.mirror_view_side != "back":
            return holes
        bounds = self._preview_mirror_bounds()
        if not self.project.mirror_flip_edge or bounds is None:
            return holes
        mirrored_holes = []
        for x_pos, y_pos, diameter in holes:
            point = self._mirror_toolpath_point(Point3D(x_pos, y_pos, 0.0), bounds)
            mirrored_holes.append((point.x, point.y, diameter))
        return mirrored_holes

    def _toolpath_defined_alignment_holes(self) -> list[tuple[float, float, float]]:
        origin_point = self._current_origin_point()
        if origin_point is None:
            return []
        origin_x, origin_y = origin_point
        holes = [
            (x_pos - origin_x, y_pos - origin_y, diameter)
            for x_pos, y_pos, diameter in self._defined_alignment_hole_positions()
        ]
        if self.project.mirror_view_side != "back":
            return holes
        bounds = self._preview_mirror_bounds()
        if not self.project.mirror_flip_edge or bounds is None:
            return holes
        mirrored_holes = []
        for x_pos, y_pos, diameter in holes:
            point = self._mirror_toolpath_point(Point3D(x_pos, y_pos, 0.0), bounds)
            mirrored_holes.append((point.x, point.y, diameter))
        return mirrored_holes

    def _bounds_relative_to_origin(
        self,
        bounds: tuple[float, float, float, float],
        origin_point: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        origin_x, origin_y = origin_point
        x_min, x_max, y_min, y_max = bounds
        return (x_min - origin_x, x_max - origin_x, y_min - origin_y, y_max - origin_y)

    def _file_alignment_point(self) -> tuple[float, float] | None:
        bounds = self._stock_bounds()
        if bounds is None:
            return None
        return self._alignment_target_point_for_bounds(
            bounds, self.project.file_alignment
        )

    def _file_alignment_offset(self) -> tuple[float, float]:
        source_bounds = self._selected_import_bounds()
        target_bounds = self._stock_bounds()
        if source_bounds is None or target_bounds is None:
            return 0.0, 0.0
        source_point = legacy_origin_point_for_bounds(
            source_bounds, self.project.file_alignment
        )
        target_point = self._alignment_target_point_for_bounds(
            target_bounds, self.project.file_alignment
        )
        return target_point[0] - source_point[0], target_point[1] - source_point[1]

    def _alignment_target_point_for_bounds(
        self, bounds: tuple[float, float, float, float], alignment: str
    ) -> tuple[float, float]:
        x_min, x_max, y_min, y_max = bounds
        x_mid = (x_min + x_max) * 0.5
        y_mid = (y_min + y_max) * 0.5
        horizontal_offset = min(
            self.project.file_alignment_horizontal_offset,
            max(0.0, (x_max - x_min) * 0.5),
        )
        vertical_offset = min(
            self.project.file_alignment_vertical_offset, max(0.0, (y_max - y_min) * 0.5)
        )
        if alignment.endswith("_left"):
            x_pos = x_min + horizontal_offset
        elif alignment.endswith("_right"):
            x_pos = x_max - horizontal_offset
        else:
            x_pos = x_mid

        if alignment.startswith("top_"):
            y_pos = y_max - vertical_offset
        elif alignment.startswith("bottom_"):
            y_pos = y_min + vertical_offset
        else:
            y_pos = y_mid
        return x_pos, y_pos

    def _selected_import_bounds(self) -> tuple[float, float, float, float] | None:
        bounds = None
        for gerber in self._raw_active_gerbers():
            if gerber.bounds.is_empty:
                continue
            bounds = self._extend_bounds(bounds, gerber.bounds)
        for drill in self._raw_active_drills():
            if drill.bounds.is_empty:
                continue
            bounds = self._extend_bounds(bounds, drill.bounds)
        if bounds is None:
            return None
        return bounds[0], bounds[1], bounds[2], bounds[3]

    def _extend_bounds(self, bounds: list[float] | None, next_bounds) -> list[float]:
        if bounds is None:
            return [
                next_bounds.x_min,
                next_bounds.x_max,
                next_bounds.y_min,
                next_bounds.y_max,
            ]
        bounds[0] = min(bounds[0], next_bounds.x_min)
        bounds[1] = max(bounds[1], next_bounds.x_max)
        bounds[2] = min(bounds[2], next_bounds.y_min)
        bounds[3] = max(bounds[3], next_bounds.y_max)
        return bounds

    def _origin_hotspot_points(self) -> dict[str, tuple[float, float]]:
        bounds = self._stock_bounds()
        if bounds is None:
            return {}
        x_min, x_max, y_min, y_max = bounds
        x_mid = (x_min + x_max) * 0.5
        y_mid = (y_min + y_max) * 0.5
        return {
            "top_left": (x_min, y_max),
            "top_center": (x_mid, y_max),
            "top_right": (x_max, y_max),
            "center_left": (x_min, y_mid),
            "center": (x_mid, y_mid),
            "center_right": (x_max, y_mid),
            "bottom_left": (x_min, y_min),
            "bottom_center": (x_mid, y_min),
            "bottom_right": (x_max, y_min),
        }

    def _file_alignment_hotspot_points(self) -> dict[str, tuple[float, float]]:
        bounds = self._stock_bounds()
        if bounds is None:
            return {}
        return {
            key: self._alignment_target_point_for_bounds(bounds, key)
            for key in self._origin_hotspot_points()
        }

    def _assigned_gerber(self, role: str) -> ImportedGerberFile | None:
        assigned_path = self.project.layer_assignments.get(role)
        if assigned_path is None:
            return None
        gerber = self._imported_gerber_by_path(assigned_path)
        if gerber is None:
            return None
        x_offset, y_offset = self._file_alignment_offset()
        return self._translate_gerber(gerber, x_offset, y_offset)

    def _imported_gerber_by_path(self, path: Path) -> ImportedGerberFile | None:
        resolved_path = path.resolve()
        for gerber in self.imported_gerbers:
            if gerber.path == resolved_path:
                return gerber
        return None

    def _alignment_hole_position_for_bounds(
        self, hole: AlignmentHole, bounds: tuple[float, float, float, float]
    ) -> tuple[float, float, float]:
        x_min, x_max, y_min, y_max = bounds
        return (
            min(max(hole.x_offset, x_min), x_max),
            min(max(hole.y_offset, y_min), y_max),
            hole.diameter,
        )

    def _mirrored_alignment_hole_position(
        self,
        hole_position: tuple[float, float, float],
        mirror_direction: str,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float, float]:
        x_min, x_max, y_min, y_max = bounds
        x_pos, y_pos, diameter = hole_position
        if mirror_direction == "vertical":
            return x_pos, y_min + y_max - y_pos, diameter
        return x_min + x_max - x_pos, y_pos, diameter

    def _operation_tool(self, operation_key: str, role: str):
        if self.tool_library is None:
            return None
        selected_id = self.project.operation_tools.get(operation_key, "")
        for tool in self.tool_library.tools_by_category[role]:
            if tool.identifier == selected_id:
                return tool
        return None

    def _set_origin_location(self, x_pos: float, y_pos: float) -> None:
        next_point = (x_pos, y_pos)
        hotspot_points = (
            self._file_alignment_hotspot_points()
            if self.project.current_step_index == PcbProject.STEP_ALIGNMENT_HOLES
            else self._origin_hotspot_points()
        )
        selected_origin = next(
            (key for key, point in hotspot_points.items() if point == next_point), None
        )
        if self.project.current_step_index == PcbProject.STEP_ALIGNMENT_HOLES:
            if (
                selected_origin is None
                or self.project.file_alignment == selected_origin
            ):
                return
            self.project.set_file_alignment(selected_origin)
            self.toolpath_viewer.load_document(None)
            self._mark_project_dirty()
            self.statusBar().showMessage(
                "Imported files aligned to "
                f"{self.FILE_ALIGNMENT_LABELS.get(selected_origin, selected_origin)}",
                3000,
            )
            self._sync_ui()
            return
        if selected_origin is None or self.project.stock_origin == selected_origin:
            return
        self.project.set_stock_origin(selected_origin)
        self.toolpath_viewer.load_document(None)
        self._mark_project_dirty()
        self.statusBar().showMessage(
            f"Stock origin set to {NC_ORIGIN_LABELS.get(selected_origin, selected_origin)}",
            3000,
        )
        self._sync_ui()

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
        self._hidden_generated_output_keys.discard(operation_key)
        self._loaded_generated_output_keys = ()
        self._loaded_generated_output_paths = ()
        self._mark_project_dirty()
        self._load_generated_document(operation_key, path)
        self._sync_ui()

    def _load_generated_document(self, operation_key: str, path: Path) -> None:
        document = self._generated_document(path, force=True)
        self.toolpath_viewer.load_document(
            self._preview_toolpath_document(operation_key, document)
        )

    def _generated_document(
        self, path: Path, *, force: bool = False
    ) -> ToolpathDocument:
        resolved_path = path.resolve()
        cache_key = str(resolved_path)
        document = self.generated_documents.get(cache_key)
        if force or document is None:
            document = self.gcode_parser.parse_file(resolved_path)
            self.generated_documents[cache_key] = document
        return document

    def _sync_generated_outputs(self) -> None:
        generated_map = {
            "front_isolation": getattr(self, "front_isolation_value", None),
            "back_isolation": getattr(self, "back_isolation_value", None),
            "alignment_drill": getattr(self, "alignment_drill_value", None),
            "alignment_mill": getattr(self, "alignment_mill_value", None),
            "drilling": getattr(self, "drilling_value", None),
            "edge_cuts": getattr(self, "edge_cuts_value", None),
        }
        for key, label in generated_map.items():
            if label is None:
                continue
            path = self.project.generated_outputs.get(key)
            if key == "alignment_drill":
                label.setText(
                    f"Alignment drill path: {path}"
                    if path is not None
                    else "Alignment drill path: not generated yet"
                )
            elif key == "alignment_mill":
                label.setText(
                    f"Alignment mill path: {path}"
                    if path is not None
                    else "Alignment mill path: not generated yet"
                )
            else:
                label.setText(str(path) if path is not None else "Not generated yet")

        generated_keys = set(self.project.generated_outputs)
        self._hidden_generated_output_keys.intersection_update(generated_keys)

        self.generated_output_list.blockSignals(True)
        self.generated_output_list.clear()
        for key, title in (
            ("front_isolation", "Front Isolation"),
            ("back_isolation", "Back Isolation"),
            ("alignment_drill", "Alignment Drill"),
            ("alignment_mill", "Alignment Mill"),
            ("drilling", "Drilling"),
            ("edge_cuts", "Edge Cuts"),
        ):
            path = self.project.generated_outputs.get(key)
            if path is None:
                continue
            item = QListWidgetItem(f"{title}: {path.name}")
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked
                if key in self._hidden_generated_output_keys
                else Qt.CheckState.Checked
            )
            self.generated_output_list.addItem(item)
        self.generated_output_list.blockSignals(False)
        self._load_selected_generated_documents()

    def _load_selected_generated_documents(self) -> None:
        documents = []
        selected_keys = []
        selected_paths = []
        for key in self._visible_generated_output_keys_for_preview():
            if key in self._hidden_generated_output_keys:
                continue
            path = self.project.generated_outputs.get(key)
            if path is None:
                continue
            try:
                document = self._generated_document(path)
            except Exception:
                logger.exception("Failed to load generated NC output: %s", path)
                continue
            selected_keys.append(key)
            selected_paths.append(str(path.resolve()))
            documents.append(document)
        selected_key_state = (
            self.project.mirror_view_side,
            self.project.mirror_flip_edge,
            *selected_keys,
        )
        selected_path_state = tuple(selected_paths)
        if (
            selected_key_state == self._loaded_generated_output_keys
            and selected_path_state == self._loaded_generated_output_paths
        ):
            return
        self._loaded_generated_output_keys = selected_key_state
        self._loaded_generated_output_paths = selected_path_state
        if not documents:
            self.toolpath_viewer.load_document(None)
            return
        documents = [
            self._preview_toolpath_document(key, document)
            for key, document in zip(selected_keys, documents)
        ]
        if len(documents) == 1:
            self.toolpath_viewer.load_document(documents[0])
            return
        self.toolpath_viewer.load_document(self._combined_toolpath_document(documents))

    def _visible_generated_output_keys_for_preview(self) -> tuple[str, ...]:
        if self.project.mirror_view_side == "back":
            side_key = "back_isolation"
        else:
            side_key = "front_isolation"

        current = self.project.current_step_index
        if current in {PcbProject.STEP_FRONT_ISOLATION, PcbProject.STEP_BACK_ISOLATION}:
            return (side_key,)
        if current == PcbProject.STEP_ALIGNMENT_HOLES:
            return ("alignment_drill", "alignment_mill")
        if current == PcbProject.STEP_DRILLING:
            return ("drilling",)
        if current == PcbProject.STEP_NC_PREVIEW:
            return (
                side_key,
                "alignment_drill",
                "alignment_mill",
                "drilling",
                "edge_cuts",
            )
        return (
            "front_isolation",
            "back_isolation",
            "alignment_drill",
            "alignment_mill",
            "drilling",
            "edge_cuts",
        )

    def _preview_toolpath_document(
        self, operation_key: str, document: ToolpathDocument
    ) -> ToolpathDocument:
        if self.project.mirror_view_side != "back" or operation_key == "back_isolation":
            return document
        bounds = self._preview_mirror_bounds()
        if not self.project.mirror_flip_edge or bounds is None:
            return document
        segments = [
            Segment3D(
                start=self._mirror_toolpath_point(segment.start, bounds),
                end=self._mirror_toolpath_point(segment.end, bounds),
                rapid=segment.rapid,
                line_number=segment.line_number,
                source=segment.source,
            )
            for segment in document.segments
        ]
        return self._toolpath_document_from_segments(document.path, segments)

    def _mirror_toolpath_point(
        self, point: Point3D, bounds: tuple[float, float, float, float]
    ) -> Point3D:
        x_min, x_max, y_min, y_max = bounds
        if self.project.mirror_flip_edge == "left":
            return Point3D((2.0 * x_min) - point.x, point.y, point.z)
        if self.project.mirror_flip_edge == "right":
            return Point3D((2.0 * x_max) - point.x, point.y, point.z)
        if self.project.mirror_flip_edge == "top":
            return Point3D(point.x, (2.0 * y_max) - point.y, point.z)
        if self.project.mirror_flip_edge == "bottom":
            return Point3D(point.x, (2.0 * y_min) - point.y, point.z)
        return point

    def _preview_mirror_bounds(self) -> tuple[float, float, float, float] | None:
        return self._toolpath_board_bounds() or self._toolpath_stock_bounds()

    def _toolpath_document_from_segments(
        self, path: Path, segments: list[Segment3D]
    ) -> ToolpathDocument:
        if not segments:
            return ToolpathDocument(
                path=path,
                segments=[],
                stats=ToolpathStats(
                    min_point=Point3D(0.0, 0.0, 0.0),
                    max_point=Point3D(0.0, 0.0, 0.0),
                    segment_count=0,
                    rapid_count=0,
                    cut_count=0,
                    path_length=0.0,
                ),
            )
        min_point = Point3D(
            min(min(segment.start.x, segment.end.x) for segment in segments),
            min(min(segment.start.y, segment.end.y) for segment in segments),
            min(min(segment.start.z, segment.end.z) for segment in segments),
        )
        max_point = Point3D(
            max(max(segment.start.x, segment.end.x) for segment in segments),
            max(max(segment.start.y, segment.end.y) for segment in segments),
            max(max(segment.start.z, segment.end.z) for segment in segments),
        )
        path_length = sum(
            math.dist(
                (segment.start.x, segment.start.y, segment.start.z),
                (segment.end.x, segment.end.y, segment.end.z),
            )
            for segment in segments
        )
        return ToolpathDocument(
            path=path,
            segments=segments,
            stats=ToolpathStats(
                min_point=min_point,
                max_point=max_point,
                segment_count=len(segments),
                rapid_count=sum(1 for segment in segments if segment.rapid),
                cut_count=sum(1 for segment in segments if not segment.rapid),
                path_length=path_length,
            ),
        )

    def _combined_toolpath_document(
        self, documents: list[ToolpathDocument]
    ) -> ToolpathDocument:
        segments = [segment for document in documents for segment in document.segments]
        if not segments:
            return self._toolpath_document_from_segments(Path("Selected NC paths"), [])
        return self._toolpath_document_from_segments(
            Path("Selected NC paths"), segments
        )

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

    def _operation_optional_or_generated(
        self, operation_key: str, layer_role: str
    ) -> bool:
        if self.project.layer_assignments.get(layer_role) is None:
            return True
        return operation_key in self.project.generated_outputs

    def _drilling_optional_or_generated(self) -> bool:
        if not self._active_drills():
            return True
        return "drilling" in self.project.generated_outputs

    def _raw_active_gerbers(self) -> list[ImportedGerberFile]:
        return [
            gerber
            for gerber in self.imported_gerbers
            if self.project.is_gerber_selected(gerber.path)
        ]

    def _raw_active_drills(self) -> list[ImportedDrillFile]:
        return [
            drill
            for drill in self.imported_drills
            if self.project.is_drill_selected(drill.path)
        ]

    def _active_gerbers(self) -> list[ImportedGerberFile]:
        return self._aligned_gerbers(self._raw_active_gerbers())

    def _active_drills(self) -> list[ImportedDrillFile]:
        return self._aligned_drills(self._raw_active_drills())

    def _aligned_gerbers(
        self, gerbers: list[ImportedGerberFile]
    ) -> list[ImportedGerberFile]:
        x_offset, y_offset = self._file_alignment_offset()
        return [
            self._translate_gerber(gerber, x_offset, y_offset) for gerber in gerbers
        ]

    def _aligned_drills(
        self, drills: list[ImportedDrillFile]
    ) -> list[ImportedDrillFile]:
        x_offset, y_offset = self._file_alignment_offset()
        return [self._translate_drill(drill, x_offset, y_offset) for drill in drills]

    def _translate_gerber(
        self, gerber: ImportedGerberFile, x_offset: float, y_offset: float
    ) -> ImportedGerberFile:
        if abs(x_offset) < 1e-9 and abs(y_offset) < 1e-9:
            return gerber

        def point(point: tuple[float, float]) -> tuple[float, float]:
            return point[0] + x_offset, point[1] + y_offset

        def polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
            return [point(item) for item in points]

        bounds = copy.copy(gerber.bounds)
        if not bounds.is_empty:
            bounds.x_min += x_offset
            bounds.x_max += x_offset
            bounds.y_min += y_offset
            bounds.y_max += y_offset
        return ImportedGerberFile(
            path=gerber.path,
            display_name=gerber.display_name,
            traces=[
                (point(start), point(end), width) for start, end, width in gerber.traces
            ],
            segments=[(point(start), point(end)) for start, end in gerber.segments],
            arc_centers=[point(center) for center in gerber.arc_centers],
            pads=[
                (point(center), copy.copy(definition))
                for center, definition in gerber.pads
            ],
            regions=[polygon(region) for region in gerber.regions],
            outline=polygon(gerber.outline),
            bounds=bounds,
        )

    def _translate_drill(
        self, drill: ImportedDrillFile, x_offset: float, y_offset: float
    ) -> ImportedDrillFile:
        if abs(x_offset) < 1e-9 and abs(y_offset) < 1e-9:
            return drill
        bounds = copy.copy(drill.bounds)
        if not bounds.is_empty:
            bounds.x_min += x_offset
            bounds.x_max += x_offset
            bounds.y_min += y_offset
            bounds.y_max += y_offset
        return ImportedDrillFile(
            path=drill.path,
            display_name=drill.display_name,
            holes=[
                (x_pos + x_offset, y_pos + y_offset, diameter)
                for x_pos, y_pos, diameter in drill.holes
            ],
            bounds=bounds,
        )

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
        self._update_window_title()

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
