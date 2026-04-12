from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .gcode_parser import GCodeParser
from .toolpath_document import ToolpathDocument
from .viewer import ToolpathViewer


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.parser = GCodeParser()
        self.document: ToolpathDocument | None = None
        self.viewer = ToolpathViewer()

        self.setWindowTitle("mekatrol-pcbcam")
        self.resize(1280, 840)
        self._build_ui()
        self._build_menu()
        self._update_summary(None)
        logger.debug("Main window initialized")

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 960])
        self.setCentralWidget(splitter)

        status = QStatusBar(self)
        status.showMessage("Ready")
        self.setStatusBar(status)

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Mekatrol PCBCAM")
        title.setStyleSheet("font-size: 26px; font-weight: 700;")
        subtitle = QLabel("Load generated G-code and inspect its motion path in 3D.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #5b6571;")

        open_button = QPushButton("Open .nc File")
        open_button.clicked.connect(self.open_file)
        fit_button = QPushButton("Fit View")
        fit_button.clicked.connect(self.viewer.fit_to_view)

        button_row = QHBoxLayout()
        button_row.addWidget(open_button)
        button_row.addWidget(fit_button)

        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        form = QFormLayout(card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setSpacing(10)
        self.file_value = QLabel("No file loaded")
        self.file_value.setWordWrap(True)
        self.segment_value = QLabel("-")
        self.bounds_value = QLabel("-")
        self.length_value = QLabel("-")
        form.addRow("File", self.file_value)
        form.addRow("Segments", self.segment_value)
        form.addRow("Bounds", self.bounds_value)
        form.addRow("Path length", self.length_value)

        help_label = QLabel(
            "Controls\n"
            "Left drag: orbit\n"
            "Right drag: pan\n"
            "Wheel: zoom\n"
            "F: fit"
        )
        help_label.setStyleSheet("color: #5b6571;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(button_row)
        layout.addWidget(card)
        layout.addWidget(help_label)
        layout.addStretch(1)
        return panel

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = self.menuBar().addMenu("&View")
        fit_action = QAction("Fit to View", self)
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self.viewer.fit_to_view)
        view_menu.addAction(fit_action)

    def open_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open NC/G-code File",
            str(Path.home()),
            "NC Files (*.nc *.gcode *.tap *.ngc);;All Files (*)",
        )
        if not selected:
            logger.debug("Open-file dialog cancelled")
            return
        logger.info("Selected file for loading: %s", selected)
        self.load_file(selected)

    def load_file(self, file_path: str) -> None:
        logger.info("Loading toolpath file: %s", file_path)
        try:
            document = self.parser.parse_file(file_path)
        except Exception as exc:
            logger.exception("Failed to load toolpath file: %s", file_path)
            QMessageBox.critical(self, "Failed to load file", str(exc))
            self.statusBar().showMessage("Load failed", 3000)
            return

        self.document = document
        self.viewer.load_document(document)
        self._update_summary(document)
        self.statusBar().showMessage(f"Loaded {document.path.name}", 3000)
        logger.info(
            "Loaded toolpath file: %s segments=%d cut=%d rapid=%d",
            document.path,
            document.stats.segment_count,
            document.stats.cut_count,
            document.stats.rapid_count,
        )

    def _update_summary(self, document: ToolpathDocument | None) -> None:
        if document is None:
            self.file_value.setText("No file loaded")
            self.segment_value.setText("-")
            self.bounds_value.setText("-")
            self.length_value.setText("-")
            return

        stats = document.stats
        self.file_value.setText(str(document.path))
        self.segment_value.setText(
            f"{stats.segment_count} total  |  {stats.cut_count} cut  |  {stats.rapid_count} rapid"
        )
        self.bounds_value.setText(
            f"X {stats.min_point.x:.2f}..{stats.max_point.x:.2f}  "
            f"Y {stats.min_point.y:.2f}..{stats.max_point.y:.2f}  "
            f"Z {stats.min_point.z:.2f}..{stats.max_point.z:.2f}"
        )
        self.length_value.setText(f"{stats.path_length:.2f} mm")
