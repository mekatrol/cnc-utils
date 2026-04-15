from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from .theme import load_theme


@dataclass(frozen=True)
class ThemeOption:
    file_name: str
    display_name: str
    description: str
    author: str


def discover_theme_options(themes_directory: Path) -> list[ThemeOption]:
    options: list[ThemeOption] = []
    if not themes_directory.exists():
        return options

    for path in sorted(themes_directory.glob("*.yaml"), key=lambda item: item.name.lower()):
        theme, _warnings = load_theme(path)
        display_name = theme.theme_info.name.strip() or path.stem
        options.append(
            ThemeOption(
                file_name=path.name,
                display_name=display_name,
                description=theme.theme_info.description.strip(),
                author=theme.theme_info.author.strip(),
            )
        )

    return sorted(
        options,
        key=lambda option: (option.display_name.lower(), option.file_name.lower()),
    )


class ThemeSettingsDialog(QDialog):
    def __init__(
        self,
        themes_directory: Path,
        options: list[ThemeOption],
        current_theme_file: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._themes_directory = themes_directory
        self._options = options

        self.setWindowTitle("Theme Settings")
        self.setModal(True)
        self.resize(500, 220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        intro = QLabel("Choose the application theme.")
        layout.addWidget(intro)

        self.theme_combo = QComboBox()
        for option in options:
            self.theme_combo.addItem(option.display_name, option.file_name)
        self.theme_combo.currentIndexChanged.connect(self._sync_details)
        layout.addWidget(self.theme_combo)

        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setTextFormat(Qt.TextFormat.RichText)
        self.details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        layout.addWidget(self.details_label)

        folder_link = QLabel(
            f'Themes folder: <a href="themes-folder">{themes_directory}</a>'
        )
        folder_link.setTextFormat(Qt.TextFormat.RichText)
        folder_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        folder_link.setOpenExternalLinks(False)
        folder_link.linkActivated.connect(self._open_themes_folder)
        layout.addWidget(folder_link)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        current_index = self.theme_combo.findData(current_theme_file)
        if current_index >= 0:
            self.theme_combo.setCurrentIndex(current_index)
        self._sync_details()

    def selected_theme_file(self) -> str:
        return str(self.theme_combo.currentData() or "")

    def _sync_details(self) -> None:
        index = self.theme_combo.currentIndex()
        if index < 0 or index >= len(self._options):
            self.details_label.setText("No theme selected.")
            return

        option = self._options[index]
        description = option.description or "No description provided."
        author = option.author or "Unknown"
        self.details_label.setText(
            "<b>Description:</b> "
            f"{description}<br><b>Author:</b> {author}<br><b>File:</b> {option.file_name}"
        )

    def _open_themes_folder(self, _link: str) -> None:
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._themes_directory))):
            return
        QMessageBox.warning(
            self,
            "Open themes folder",
            f"Could not open {self._themes_directory}.",
        )
