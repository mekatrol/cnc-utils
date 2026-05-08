from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .tool_definition import ToolDefinition

if TYPE_CHECKING:
    from .tool_library import ToolLibrary


TOOL_TYPES = ["drill", "endmill", "v-bit"]
SETTING_COLUMNS = [
    ("diameter", "Diameter"),
    ("tip_diameter", "Tip Diameter"),
    ("tip_angle", "Tip Angle"),
    ("feed_rate", "Feed Rate"),
    ("preferred_speed", "Rotation Speed"),
]


def _table_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
    return item


def _display_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


class ToolSettingsDialog(QDialog):
    COLUMNS = [
        "Name",
        "Type",
        "Diameter",
        "Tip Diameter",
        "Tip Angle",
        "Feed Rate",
        "Rotation Speed",
    ]

    def __init__(
        self, path: Path, tool_library: ToolLibrary | None, parent=None
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._saved = False

        self.setWindowTitle("Tool Settings")
        self.setModal(True)
        self.resize(880, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        intro = QLabel(f"Edit tool library:\n{path}")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(1, 110)
        layout.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        add_button = QPushButton("Add Tool")
        add_button.clicked.connect(self._add_tool)
        modify_button = QPushButton("Modify Selected")
        modify_button.clicked.connect(self._modify_selected)
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self._delete_selected)
        button_row.addWidget(add_button)
        button_row.addWidget(modify_button)
        button_row.addWidget(delete_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        if save_button is not None:
            save_button.clicked.connect(self._save)
        save_close_button = buttons.addButton(
            "Save and Close", QDialogButtonBox.ButtonRole.AcceptRole
        )
        save_close_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        save_close_button.clicked.connect(self._save_and_close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if tool_library is not None:
            self._load_tools(tool_library)

    def saved(self) -> bool:
        return self._saved

    def _load_tools(self, tool_library: ToolLibrary) -> None:
        for category in ("drilling", "milling", "v_bits"):
            for tool in tool_library.tools_by_category[category]:
                self._append_tool(tool)

    def _append_tool(self, tool: ToolDefinition | None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        parameters = dict(tool.parameters) if tool is not None else {}
        name = str(parameters.get("name") or parameters.get("label") or "")
        tool_type = self._tool_type(parameters)

        id_item = _table_item(tool.identifier if tool is not None else self._new_id())
        id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setVerticalHeaderItem(row, id_item)

        self.table.setItem(row, 0, _table_item(name))
        self.table.setCellWidget(row, 1, self._tool_type_combo(tool_type))
        values = []
        for key, _label in SETTING_COLUMNS:
            values.append(_display_value(parameters.get(key)))

        for column, value in enumerate(values, start=2):
            self.table.setItem(row, column, _table_item(value))

    def _add_tool(self) -> None:
        self._append_tool(None)
        row = self.table.rowCount() - 1
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def _modify_selected(self) -> None:
        row = self.table.currentRow()
        column = self.table.currentColumn()
        if row < 0:
            return
        if column < 0:
            column = 0
            self.table.setCurrentCell(row, column)
        widget = self.table.cellWidget(row, column)
        if widget is not None:
            widget.setFocus()
            return
        item = self.table.item(row, column)
        if item is not None:
            self.table.editItem(item)

    def _delete_selected(self) -> None:
        rows = sorted({item.row() for item in self.table.selectedItems()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _save(self) -> None:
        self._save_file(close_after_save=False)

    def _save_and_close(self) -> None:
        self._save_file(close_after_save=True)

    def _save_file(self, *, close_after_save: bool) -> None:
        try:
            data = self._collect_data()
        except ValueError as exc:
            QMessageBox.warning(self, "Tool settings", str(exc))
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
        self._path.write_text(content, encoding="utf-8")
        self._saved = True
        if close_after_save:
            self.accept()

    def _collect_data(self) -> dict[str, list[dict[str, object]]]:
        data: dict[str, list[dict[str, object]]] = {"tools": []}
        seen_ids: set[str] = set()
        for row in range(self.table.rowCount()):
            identifier = self._row_id(row)
            if not identifier:
                identifier = self._new_id()
            if identifier in seen_ids:
                raise ValueError(f"Row {row + 1}: duplicate ID {identifier}.")
            seen_ids.add(identifier)

            name = self._cell_text(row, 0)
            tool_type = self._selected_tool_type(row)
            entry: dict[str, object] = {
                "id": identifier,
                "name": name or identifier,
                "type": tool_type,
            }
            for index, (key, _label) in enumerate(SETTING_COLUMNS, start=2):
                value = self._cell_text(row, index)
                if value:
                    entry[key] = self._yaml_scalar(value, row)
            data["tools"].append(entry)
        return data

    def _cell_text(self, row: int, column: int) -> str:
        item = self.table.item(row, column)
        if item is None:
            return ""
        return item.text().strip()

    def _yaml_scalar(self, value: str, row: int) -> object:
        try:
            loaded = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ValueError(f"Row {row + 1}: invalid value {value}.") from exc
        if isinstance(loaded, (dict, list)):
            raise ValueError(f"Row {row + 1}: value must be a scalar.")
        return loaded

    def _tool_type_combo(self, selected_tool_type: str) -> QComboBox:
        combo = QComboBox()
        for tool_type in TOOL_TYPES:
            combo.addItem(tool_type, tool_type)
        index = combo.findData(selected_tool_type)
        combo.setCurrentIndex(max(0, index))
        return combo

    def _selected_tool_type(self, row: int) -> str:
        widget = self.table.cellWidget(row, 1)
        if isinstance(widget, QComboBox):
            return str(widget.currentData() or "drill")
        return "drill"

    def _tool_type(self, parameters: dict[str, object]) -> str:
        tool_type = str(parameters.get("type") or "drill").strip().lower()
        if tool_type in TOOL_TYPES:
            return tool_type
        return "drill"

    def _row_id(self, row: int) -> str:
        item = self.table.verticalHeaderItem(row)
        if item is None:
            return ""
        return item.text().strip()

    def _new_id(self) -> str:
        return str(uuid4())
