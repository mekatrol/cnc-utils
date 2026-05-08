from __future__ import annotations

from pathlib import Path

import yaml

from .tool_definition import ToolDefinition


class ToolLibrary:
    TOOL_TYPE_CATEGORIES = {
        "drill": "drilling",
        "endmill": "milling",
        "v-bit": "v_bits",
    }

    def __init__(
        self, path: Path, tools_by_category: dict[str, list[ToolDefinition]]
    ) -> None:
        self.path = path
        self.tools_by_category = tools_by_category

    @classmethod
    def load(cls, path: str | Path) -> ToolLibrary:
        file_path = Path(path).resolve()
        loaded = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("tools.yaml must contain a top-level mapping.")

        raw_tools = loaded.get("tools")
        if not isinstance(raw_tools, list):
            raise ValueError("tools.yaml must contain a tools list.")

        tools_by_category = {"drilling": [], "milling": [], "v_bits": []}
        for index, item in enumerate(raw_tools):
            tool = cls._coerce_tool(item, index)
            category = cls.TOOL_TYPE_CATEGORIES[tool.category]
            tools_by_category[category].append(tool)

        return cls(file_path, tools_by_category)

    @classmethod
    def _coerce_tool(cls, item: object, index: int) -> ToolDefinition:
        if not isinstance(item, dict):
            raise ValueError(f"Tool {index + 1} must be a mapping.")
        raw_id = item.get("id", "")
        identifier = str(raw_id).strip()
        if not identifier:
            raise ValueError(f"Tool {index + 1} is missing id.")
        raw_name = item.get("name") or identifier
        label = str(raw_name).strip() or identifier
        raw_type = item.get("type", "")
        tool_type = str(raw_type).strip().lower()
        if tool_type not in cls.TOOL_TYPE_CATEGORIES:
            allowed = ", ".join(cls.TOOL_TYPE_CATEGORIES)
            raise ValueError(f"Tool {index + 1} type must be one of: {allowed}.")
        details: list[str] = []
        for key in (
            "diameter",
            "tip_diameter",
            "tip_angle",
            "feed_rate",
            "preferred_speed",
        ):
            if key in item:
                details.append(f"{key}={item[key]}")
        if details:
            label = f"{label} ({', '.join(details)})"
        parameters = dict(item)
        return ToolDefinition(identifier, label, tool_type, parameters)
