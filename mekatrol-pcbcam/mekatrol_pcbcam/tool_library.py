from __future__ import annotations

from pathlib import Path

import yaml

from .tool_definition import ToolDefinition


class ToolLibrary:
    CATEGORY_ALIASES = {
        "drilling": "drilling",
        "drill": "drilling",
        "drills": "drilling",
        "twist_drill": "drilling",
        "twist-drill": "drilling",
        "milling": "milling",
        "mill": "milling",
        "mills": "milling",
        "end_mill": "milling",
        "end-mill": "milling",
        "square_end_mill": "milling",
        "square-end-mill": "milling",
        "v_bits": "v_bits",
        "vbits": "v_bits",
        "vbit": "v_bits",
        "v-bits": "v_bits",
        "v-bit": "v_bits",
        "chamfer_mill": "v_bits",
        "chamfer-mill": "v_bits",
        "engraver": "v_bits",
        "engraving": "v_bits",
    }

    def __init__(self, path: Path, tools_by_category: dict[str, list[ToolDefinition]]) -> None:
        self.path = path
        self.tools_by_category = tools_by_category

    @classmethod
    def load(cls, path: str | Path) -> ToolLibrary:
        file_path = Path(path).resolve()
        loaded = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("tools.yaml must contain a top-level mapping.")

        tools_by_category = {
            "drilling": [],
            "milling": [],
            "v_bits": [],
        }

        categorized = False
        for raw_key, raw_value in loaded.items():
            if not isinstance(raw_key, str):
                continue
            category = cls.CATEGORY_ALIASES.get(raw_key.strip().lower())
            if category is None:
                continue
            categorized = True
            tools_by_category[category].extend(
                cls._coerce_entries(raw_value, fallback_category=category)
            )

        if not categorized:
            tools_by_category = {
                "drilling": [],
                "milling": [],
                "v_bits": [],
            }
            for entry in cls._coerce_entries(loaded, fallback_category=""):
                category = cls._infer_category(entry.category)
                if category is None:
                    continue
                tools_by_category[category].append(
                    ToolDefinition(entry.identifier, entry.label, category)
                )

        return cls(file_path, tools_by_category)

    @staticmethod
    def _coerce_entries(value: object, *, fallback_category: str) -> list[ToolDefinition]:
        entries: list[ToolDefinition] = []
        if isinstance(value, list):
            for index, item in enumerate(value):
                tool = ToolLibrary._coerce_tool(item, fallback_category, f"tool_{index + 1}")
                if tool is not None:
                    entries.append(tool)
        elif isinstance(value, dict):
            for key, item in value.items():
                tool = ToolLibrary._coerce_tool(item, fallback_category, str(key))
                if tool is not None:
                    entries.append(tool)
        return entries

    @staticmethod
    def _coerce_tool(
        item: object,
        fallback_category: str,
        fallback_identifier: str,
    ) -> ToolDefinition | None:
        if not isinstance(item, dict):
            return None
        raw_id = item.get("id", fallback_identifier)
        identifier = str(raw_id).strip()
        if not identifier:
            identifier = fallback_identifier
        raw_name = item.get("name") or item.get("label") or identifier
        label = str(raw_name).strip() or identifier
        details: list[str] = []
        for key in ("diameter", "tip_diameter", "angle", "flutes"):
            if key in item:
                details.append(f"{key}={item[key]}")
        if details:
            label = f"{label} ({', '.join(details)})"
        raw_category = item.get("category") or item.get("type") or fallback_category
        category = str(raw_category).strip()
        parameters = dict(item)
        return ToolDefinition(identifier, label, category, parameters)

    @classmethod
    def _infer_category(cls, raw_category: str) -> str | None:
        return cls.CATEGORY_ALIASES.get(raw_category.strip().lower())
