from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolDefinition:
    identifier: str
    label: str
    category: str
    parameters: dict[str, object]

    def numeric_parameter(self, name: str, default: float = 0.0) -> float:
        value = self.parameters.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
