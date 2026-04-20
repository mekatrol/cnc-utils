from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EdgeCutPath:
    polygon_keys: list[str] = field(default_factory=list)
    mode: str = "none"
    tool_id: str = ""
    cut_depth: float = 1.8
    step_down: float = 0.4
    generated: bool = False
    visible: bool = True


EdgeCutProfile = EdgeCutPath
