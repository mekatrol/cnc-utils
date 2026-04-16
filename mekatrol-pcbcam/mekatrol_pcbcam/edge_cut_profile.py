from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EdgeCutProfile:
    polygon_keys: list[str] = field(default_factory=list)
    mode: str = "outside_profile"
