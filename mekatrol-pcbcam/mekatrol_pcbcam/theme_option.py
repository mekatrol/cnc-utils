from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeOption:
    file_name: str
    display_name: str
    description: str
    author: str
