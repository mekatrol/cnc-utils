from __future__ import annotations

from pathlib import Path

import yaml


class PcbProject:
    VERSION = 1
    TOTAL_STEPS = 11

    def __init__(self) -> None:
        self.project_path: Path | None = None
        self.gerber_paths: list[Path] = []
        self.drill_paths: list[Path] = []
        self.current_step_index = 0
        self.highest_commenced_step = 0
        self.completed_steps: set[int] = set()
        self.dirty_from_step: int | None = None

    def reset(self) -> None:
        self.__init__()

    def replace_gerber_paths(self, paths: list[Path]) -> bool:
        normalized = [path.resolve() for path in paths]
        changed = normalized != self.gerber_paths
        self.gerber_paths = normalized
        if changed:
            self._invalidate_from(0)
        if self.gerber_paths:
            self.completed_steps.add(0)
        else:
            self.completed_steps.discard(0)
        return changed

    def replace_drill_paths(self, paths: list[Path]) -> bool:
        normalized = [path.resolve() for path in paths]
        changed = normalized != self.drill_paths
        self.drill_paths = normalized
        if changed:
            self._invalidate_from(1)
        self.completed_steps.add(1)
        return changed

    def set_current_step(self, index: int) -> None:
        self.current_step_index = max(0, min(index, self.TOTAL_STEPS - 1))
        self.highest_commenced_step = max(
            self.highest_commenced_step,
            self.current_step_index,
        )

    def can_navigate_to(self, index: int, implemented_step_count: int) -> bool:
        if index < 0 or index >= min(implemented_step_count, self.TOTAL_STEPS):
            return False
        if index <= self.current_step_index:
            return True
        if self.dirty_from_step is not None:
            return index <= self.dirty_from_step + 1
        return index <= self.highest_commenced_step + 1

    def clear_dirty_state_through(self, index: int) -> None:
        if self.dirty_from_step is not None and index > self.dirty_from_step:
            self.dirty_from_step = None

    def save_to_path(self, path: Path) -> None:
        self.project_path = path.resolve()
        self.project_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "gerber_files": [
                self._to_relative_path(item, self.project_path.parent)
                for item in self.gerber_paths
            ],
            "drill_files": [
                self._to_relative_path(item, self.project_path.parent)
                for item in self.drill_paths
            ],
            "wizard": {
                "current_step_index": self.current_step_index,
                "highest_commenced_step": self.highest_commenced_step,
                "completed_steps": sorted(self.completed_steps),
            },
        }
        self.project_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def load_from_path(cls, path: Path) -> PcbProject:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        project = cls()
        project.project_path = path.resolve()
        project.gerber_paths = [
            project._from_relative_path(item, project.project_path.parent)
            for item in loaded.get("gerber_files", [])
        ]
        project.drill_paths = [
            project._from_relative_path(item, project.project_path.parent)
            for item in loaded.get("drill_files", [])
        ]
        wizard_data = loaded.get("wizard", {})
        project.current_step_index = int(wizard_data.get("current_step_index", 0))
        project.highest_commenced_step = int(
            wizard_data.get("highest_commenced_step", project.current_step_index)
        )
        project.completed_steps = {
            int(step) for step in wizard_data.get("completed_steps", [])
        }
        if not project.gerber_paths:
            project.completed_steps.discard(0)
        project.completed_steps.add(1)
        project.dirty_from_step = None
        return project

    def _invalidate_from(self, index: int) -> None:
        self.completed_steps = {
            step for step in self.completed_steps if step <= index
        }
        if index < self.highest_commenced_step:
            self.dirty_from_step = (
                index
                if self.dirty_from_step is None
                else min(self.dirty_from_step, index)
            )
        self.highest_commenced_step = min(self.highest_commenced_step, index + 1)
        self.current_step_index = min(self.current_step_index, index + 1)

    def _to_relative_path(self, path: Path, base_dir: Path) -> str:
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            return str(path)

    def _from_relative_path(self, raw_path: str, base_dir: Path) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (base_dir / candidate).resolve()
