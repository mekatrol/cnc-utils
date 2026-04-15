from __future__ import annotations

from pathlib import Path

import yaml

from .alignment_hole import AlignmentHole


class PcbProject:
    VERSION = 1
    TOTAL_STEPS = 12
    STEP_KEYS = [
        "project",
        "gerber_import",
        "drill_import",
        "tool_selection",
        "layer_assignment",
        "mirror_setup",
        "alignment_holes",
        "front_isolation",
        "back_isolation",
        "drilling",
        "edge_cuts",
        "nc_preview",
    ]

    def __init__(self) -> None:
        self.project_path: Path | None = None
        self.gerber_paths: list[Path] = []
        self.drill_paths: list[Path] = []
        self.selected_gerber_paths: set[Path] = set()
        self.selected_drill_paths: set[Path] = set()
        self.tool_library_path: Path | None = None
        self.selected_tools: dict[str, str] = {
            "drilling": "",
            "milling": "",
            "v_bits": "",
        }
        self.layer_assignments: dict[str, Path | None] = {
            "front_copper": None,
            "back_copper": None,
            "edges": None,
        }
        self.mirror_flip_edge: str = ""
        self.alignment_holes: list[AlignmentHole] = []
        self.generated_outputs: dict[str, Path] = {}
        self.current_step_index = 0
        self.highest_commenced_step = 0
        self.completed_steps: set[int] = {0}
        self.dirty_from_step: int | None = None

    def reset(self) -> None:
        self.__init__()

    def replace_gerber_paths(self, paths: list[Path]) -> bool:
        normalized = [path.resolve() for path in paths]
        changed = normalized != self.gerber_paths
        self.gerber_paths = normalized
        current_selection = {
            path for path in self.selected_gerber_paths if path in self.gerber_paths
        }
        for path in self.gerber_paths:
            if path not in current_selection:
                current_selection.add(path)
        self.selected_gerber_paths = current_selection
        if changed:
            self._prune_missing_layer_assignments()
            self._invalidate_from(1)
        return changed

    def replace_drill_paths(self, paths: list[Path]) -> bool:
        normalized = [path.resolve() for path in paths]
        changed = normalized != self.drill_paths
        self.drill_paths = normalized
        current_selection = {
            path for path in self.selected_drill_paths if path in self.drill_paths
        }
        for path in self.drill_paths:
            if path not in current_selection:
                current_selection.add(path)
        self.selected_drill_paths = current_selection
        if changed:
            self._invalidate_from(2)
        return changed

    def set_gerber_selected(self, path: Path, selected: bool) -> bool:
        resolved = path.resolve()
        if resolved not in self.gerber_paths:
            return False
        changed = resolved in self.selected_gerber_paths
        if selected:
            self.selected_gerber_paths.add(resolved)
            changed = not changed
        else:
            self.selected_gerber_paths.discard(resolved)
            changed = changed
        if changed:
            self._prune_missing_layer_assignments()
            self._invalidate_from(1)
        return changed

    def set_drill_selected(self, path: Path, selected: bool) -> bool:
        resolved = path.resolve()
        if resolved not in self.drill_paths:
            return False
        changed = resolved in self.selected_drill_paths
        if selected:
            self.selected_drill_paths.add(resolved)
            changed = not changed
        else:
            self.selected_drill_paths.discard(resolved)
            changed = changed
        if changed:
            self._invalidate_from(2)
        return changed

    def is_gerber_selected(self, path: Path) -> bool:
        return path.resolve() in self.selected_gerber_paths

    def is_drill_selected(self, path: Path) -> bool:
        return path.resolve() in self.selected_drill_paths

    def set_tool_library_path(self, path: Path | None) -> bool:
        resolved = None if path is None else path.resolve()
        changed = resolved != self.tool_library_path
        self.tool_library_path = resolved
        if changed:
            self._invalidate_from(3)
        return changed

    def set_selected_tool(self, role: str, tool_id: str) -> bool:
        normalized = tool_id.strip()
        changed = self.selected_tools.get(role, "") != normalized
        self.selected_tools[role] = normalized
        if changed:
            self._invalidate_from(3)
        return changed

    def set_layer_assignment(self, role: str, path: Path | None) -> bool:
        resolved = None if path is None else path.resolve()
        changed = self.layer_assignments.get(role) != resolved
        self.layer_assignments[role] = resolved
        if changed:
            self._invalidate_from(4)
        if not self.requires_mirror_setup():
            self.mirror_flip_edge = ""
        return changed

    def set_mirror_flip_edge(self, edge: str) -> bool:
        normalized = edge.strip()
        changed = self.mirror_flip_edge != normalized
        self.mirror_flip_edge = normalized
        if changed:
            self._invalidate_from(5)
        return changed

    def replace_alignment_holes(self, holes: list[AlignmentHole]) -> bool:
        changed = holes != self.alignment_holes
        self.alignment_holes = holes
        if changed:
            self._invalidate_from(6)
        return changed

    def requires_mirror_setup(self) -> bool:
        return (
            self.layer_assignments.get("front_copper") is not None
            and self.layer_assignments.get("back_copper") is not None
        )

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
        return index <= self.highest_commenced_step

    def clear_dirty_state_through(self, index: int) -> None:
        if self.dirty_from_step is not None and index > self.dirty_from_step:
            self.dirty_from_step = None

    def last_completed_step_index(self, implemented_step_count: int | None = None) -> int:
        upper_bound = self.TOTAL_STEPS if implemented_step_count is None else min(
            implemented_step_count,
            self.TOTAL_STEPS,
        )
        valid_completed_steps = [
            step for step in self.completed_steps if 0 <= step < upper_bound
        ]
        if not valid_completed_steps:
            return 0
        return max(valid_completed_steps)

    def save_to_path(self, path: Path) -> None:
        self.project_path = path.resolve()
        self.completed_steps.add(0)
        self.project_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "gerber_files": [
                self._to_relative_path(item, self.project_path.parent)
                for item in self.gerber_paths
            ],
            "selected_gerber_files": [
                self._to_relative_path(item, self.project_path.parent)
                for item in self.gerber_paths
                if item in self.selected_gerber_paths
            ],
            "drill_files": [
                self._to_relative_path(item, self.project_path.parent)
                for item in self.drill_paths
            ],
            "selected_drill_files": [
                self._to_relative_path(item, self.project_path.parent)
                for item in self.drill_paths
                if item in self.selected_drill_paths
            ],
            "tool_library": {
                "path": (
                    None
                    if self.tool_library_path is None
                    else self._to_relative_path(
                        self.tool_library_path,
                        self.project_path.parent,
                    )
                ),
                "selected_tools": dict(self.selected_tools),
            },
            "layers": {
                key: (
                    None
                    if value is None
                    else self._to_relative_path(value, self.project_path.parent)
                )
                for key, value in self.layer_assignments.items()
            },
            "mirror": {
                "flip_edge": self.mirror_flip_edge or None,
            },
            "alignment_holes": [
                {
                    "edge": hole.edge,
                    "offset_along_edge": hole.offset_along_edge,
                    "offset_from_edge": hole.offset_from_edge,
                    "diameter": hole.diameter,
                }
                for hole in self.alignment_holes
            ],
            "generated_outputs": {
                key: self._to_relative_path(value, self.project_path.parent)
                for key, value in self.generated_outputs.items()
            },
            "wizard": {
                "current_step": self._step_key_for_index(self.current_step_index),
                "highest_commenced_step": self._step_key_for_index(
                    self.highest_commenced_step
                ),
                "completed_steps": [
                    self._step_key_for_index(step)
                    for step in sorted(self.completed_steps)
                ],
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
        project.selected_gerber_paths = set(project.gerber_paths)
        raw_selected_gerbers = loaded.get("selected_gerber_files")
        if isinstance(raw_selected_gerbers, list):
            project.selected_gerber_paths = {
                project._from_relative_path(item, project.project_path.parent)
                for item in raw_selected_gerbers
                if isinstance(item, str) and item.strip()
            }
        project.drill_paths = [
            project._from_relative_path(item, project.project_path.parent)
            for item in loaded.get("drill_files", [])
        ]
        project.selected_drill_paths = set(project.drill_paths)
        raw_selected_drills = loaded.get("selected_drill_files")
        if isinstance(raw_selected_drills, list):
            project.selected_drill_paths = {
                project._from_relative_path(item, project.project_path.parent)
                for item in raw_selected_drills
                if isinstance(item, str) and item.strip()
            }
        tool_library_data = loaded.get("tool_library", {})
        if not isinstance(tool_library_data, dict):
            tool_library_data = {}
        raw_tool_library_path = tool_library_data.get("path")
        if isinstance(raw_tool_library_path, str) and raw_tool_library_path.strip():
            project.tool_library_path = project._from_relative_path(
                raw_tool_library_path,
                project.project_path.parent,
            )
        selected_tools = tool_library_data.get("selected_tools", {})
        if isinstance(selected_tools, dict):
            for role in project.selected_tools:
                value = selected_tools.get(role, "")
                if isinstance(value, str):
                    project.selected_tools[role] = value.strip()
        layer_data = loaded.get("layers", {})
        if isinstance(layer_data, dict):
            for role in project.layer_assignments:
                raw_path = layer_data.get(role)
                if isinstance(raw_path, str) and raw_path.strip():
                    project.layer_assignments[role] = project._from_relative_path(
                        raw_path,
                        project.project_path.parent,
                    )
        mirror_data = loaded.get("mirror", {})
        if isinstance(mirror_data, dict):
            raw_edge = mirror_data.get("flip_edge", "")
            if isinstance(raw_edge, str):
                project.mirror_flip_edge = raw_edge.strip()
        alignment_data = loaded.get("alignment_holes", [])
        if isinstance(alignment_data, list):
            for item in alignment_data:
                if not isinstance(item, dict):
                    continue
                try:
                    project.alignment_holes.append(
                        AlignmentHole(
                            edge=str(item.get("edge", "")).strip(),
                            offset_along_edge=float(item.get("offset_along_edge", 0.0)),
                            offset_from_edge=float(item.get("offset_from_edge", 0.0)),
                            diameter=float(item.get("diameter", 0.0)),
                        )
                    )
                except (TypeError, ValueError):
                    continue
        generated_output_data = loaded.get("generated_outputs", {})
        if isinstance(generated_output_data, dict):
            for key, raw_path in generated_output_data.items():
                if isinstance(key, str) and isinstance(raw_path, str) and raw_path.strip():
                    project.generated_outputs[key] = project._from_relative_path(
                        raw_path,
                        project.project_path.parent,
                    )
        wizard_data = loaded.get("wizard", {})
        project.current_step_index = project._step_index_for_key(
            wizard_data.get("current_step"),
            0,
        )
        project.highest_commenced_step = project._step_index_for_key(
            wizard_data.get("highest_commenced_step"),
            project.current_step_index,
        )
        raw_completed_steps = wizard_data.get("completed_steps", [])
        if isinstance(raw_completed_steps, list):
            project.completed_steps = {
                project._step_index_for_key(step)
                for step in raw_completed_steps
                if isinstance(step, str)
            }
        else:
            project.completed_steps = set()
        project.completed_steps.add(0)
        project.selected_gerber_paths = {
            path for path in project.selected_gerber_paths if path in project.gerber_paths
        }
        project.selected_drill_paths = {
            path for path in project.selected_drill_paths if path in project.drill_paths
        }
        project._prune_missing_layer_assignments()
        if not project.requires_mirror_setup():
            project.mirror_flip_edge = ""
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
        if index <= 10:
            self.generated_outputs = {}
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

    def _prune_missing_layer_assignments(self) -> None:
        valid_paths = set(self.selected_gerber_paths)
        for role, path in list(self.layer_assignments.items()):
            if path is not None and path not in valid_paths:
                self.layer_assignments[role] = None

    def _step_key_for_index(self, index: int) -> str:
        normalized_index = max(0, min(index, self.TOTAL_STEPS - 1))
        return self.STEP_KEYS[normalized_index]

    def _step_index_for_key(self, key: object, default: int = 0) -> int:
        if not isinstance(key, str):
            return default
        try:
            return self.STEP_KEYS.index(key.strip())
        except ValueError:
            return default
