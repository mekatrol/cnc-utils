from __future__ import annotations

from pathlib import Path

import yaml

from .alignment_hole import AlignmentHole
from .edge_cut_profile import EdgeCutPath
from .nc_origin import DEFAULT_NC_ORIGIN, normalize_nc_origin


class PcbProject:
    VERSION = 10
    STEP_PROJECT = 0
    STEP_STOCK_DEFINITION = 1
    STEP_GERBER_IMPORT = 2
    STEP_DRILL_IMPORT = 3
    STEP_ALIGNMENT_HOLES = 4
    STEP_TOOL_SELECTION = 5
    STEP_FRONT_ISOLATION = 6
    STEP_BACK_ISOLATION = 7
    STEP_DRILLING = 8
    STEP_EDGE_CUTS = 9
    STEP_NC_PREVIEW = 10
    TOTAL_STEPS = 11
    STEP_KEYS = [
        "project",
        "stock_definition",
        "gerber_import",
        "drill_import",
        "alignment_holes",
        "tool_selection",
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
        self.stock_width: float = 100.0
        self.stock_height: float = 60.0
        self.stock_thickness: float = 1.6
        self.stock_origin: str = DEFAULT_NC_ORIGIN
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
        self.mirror_preview_mode: str = "side_by_side"
        self.file_alignment: str = DEFAULT_NC_ORIGIN
        self.file_alignment_horizontal_offset: float = 0.0
        self.file_alignment_vertical_offset: float = 0.0
        self.alignment_grid_size: float = 5.0
        self.alignment_holes: list[AlignmentHole] = []
        self.edge_cut_profiles: list[EdgeCutPath] = []
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
            self._invalidate_from(self.STEP_GERBER_IMPORT)
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
            self._invalidate_from(self.STEP_DRILL_IMPORT)
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
            self._invalidate_from(self.STEP_GERBER_IMPORT)
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
            self._invalidate_from(self.STEP_DRILL_IMPORT)
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
            self._invalidate_from(self.STEP_TOOL_SELECTION)
        return changed

    def set_stock_dimensions(
        self,
        *,
        width: float | None = None,
        height: float | None = None,
        thickness: float | None = None,
    ) -> bool:
        changed = False
        if width is not None and self.stock_width != width:
            self.stock_width = width
            changed = True
        if height is not None and self.stock_height != height:
            self.stock_height = height
            changed = True
        if thickness is not None and self.stock_thickness != thickness:
            self.stock_thickness = thickness
            changed = True
        if changed:
            self._invalidate_from(self.STEP_STOCK_DEFINITION)
        return changed

    def set_stock_origin(self, origin: str) -> bool:
        normalized = normalize_nc_origin(origin)
        changed = self.stock_origin != normalized
        self.stock_origin = normalized
        if changed:
            self._invalidate_from(self.STEP_STOCK_DEFINITION)
        return changed

    def set_selected_tool(self, role: str, tool_id: str) -> bool:
        normalized = tool_id.strip()
        changed = self.selected_tools.get(role, "") != normalized
        self.selected_tools[role] = normalized
        if changed:
            self._invalidate_from(self.STEP_TOOL_SELECTION)
        return changed

    def set_layer_assignment(self, role: str, path: Path | None) -> bool:
        resolved = None if path is None else path.resolve()
        changed = self.layer_assignments.get(role) != resolved
        self.layer_assignments[role] = resolved
        if resolved is not None:
            for other_role, assigned_path in self.layer_assignments.items():
                if other_role == role or assigned_path != resolved:
                    continue
                self.layer_assignments[other_role] = None
                changed = True
        if changed:
            self._invalidate_from(self.STEP_GERBER_IMPORT)
        if not self.requires_mirror_setup():
            self.mirror_flip_edge = ""
        return changed

    def set_mirror_flip_edge(self, edge: str) -> bool:
        normalized = edge.strip()
        changed = self.mirror_flip_edge != normalized
        self.mirror_flip_edge = normalized
        if changed:
            self._invalidate_from(self.STEP_GERBER_IMPORT)
        return changed

    def set_mirror_preview_mode(self, mode: str) -> bool:
        normalized = mode.strip() or "side_by_side"
        if normalized not in {"overlay", "side_by_side"}:
            normalized = "side_by_side"
        changed = self.mirror_preview_mode != normalized
        self.mirror_preview_mode = normalized
        return changed

    def set_file_alignment(self, alignment: str) -> bool:
        normalized = normalize_nc_origin(alignment)
        changed = self.file_alignment != normalized
        self.file_alignment = normalized
        if changed:
            self._invalidate_from(self.STEP_ALIGNMENT_HOLES)
        return changed

    def set_file_alignment_offsets(
        self, *, horizontal: float | None = None, vertical: float | None = None
    ) -> bool:
        changed = False
        if horizontal is not None:
            normalized_horizontal = max(0.0, float(horizontal))
            if self.file_alignment_horizontal_offset != normalized_horizontal:
                self.file_alignment_horizontal_offset = normalized_horizontal
                changed = True
        if vertical is not None:
            normalized_vertical = max(0.0, float(vertical))
            if self.file_alignment_vertical_offset != normalized_vertical:
                self.file_alignment_vertical_offset = normalized_vertical
                changed = True
        if changed:
            self._invalidate_from(self.STEP_ALIGNMENT_HOLES)
        return changed

    def set_alignment_grid_size(self, grid_size: float) -> bool:
        normalized = max(0.1, float(grid_size))
        changed = self.alignment_grid_size != normalized
        self.alignment_grid_size = normalized
        if changed:
            self._invalidate_from(self.STEP_ALIGNMENT_HOLES)
        return changed

    def replace_alignment_holes(self, holes: list[AlignmentHole]) -> bool:
        changed = holes != self.alignment_holes
        self.alignment_holes = holes
        if changed:
            self._invalidate_from(self.STEP_ALIGNMENT_HOLES)
        return changed

    def replace_edge_cut_profiles(self, profiles: list[EdgeCutPath]) -> bool:
        changed = profiles != self.edge_cut_profiles
        self.edge_cut_profiles = profiles
        if changed:
            self._invalidate_from(self.STEP_EDGE_CUTS)
        return changed

    def requires_mirror_setup(self) -> bool:
        return (
            self.layer_assignments.get("front_copper") is not None
            and self.layer_assignments.get("back_copper") is not None
        )

    def set_current_step(self, index: int) -> None:
        self.current_step_index = max(0, min(index, self.TOTAL_STEPS - 1))
        self.highest_commenced_step = max(
            self.highest_commenced_step, self.current_step_index
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

    def last_completed_step_index(
        self, implemented_step_count: int | None = None
    ) -> int:
        upper_bound = (
            self.TOTAL_STEPS
            if implemented_step_count is None
            else min(implemented_step_count, self.TOTAL_STEPS)
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
        alignment_holes_data = []
        for hole in self.alignment_holes:
            alignment_holes_data.append(
                {
                    "x_offset": hole.x_offset,
                    "y_offset": hole.y_offset,
                    "diameter": hole.diameter,
                    "mirror_direction": hole.mirror_direction,
                    "enabled": hole.enabled,
                }
            )
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
            "stock": {
                "width": self.stock_width,
                "height": self.stock_height,
                "thickness": self.stock_thickness,
                "origin": self.stock_origin,
            },
            "tool_library": {
                "path": (
                    None
                    if self.tool_library_path is None
                    else self._to_relative_path(
                        self.tool_library_path, self.project_path.parent
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
                "preview_mode": self.mirror_preview_mode,
            },
            "alignment": {
                "file_alignment": self.file_alignment,
                "horizontal_offset": self.file_alignment_horizontal_offset,
                "vertical_offset": self.file_alignment_vertical_offset,
                "grid_size": self.alignment_grid_size,
            },
            "alignment_holes": alignment_holes_data,
            "edge_cuts": {
                "profiles": [
                    {
                        "polygon_keys": list(profile.polygon_keys),
                        "mode": profile.mode,
                        "tool_id": profile.tool_id,
                        "cut_depth": profile.cut_depth,
                        "step_down": profile.step_down,
                        "generated": profile.generated,
                        "visible": profile.visible,
                    }
                    for profile in self.edge_cut_profiles
                ]
            },
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
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
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
        stock_data = loaded.get("stock", {})
        if isinstance(stock_data, dict):
            try:
                project.stock_width = float(stock_data.get("width", 0.0))
            except (TypeError, ValueError):
                project.stock_width = 0.0
            try:
                project.stock_height = float(stock_data.get("height", 0.0))
            except (TypeError, ValueError):
                project.stock_height = 0.0
            try:
                project.stock_thickness = float(stock_data.get("thickness", 0.0))
            except (TypeError, ValueError):
                project.stock_thickness = 0.0
            raw_origin = stock_data.get("origin", project.stock_origin)
            if isinstance(raw_origin, str) and raw_origin.strip():
                project.stock_origin = normalize_nc_origin(raw_origin)
        tool_library_data = loaded.get("tool_library", {})
        if not isinstance(tool_library_data, dict):
            tool_library_data = {}
        raw_tool_library_path = tool_library_data.get("path")
        if isinstance(raw_tool_library_path, str) and raw_tool_library_path.strip():
            project.tool_library_path = project._from_relative_path(
                raw_tool_library_path, project.project_path.parent
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
                        raw_path, project.project_path.parent
                    )
        mirror_data = loaded.get("mirror", {})
        if isinstance(mirror_data, dict):
            raw_edge = mirror_data.get("flip_edge", "")
            if isinstance(raw_edge, str):
                project.mirror_flip_edge = raw_edge.strip()
            raw_mode = mirror_data.get("preview_mode", "side_by_side")
            if isinstance(raw_mode, str):
                project.set_mirror_preview_mode(raw_mode)
        alignment_settings = loaded.get("alignment", {})
        if isinstance(alignment_settings, dict):
            raw_file_alignment = alignment_settings.get("file_alignment", "")
            if isinstance(raw_file_alignment, str) and raw_file_alignment.strip():
                project.file_alignment = normalize_nc_origin(raw_file_alignment)
            try:
                project.file_alignment_horizontal_offset = max(
                    0.0, float(alignment_settings["horizontal_offset"])
                )
            except (TypeError, ValueError):
                project.file_alignment_horizontal_offset = 0.0
            try:
                project.file_alignment_vertical_offset = max(
                    0.0, float(alignment_settings["vertical_offset"])
                )
            except (TypeError, ValueError):
                project.file_alignment_vertical_offset = 0.0
            try:
                project.alignment_grid_size = max(
                    0.1, float(alignment_settings["grid_size"])
                )
            except (TypeError, ValueError):
                project.alignment_grid_size = 5.0
        alignment_data = loaded.get("alignment_holes", [])
        if isinstance(alignment_data, list):
            for item in alignment_data:
                if not isinstance(item, dict):
                    continue
                try:
                    project.alignment_holes.append(
                        AlignmentHole(
                            x_offset=float(item["x_offset"]),
                            y_offset=float(item["y_offset"]),
                            diameter=float(item["diameter"]),
                            mirror_direction=str(item["mirror_direction"]).strip(),
                            enabled=bool(item["enabled"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        edge_cut_data = loaded.get("edge_cuts", {})
        if isinstance(edge_cut_data, dict):
            raw_profiles = edge_cut_data.get("profiles")
            if isinstance(raw_profiles, list):
                for raw_profile in raw_profiles:
                    if not isinstance(raw_profile, dict):
                        continue
                    raw_polygon_keys = raw_profile.get("polygon_keys", [])
                    if not isinstance(raw_polygon_keys, list):
                        continue
                    polygon_keys = [
                        str(key).strip()
                        for key in raw_polygon_keys
                        if isinstance(key, str) and str(key).strip()
                    ]
                    mode = str(raw_profile.get("mode", "")).strip()
                    if polygon_keys and mode:
                        project.edge_cut_profiles.append(
                            EdgeCutPath(
                                polygon_keys=polygon_keys,
                                mode=mode,
                                tool_id=str(raw_profile.get("tool_id", "")).strip(),
                                cut_depth=float(raw_profile.get("cut_depth", 1.8)),
                                step_down=float(raw_profile.get("step_down", 0.4)),
                                generated=bool(raw_profile.get("generated", False)),
                                visible=bool(raw_profile.get("visible", True)),
                            )
                        )
        generated_output_data = loaded.get("generated_outputs", {})
        if isinstance(generated_output_data, dict):
            for key, raw_path in generated_output_data.items():
                if (
                    isinstance(key, str)
                    and isinstance(raw_path, str)
                    and raw_path.strip()
                ):
                    project.generated_outputs[key] = project._from_relative_path(
                        raw_path, project.project_path.parent
                    )
        wizard_data = loaded.get("wizard", {})
        project.current_step_index = project._step_index_for_key(
            wizard_data.get("current_step"), 0
        )
        project.highest_commenced_step = project._step_index_for_key(
            wizard_data.get("highest_commenced_step"), project.current_step_index
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
            path
            for path in project.selected_gerber_paths
            if path in project.gerber_paths
        }
        project.selected_drill_paths = {
            path for path in project.selected_drill_paths if path in project.drill_paths
        }
        project._prune_missing_layer_assignments()
        project._ensure_unique_layer_assignments()
        if not project.requires_mirror_setup():
            project.mirror_flip_edge = ""
        project.dirty_from_step = None
        return project

    def _invalidate_from(self, index: int) -> None:
        self.completed_steps = {step for step in self.completed_steps if step <= index}
        if index < self.highest_commenced_step:
            self.dirty_from_step = (
                index
                if self.dirty_from_step is None
                else min(self.dirty_from_step, index)
            )
        if index <= self.STEP_EDGE_CUTS:
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
        valid_paths = set(self.gerber_paths)
        for role, path in list(self.layer_assignments.items()):
            if path is not None and path not in valid_paths:
                self.layer_assignments[role] = None

    def _ensure_unique_layer_assignments(self) -> None:
        seen_paths: set[Path] = set()
        for role in ("front_copper", "back_copper", "edges"):
            assigned_path = self.layer_assignments.get(role)
            if assigned_path is None:
                continue
            if assigned_path in seen_paths:
                self.layer_assignments[role] = None
                continue
            seen_paths.add(assigned_path)

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
