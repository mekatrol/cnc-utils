from __future__ import annotations
import json
import math
from pathlib import Path
import copy
import sys
import threading
from typing import List, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from geometry.PointInt import PointInt
from geometry.PolylineInt import PolylineInt
from geometry.GeometryInt import GeometryInt
from geometry.GeoUtil import GeoUtil
from geometry.poly_processor import PolyProcessor
from export.JsonExporter import JsonExporter
from svg.SvgConverter import SvgConverter

from views.view_2d import View2D
from views.view_3d import View3D
from views.menubar import Menubar
from views.view_constants import (
    SPINNER_ACTIVE_COLOR,
    SPINNER_BAR_COLOR,
    SPINNER_FRAME_BG_COLOR,
    SVG_STROKE_COLOR,
    TREE_ICON_HIDDEN_COLOR,
    TREE_ICON_SHOWN_COLOR,
)
from views.view_spinner import Spinner


class AppView(tk.Tk):
    def __init__(self):
        super().__init__()
        self.spinner: tk.Toplevel | None = None
        self.title("Polygon Engraver")
        icon_path = Path(__file__).resolve().parents[2] / "assets" / "app-icon.png"
        try:
            self._app_icon = tk.PhotoImage(file=str(icon_path))
            self.iconphoto(True, self._app_icon)
        except Exception:
            self._app_icon = None
        self.geometry("1200x800")
        self.minsize(900, 600)
        try:
            self.tk.call("tk", "scaling", 1.2)  # slightly larger UI if supported
        except Exception:
            pass

        self._first_load_done = False
        self._mapped = False
        self.bind("<Map>", self._on_map)
        self.bind("<Configure>", self._on_configure)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._settings_path = Path(__file__).resolve().parents[2] / "settings.json"

        # Shared state
        self.model: Optional[GeometryInt] = None
        self.source_label_var = tk.StringVar(value="No file loaded")
        self.source_path: Optional[str] = None
        self.selected_polygons = []
        self.generated_paths = []
        settings = self._load_settings()
        path_settings = settings["path_generation"]
        self.hatch_angle_deg = path_settings["hatch_angle_deg"]
        self.hatch_spacing_px = path_settings["hatch_spacing_px"]
        self._loaded_settings = copy.deepcopy(settings)
        self._settings_dirty = False
        self.show_generated_paths = tk.BooleanVar(value=True)
        self.show_geometry = tk.BooleanVar(value=True)
        self._hatch_angle_var = tk.StringVar(value=str(self.hatch_angle_deg))
        self._hatch_spacing_var = tk.StringVar(value=str(self.hatch_spacing_px))
        self._hatch_angle_var.trace_add("write", self._on_settings_var_change)
        self._hatch_spacing_var.trace_add("write", self._on_settings_var_change)
        self.properties_var = tk.StringVar(value="No selection")
        self._tree_item_info = {}
        self._tree_item_action = {}
        self._tree_icon_shown = None
        self._tree_icon_hidden = None
        self._startup_load_params: Optional[tuple[str, int, float]] = None
        self._startup_export_json: Optional[str] = None

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Create main menubar
        self.menubar = Menubar(self)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.main_frame = ttk.Frame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=0)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=0)

        self._build_left_sidebar()
        self._init_tree_icons()

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=1, sticky="nsew")

        self._build_right_sidebar()
        self._update_settings_dirty()

        # Default tabs
        self.add_view("3D")
        self.add_view("2D")

        self.maximize()
        self._refresh_tree()
        self.update_properties()

    def _default_settings(self) -> dict:
        return {
            "app": {
                "version": 1,
            },
            "path_generation": {
                "hatch_angle_deg": 45.0,
                "hatch_spacing_px": 0.25,
            },
        }

    def _upgrade_settings(self, data: dict) -> dict:
        upgraded = dict(data)
        if "app" not in upgraded or not isinstance(upgraded.get("app"), dict):
            upgraded["app"] = {}
        app_settings = upgraded["app"]
        if "path_generation" not in upgraded or not isinstance(
            upgraded.get("path_generation"), dict
        ):
            upgraded["path_generation"] = {}
        path_settings = upgraded["path_generation"]
        if "hatch_angle_deg" in upgraded:
            path_settings["hatch_angle_deg"] = upgraded["hatch_angle_deg"]
        if "hatch_spacing_px" in upgraded:
            path_settings["hatch_spacing_px"] = upgraded["hatch_spacing_px"]
        if "version" in upgraded:
            app_settings["version"] = upgraded["version"]
        app_settings["version"] = 1
        return upgraded

    def _load_settings(self) -> dict:
        settings = self._default_settings()
        data: dict = {}
        save_needed = False

        if self._settings_path.exists():
            try:
                raw = self._settings_path.read_text(encoding="utf-8")
                data = json.loads(raw)
            except Exception:
                data = {}
                save_needed = True
        else:
            save_needed = True

        if not isinstance(data, dict):
            data = {}
            save_needed = True

        app_data = data.get("app")
        app_version = None
        if isinstance(app_data, dict):
            app_version = app_data.get("version")

        if app_version != settings["app"]["version"]:
            data = self._upgrade_settings(data)
            save_needed = True

        path_data = data.get("path_generation")
        if isinstance(path_data, dict):
            for key in ("hatch_angle_deg", "hatch_spacing_px"):
                if key in path_data:
                    try:
                        settings["path_generation"][key] = float(path_data[key])
                    except Exception:
                        save_needed = True
        else:
            save_needed = True

        if save_needed:
            self._save_settings(settings)

        return settings

    def _settings_from_vars(self) -> dict | None:
        try:
            angle = float(self._hatch_angle_var.get())
            spacing = float(self._hatch_spacing_var.get())
        except Exception:
            return None
        version = self._default_settings()["app"]["version"]
        return {
            "app": {
                "version": version,
            },
            "path_generation": {
                "hatch_angle_deg": angle,
                "hatch_spacing_px": spacing,
            },
        }

    def _save_settings(self, settings: dict | None = None) -> None:
        if settings is None:
            settings = self._settings_from_vars()
            if settings is None:
                messagebox.showerror(
                    "Invalid settings",
                    "Hatch angle and spacing must be numeric values.",
                )
                return
            path_settings = settings["path_generation"]
            self.hatch_angle_deg = path_settings["hatch_angle_deg"]
            self.hatch_spacing_px = path_settings["hatch_spacing_px"]
            self.update_properties()
        try:
            self._settings_path.write_text(
                json.dumps(settings, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if hasattr(self, "_loaded_settings"):
                self._loaded_settings = copy.deepcopy(settings)
                self._update_settings_dirty()
        except Exception:
            pass

    def _update_settings_dirty(self) -> None:
        current = self._settings_from_vars()
        if current is None:
            is_dirty = True
        else:
            is_dirty = current != self._loaded_settings
        self._settings_dirty = is_dirty
        if hasattr(self, "_save_button"):
            if is_dirty:
                self._save_button.state(["!disabled"])
            else:
                self._save_button.state(["disabled"])

    def _on_settings_var_change(self, *_args) -> None:
        self._update_settings_dirty()

    def _on_save_settings(self) -> None:
        self._save_settings()

    def _on_restore_defaults(self) -> None:
        defaults = self._default_settings()["path_generation"]
        self._hatch_angle_var.set(str(defaults["hatch_angle_deg"]))
        self._hatch_spacing_var.set(str(defaults["hatch_spacing_px"]))
        self._update_settings_dirty()

    def _on_reset_settings(self) -> None:
        settings = self._load_settings()
        path_settings = settings["path_generation"]
        self._hatch_angle_var.set(str(path_settings["hatch_angle_deg"]))
        self._hatch_spacing_var.set(str(path_settings["hatch_spacing_px"]))
        self._update_settings_dirty()

    def _build_left_sidebar(self) -> None:
        self.left_sidebar = ttk.Frame(self.main_frame, width=240)
        self.left_sidebar.grid(row=0, column=0, sticky="nsew")
        self.left_sidebar.grid_propagate(False)
        self.left_sidebar.rowconfigure(1, weight=1)
        self.left_sidebar.columnconfigure(0, weight=1)

        label = ttk.Label(self.left_sidebar, text="Scene")
        label.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

        self.scene_tree = ttk.Treeview(
            self.left_sidebar, show="tree", selectmode="browse"
        )
        self.scene_tree.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.scene_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.scene_tree.bind("<Button-1>", self._on_tree_left_click)
        self.scene_tree.bind("<Button-3>", self._on_tree_right_click)
        self.scene_tree.bind("<<TreeviewOpen>>", self._on_tree_open)
        self.scene_tree.bind("<<TreeviewClose>>", self._on_tree_close)

        self.tree_geometry_id = self.scene_tree.insert(
            "", "end", text="Geometries", open=True
        )
        self.tree_paths_id = self.scene_tree.insert(
            "", "end", text="Paths", open=True
        )
        self._tree_menu = tk.Menu(self, tearoff=False)

    def _init_tree_icons(self) -> None:
        size = 12
        pad = 6
        width = size + pad
        shown = tk.PhotoImage(width=width, height=size)
        hidden = tk.PhotoImage(width=width, height=size)
        shown.put(TREE_ICON_SHOWN_COLOR, to=(0, 0, size, size))
        hidden.put(TREE_ICON_HIDDEN_COLOR, to=(0, 0, size, size))
        self._tree_icon_shown = shown
        self._tree_icon_hidden = hidden

    def _build_right_sidebar(self) -> None:
        self.right_sidebar = ttk.Frame(self.main_frame, width=260)
        self.right_sidebar.grid(row=0, column=2, sticky="nsew")
        self.right_sidebar.grid_propagate(False)
        self.right_sidebar.rowconfigure(1, weight=1)
        self.right_sidebar.columnconfigure(0, weight=1)

        self._path_settings = ttk.Frame(self.right_sidebar)
        self._path_settings.columnconfigure(1, weight=1)
        angle_label = ttk.Label(self._path_settings, text="Hatch angle (deg)")
        angle_label.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        angle_entry = ttk.Entry(self._path_settings, textvariable=self._hatch_angle_var)
        angle_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 4))
        spacing_label = ttk.Label(self._path_settings, text="Hatch spacing")
        spacing_label.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 4))
        spacing_entry = ttk.Entry(
            self._path_settings, textvariable=self._hatch_spacing_var
        )
        spacing_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 4))
        angle_entry.bind("<Return>", self._apply_hatch_settings)
        spacing_entry.bind("<Return>", self._apply_hatch_settings)
        angle_entry.bind("<FocusOut>", self._apply_hatch_settings)
        spacing_entry.bind("<FocusOut>", self._apply_hatch_settings)
        self._save_button = ttk.Button(
            self._path_settings, text="Save settings", command=self._on_save_settings
        )
        self._save_button.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )
        self._save_button.state(["disabled"])
        self._restore_defaults_button = ttk.Button(
            self._path_settings,
            text="Reset to defaults",
            command=self._on_restore_defaults,
        )
        self._restore_defaults_button.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )
        self._reset_button = ttk.Button(
            self._path_settings, text="Reload settings", command=self._on_reset_settings
        )
        self._reset_button.grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )

        label = ttk.Label(self.right_sidebar, text="Properties")
        label.grid(row=1, column=0, sticky="w", padx=8, pady=(8, 4))

        props = ttk.Label(
            self.right_sidebar,
            textvariable=self.properties_var,
            justify="left",
            wraplength=240,
        )
        props.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def _on_close(self) -> None:
        self._save_settings()
        self.destroy()

    def _on_tree_select(self, _event) -> None:
        selection = self.scene_tree.selection()
        if not selection:
            return
        info = self._tree_item_info.get(selection[0])
        if info:
            self.properties_var.set(info)
        if selection[0] == self.tree_paths_id:
            self._show_path_settings()
        else:
            self._hide_path_settings()

    def _on_tree_left_click(self, event) -> str | None:
        row_id = self.scene_tree.identify_row(event.y)
        if not row_id:
            return None
        element = self.scene_tree.identify("element", event.x, event.y)
        if element != "image":
            return None
        self.scene_tree.selection_set(row_id)
        if row_id == self.tree_paths_id:
            self._toggle_all_paths_visibility()
            return "break"
        action = self._tree_item_action.get(row_id)
        if not action:
            return "break"
        kind, payload = action
        if kind == "geometry":
            self._toggle_geometry_visibility()
            return "break"
        if kind == "path":
            self._toggle_path_visibility(payload)
            return "break"
        if kind == "path_child":
            path_index, key = payload
            self._toggle_path_child_visibility(path_index, key)
            return "break"
        return "break"

    def _show_path_settings(self) -> None:
        if not hasattr(self, "_path_settings"):
            return
        self._hatch_angle_var.set(str(self.hatch_angle_deg))
        self._hatch_spacing_var.set(str(self.hatch_spacing_px))
        self._path_settings.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 4))

    def _hide_path_settings(self) -> None:
        if not hasattr(self, "_path_settings"):
            return
        self._path_settings.grid_remove()

    def _apply_hatch_settings(self, _event=None) -> None:
        try:
            angle = float(self._hatch_angle_var.get())
            spacing = float(self._hatch_spacing_var.get())
        except Exception:
            return
        self.hatch_angle_deg = angle
        self.hatch_spacing_px = spacing
        self.update_properties()
        self._update_settings_dirty()

    def _on_tree_right_click(self, event) -> None:
        row_id = self.scene_tree.identify_row(event.y)
        if not row_id:
            return
        self.scene_tree.selection_set(row_id)
        if row_id == self.tree_paths_id:
            action = ("paths_root", None)
        elif row_id == self.tree_geometry_id:
            action = ("geometry_root", None)
        else:
            action = self._tree_item_action.get(row_id)
        if not action:
            return
        self._tree_menu.delete(0, "end")
        kind, payload = action
        if kind == "geometry":
            is_visible = self.show_geometry.get()
            label = "Hide" if is_visible else "Show"
            self._tree_menu.add_command(
                label=label, command=self._toggle_geometry_visibility
            )
            self._tree_menu.add_command(
                label="Centre to origin", command=self.centre_to_origin
            )
            self._tree_menu.add_separator()
            self._tree_menu.add_command(
                label="Fix Self Intersecting Polygons", command=self.simplify_polygons
            )
            self._tree_menu.add_command(
                label="Close Polygons", command=self.close_polygons
            )
            self._tree_menu.add_command(label="Clip All", command=self.clip_all)
        elif kind == "geometry_root":
            self._tree_menu.add_command(
                label="Centre to origin (all)", command=self.centre_to_origin
            )
        elif kind == "paths_root":
            self._tree_menu.add_command(
                label="Generate paths for selected polygons",
                command=lambda: self.generate_paths_for_selection(append=True),
            )
            label = "Hide All" if self.show_generated_paths.get() else "Show All"
            self._tree_menu.add_command(
                label=label, command=self._toggle_all_paths_visibility
            )
            self._tree_menu.add_command(
                label="Remove All", command=self._remove_all_paths
            )
        elif kind == "path":
            entry = self._get_path_entry(payload)
            if entry is None:
                return
            is_visible = entry.get("visible", True)
            label = "Hide" if is_visible else "Show"
            self._tree_menu.add_command(
                label=label,
                command=lambda idx=payload: self._toggle_path_visibility(idx),
            )
            self._tree_menu.add_command(
                label="Remove", command=lambda idx=payload: self._remove_path(idx)
            )
        elif kind == "path_child":
            path_index, key = payload
            entry = self._get_path_entry(path_index)
            if entry is None:
                return
            child = self._get_path_child(entry, key)
            if child is None:
                return
            is_visible = child.get("visible", True)
            label = "Hide" if is_visible else "Show"
            self._tree_menu.add_command(
                label=label,
                command=lambda idx=path_index, k=key: self._toggle_path_child_visibility(
                    idx, k
                ),
            )
        else:
            return
        try:
            self._tree_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._tree_menu.grab_release()

    def _on_tree_open(self, event) -> None:
        row_id = event.widget.focus()
        action = self._tree_item_action.get(row_id)
        if not action:
            return
        kind, payload = action
        if kind == "path":
            entry = self._get_path_entry(payload)
            if entry is not None:
                entry["expanded"] = True

    def _on_tree_close(self, event) -> None:
        row_id = event.widget.focus()
        action = self._tree_item_action.get(row_id)
        if not action:
            return
        kind, payload = action
        if kind == "path":
            entry = self._get_path_entry(payload)
            if entry is not None:
                entry["expanded"] = False

    def _get_path_entry(self, index: int):
        if index < 0:
            return None
        if index >= len(self.generated_paths):
            return None
        return self.generated_paths[index]

    def _ensure_path_children(self, entry: dict) -> list[dict]:
        children = entry.get("children")
        if isinstance(children, list) and children:
            return children
        lines = entry.get("lines", {})
        children = []
        if lines.get("primary"):
            children.append(
                {"key": "primary", "name": "Fill (primary)", "visible": True}
            )
        if lines.get("secondary"):
            children.append(
                {"key": "secondary", "name": "Fill (secondary)", "visible": True}
            )
        boundary_keys = [key for key in lines.keys() if key.startswith("boundary")]
        for key in sorted(boundary_keys):
            name = "Boundary" if key == "boundary" else key.replace("_", " ").title()
            children.append({"key": key, "name": name, "visible": True})
        entry["children"] = children
        return children

    def _get_path_child(self, entry: dict, key: str) -> dict | None:
        children = self._ensure_path_children(entry)
        for child in children:
            if child.get("key") == key:
                return child
        return None

    def _toggle_geometry_visibility(self) -> None:
        self.show_geometry.set(not self.show_geometry.get())
        if not self.show_geometry.get():
            self.selected_polygons = []
        self._refresh_tree()
        self.update_properties()
        self.redraw_all()

    def _toggle_all_paths_visibility(self) -> None:
        self.show_generated_paths.set(not self.show_generated_paths.get())
        self._refresh_tree()
        self.update_properties()
        self.redraw_all()

    def _remove_all_paths(self) -> None:
        if not self.generated_paths:
            return
        self.generated_paths = []
        self.show_generated_paths.set(True)
        self._refresh_tree()
        self.update_properties()
        self.redraw_all()

    def _remove_path(self, index: int) -> None:
        entry = self._get_path_entry(index)
        if entry is None:
            return
        self.generated_paths.pop(index)
        self._refresh_tree()
        self.update_properties()
        self.redraw_all()

    def _toggle_path_visibility(self, index: int) -> None:
        entry = self._get_path_entry(index)
        if entry is None:
            return
        entry["visible"] = not entry.get("visible", True)
        self._refresh_tree()
        self.update_properties()
        self.redraw_all()

    def _toggle_path_child_visibility(self, index: int, key: str) -> None:
        entry = self._get_path_entry(index)
        if entry is None:
            return
        child = self._get_path_child(entry, key)
        if child is None:
            return
        child["visible"] = not child.get("visible", True)
        self._refresh_tree()
        self.update_properties()
        self.redraw_all()

    def _refresh_tree(self) -> None:
        if not hasattr(self, "scene_tree"):
            return

        self._hide_path_settings()
        self._tree_item_info = {}
        self._tree_item_action = {}
        self.scene_tree.delete(*self.scene_tree.get_children(self.tree_geometry_id))
        self.scene_tree.delete(*self.scene_tree.get_children(self.tree_paths_id))
        self._tree_item_info[self.tree_paths_id] = "Generated paths"
        self._tree_item_action[self.tree_paths_id] = ("paths_root", None)

        if self.model and self.model.polylines:
            source = self.source_path or "in-memory geometry"
            status = "shown" if self.show_geometry.get() else "hidden"
            icon = (
                self._tree_icon_shown
                if self.show_geometry.get()
                else self._tree_icon_hidden
            )
            geom_item = self.scene_tree.insert(
                self.tree_geometry_id,
                "end",
                text=f"{Path(source).name} ({status})",
                open=True,
                image=icon,
            )
            self._tree_item_info[geom_item] = f"Source: {source}"
            self._tree_item_action[geom_item] = ("geometry", None)
            count_item = self.scene_tree.insert(
                geom_item,
                "end",
                text=f"Polylines: {len(self.model.polylines)}",
            )
            self._tree_item_info[count_item] = f"Polylines: {len(self.model.polylines)}"
        else:
            none_item = self.scene_tree.insert(
                self.tree_geometry_id,
                "end",
                text="No geometry loaded",
                image=self._tree_icon_hidden,
            )
            self._tree_item_info[none_item] = "No geometry loaded"

        if self.generated_paths:
            for idx, entry in enumerate(self.generated_paths):
                polygon_index = entry.get("polygon_index", "?")
                lines = entry.get("lines", {})
                primary = len(lines.get("primary", []))
                secondary = len(lines.get("secondary", []))
                boundary = sum(
                    len(lines.get(key, []))
                    for key in lines.keys()
                    if key.startswith("boundary")
                )
                status = "shown" if entry.get("visible", True) else "hidden"
                if not self.show_generated_paths.get():
                    status = "hidden"
                is_visible = self.show_generated_paths.get() and entry.get(
                    "visible", True
                )
                icon = self._tree_icon_shown if is_visible else self._tree_icon_hidden
                item = self.scene_tree.insert(
                    self.tree_paths_id,
                    "end",
                    text=f"Path {polygon_index} ({status})",
                    image=icon,
                    open=entry.get("expanded", True),
                )
                self._tree_item_info[item] = (
                    f"Path {polygon_index}\n"
                    f"Primary lines: {primary}\n"
                    f"Secondary lines: {secondary}\n"
                    f"Boundary segments: {boundary}"
                )
                self._tree_item_action[item] = ("path", idx)
                children = self._ensure_path_children(entry)
                for child in children:
                    key = child.get("key")
                    if not key:
                        continue
                    child_count = len(lines.get(key, []))
                    child_visible = child.get("visible", True)
                    child_status = "shown" if child_visible else "hidden"
                    if not is_visible:
                        child_status = "hidden"
                    child_is_visible = is_visible and child_visible
                    child_icon = (
                        self._tree_icon_shown
                        if child_is_visible
                        else self._tree_icon_hidden
                    )
                    child_item = self.scene_tree.insert(
                        item,
                        "end",
                        text=f"{child.get('name', key)} ({child_status})",
                        image=child_icon,
                    )
                    self._tree_item_info[child_item] = (
                        f"{child.get('name', key)}\n"
                        f"Segments: {child_count}"
                    )
                    self._tree_item_action[child_item] = ("path_child", (idx, key))
        else:
            none_item = self.scene_tree.insert(
                self.tree_paths_id,
                "end",
                text="No generated paths",
                image=self._tree_icon_hidden,
            )
            self._tree_item_info[none_item] = "No generated paths"

    def update_properties(self) -> None:
        lines = []
        if self.model and self.model.polylines:
            source = self.source_path or "(in-memory geometry)"
            lines.append(f"Source: {source}")
            lines.append(f"Polylines: {len(self.model.polylines)}")
            lines.append(
                "Geometry: shown" if self.show_geometry.get() else "Geometry: hidden"
            )
        else:
            lines.append("No geometry loaded")

        if self.selected_polygons:
            selected = self.selected_polygons[0]["polygon"]
            holes = self.selected_polygons[0]["holes"]
            scale = self.model.scale if self.model and self.model.scale else 1
            area = abs(GeoUtil.area(selected["points"])) / (scale * scale)
            lines.append(f"Selected polygon: {selected['index']}")
            lines.append(f"Holes: {len(holes)}")
            lines.append(f"Area: {area:.3f}")
        else:
            lines.append("Selected polygon: none")

        path_count = len(self.generated_paths)
        visible_count = sum(
            1 for entry in self.generated_paths if entry.get("visible", True)
        )
        visibility = "shown" if self.show_generated_paths.get() else "hidden"
        lines.append(f"Generated paths: {visible_count}/{path_count} ({visibility})")

        self.properties_var.set("\n".join(lines))

    def _on_map(self, _):
        self._mapped = True

    def _on_configure(self, _):
        if self._first_load_done or not self._mapped:
            return

        # Debounce sziing events on startup: run after resize/placement bursts stop
        if getattr(self, "_settle_job", None):
            self.after_cancel(self._settle_job)

        self._settle_job = self.after(150, self._call_first_loaded)

    def _call_first_loaded(self):
        if self._first_load_done:
            return
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 1 or h <= 1:
            # still not realized; check again shortly
            self._settle_job = self.after(50, self._call_first_loaded)
            return
        self._first_load_done = True
        # optional: unbind to avoid later calls
        self.unbind("<Map>")
        self.unbind("<Configure>")
        self.on_first_loaded()  # <-- your method

    def on_first_loaded(self):
        # run once when window is visible and size is final-ish
        self.fit_current()

    def maximize(self):
        self.update_idletasks()
        # 1) Try native "zoomed" (Windows, many X11)
        try:
            self.state("zoomed")
            return
        except tk.TclError:
            pass
        # 2) Some X11 WMs expose -zoomed
        try:
            self.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass
        # 3) macOS fallback (fullscreen) or generic geometry fill
        if sys.platform == "darwin":
            self.attributes("-fullscreen", True)  # Esc to exit if you add a binding
        else:
            self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

    @staticmethod
    def _as_point(pt) -> PointInt:
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            return PointInt(int(pt["x"]), int(pt["y"]))
        if (
            isinstance(pt, (list, tuple))
            and len(pt) >= 2
            and all(isinstance(v, (int, float)) for v in pt[:2])
        ):
            return PointInt(int(pt[0]), int(pt[1]))
        raise ValueError(f"Unsupported point format: {pt!r}")

    @staticmethod
    def load_geometry_from_json(path: Path) -> GeometryInt:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scale = int(data.get("scale", 1) or 1)

        pls: List[PolylineInt] = []

        # Case 1: standard "polylines": [{"pts": [...]}, ...]
        if isinstance(data.get("polylines"), list):
            for pl in data.get("polylines", []):
                pts_raw = pl.get("pts", []) if isinstance(pl, dict) else []
                pts = [AppView._as_point(p) for p in pts_raw]
                if len(pts) >= 2:
                    pls.append(PolylineInt(pts))

        # Case 2: root-level "points": [ [ [x,y], ... ], [ ... ] ]
        elif isinstance(data.get("points"), list):
            for poly in data.get("points", []):
                if isinstance(poly, list):
                    pts = [AppView._as_point(p) for p in poly]
                    if len(pts) >= 2:
                        pls.append(PolylineInt(pts))

        return GeometryInt(pls, [], scale)

    # Geometry management
    def set_geometry(self, geom: GeometryInt, source: str = ""):
        self.model = geom
        self.source_label_var.set(source or "(in-memory geometry)")
        self.source_path = source or None
        self.selected_polygons = []
        self.generated_paths = []
        for i in range(self.notebook.index("end")):
            widget = self.notebook.nametowidget(self.notebook.tabs()[i])
            if isinstance(widget, View3D):
                widget.fit_to_view_pending = True
        self._refresh_tree()
        self.update_properties()
        self._maybe_export_startup(geom)
        self.redraw_all()

    def redraw_all(self):
        for i in range(self.notebook.index("end")):
            widget = self.notebook.nametowidget(self.notebook.tabs()[i])
            if hasattr(widget, "redraw"):
                widget.redraw()

    def open_file_dialog(self):
        path = filedialog.askopenfilename(
            title="Open geometry JSON",
            filetypes=[
                ("SVG Files", "*.svg"),
                ("JSON Files", "*.json"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            ext = Path(path).suffix.lower()
            if ext == ".json":
                geom = AppView.load_geometry_from_json(Path(path))
                self.set_geometry(geom, source=str(path))
                self.fit_current()
            elif ext == ".svg":
                self.open_svg_async(str(path))
            else:
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {path}:\n{e}")
            return

    def queue_startup_load(
        self,
        input_path: str,
        scale: int = 10000,
        tol: float = 0.25,
        export_json: str | None = None,
    ):
        self._startup_load_params = (input_path, scale, tol)
        self._startup_export_json = export_json
        self.after(0, self._run_startup_load)

    def _run_startup_load(self):
        if not self._startup_load_params:
            return
        path, scale, tol = self._startup_load_params
        self._startup_load_params = None
        try:
            file_path = Path(path)
            if not file_path.exists():
                messagebox.showerror("Load failed", f"File not found:\n{path}")
                return
            ext = file_path.suffix.lower()
            if ext == ".json":
                geom = AppView.load_geometry_from_json(file_path)
                self.set_geometry(geom, source=str(file_path))
                self.fit_current()
            elif ext == ".svg":
                self.open_svg_async(str(file_path), scale=scale, tol=tol)
            else:
                messagebox.showerror("Load failed", f"Unsupported file type:\n{path}")
        except Exception as e:
            messagebox.showerror("Load failed", f"{path}\n\n{e}")

    def _center_in_parent(self, win: tk.Toplevel):
        win.update_idletasks()  # ensure geometry is calculated

        pw, ph = self.winfo_width(), self.winfo_height()
        px, py = self.winfo_rootx(), self.winfo_rooty()

        ww, wh = win.winfo_width(), win.winfo_height()

        x = px + (pw // 2) - (ww // 2)
        y = py + (ph // 2) - (wh // 2)

        win.geometry(f"+{x}+{y}")

    def _show_spinner(self, message):
        # show spinner dialog
        self.spinner = tk.Toplevel(self)
        self.spinner.withdraw()
        self.spinner.overrideredirect(True)

        frame = tk.Frame(self.spinner, bg=SPINNER_FRAME_BG_COLOR, padx=15, pady=15)
        frame.pack(fill="both", expand=True)

        style = ttk.Style(self.spinner)
        style.configure(
            "Blue.Horizontal.Bar",
            troughcolor=frame.cget("bg"),  # background area
            background=SPINNER_BAR_COLOR,  # moving bar color
        )

        ttk.Label(
            frame,
            text=message,
            background=frame.cget("bg"),
            font=("Helvetica", 24, "bold"),
        ).pack(padx=0, pady=0)

        sp = Spinner(
            frame, size=64, thickness=6, color=SPINNER_ACTIVE_COLOR, bg=frame.cget("bg")
        )
        sp.pack()
        sp.start()
        self._center_in_parent(self.spinner)
        self.spinner.deiconify()

    def open_svg_async(self, path: str, scale: int = 10000, tol: float = 0.25):
        self._show_spinner("Processing…")
        threading.Thread(
            target=self._load_svg_worker, args=(path, scale, tol), daemon=True
        ).start()

    def _load_svg_worker(self, path: str, scale: int, tol: float):
        try:
            geom = SvgConverter.svg_to_geometry_int(path, scale=scale, tol=tol)
        except Exception as e:
            # report back to UI thread
            self.after(0, self._load_svg_failed, path, e)
            return

        # hand result to UI thread
        self.after(0, lambda: self._load_svg_done(path, geom))

    def _load_svg_done(self, path: str, geom):
        self._hide_spinner()
        self.set_geometry(geom, source=path)  # safe here
        self.fit_current()

    def _load_svg_failed(self, path: str, err: Exception):
        self._hide_spinner()
        from tkinter import messagebox

        messagebox.showerror("Load failed", f"{path}\n\n{err}")

    def _hide_spinner(self):
        if self.spinner and self.spinner.winfo_exists():
            self.spinner.destroy()

    def _maybe_export_startup(self, geom: GeometryInt):
        if not self._startup_export_json:
            return
        export_path = Path(self._startup_export_json)
        self._startup_export_json = None
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            JsonExporter.export(geom, str(export_path))
        except Exception as e:
            messagebox.showerror("Export failed", f"{export_path}\n\n{e}")

    def add_view(self, kind: str):
        if kind == "2D":
            frame = View2D(self.notebook, self)
            title = "2D View"
        elif kind == "3D":
            frame = View3D(self.notebook, self)
            title = "3D View"
        else:
            raise ValueError(f"Unknown view kind: {kind}")

        self.notebook.add(frame, text=title)
        self.notebook.select(frame)
        self.notebook.update_idletasks()  # ensure geometry is computed
        return frame

    def fit_current(self):
        cur = self.notebook.select()
        if not cur:
            return
        widget = self.nametowidget(cur)
        if hasattr(widget, "fit_to_view"):
            widget.fit_to_view()

    def fit_current_including_origin(self):
        cur = self.notebook.select()
        if not cur:
            return
        widget = self.nametowidget(cur)
        if hasattr(widget, "fit_to_view"):
            widget.fit_to_view(include_origin=True)

    def centre_to_origin(self):
        if not self.model:
            messagebox.showinfo("Centre to Origin", "No geometry loaded.")
            return
        minx, miny, maxx, maxy = self.model.bounds()

        if minx == 0 and miny == 0:
            self.fit_current()
            return

        midx = minx + (maxx - minx) / 2.0
        midy = miny + (maxy - miny) / 2.0
        midx_i = int(round(midx))
        midy_i = int(round(midy))

        new_polylines: List[PolylineInt] = []

        for pl in self.model.polylines:
            moved = [PointInt(p.x - midx_i, p.y - midy_i) for p in pl.points]
            new_polylines.append(
                PolylineInt(moved, simplify_tolerance=pl.simplify_tolerance)
            )
        new_points = [PointInt(p.x - midx_i, p.y - midy_i) for p in self.model.points]
        self.model = GeometryInt(new_polylines, new_points, self.model.scale)

        for i in range(self.notebook.index("end")):
            widget = self.notebook.nametowidget(self.notebook.tabs()[i])
            if hasattr(widget, "_rebuild_selected_polygons"):
                widget._rebuild_selected_polygons()
            if hasattr(widget, "_pivot_center"):
                widget._pivot_center = None

        self.menubar.files_dirty = True
        self.redraw_all()
        self.fit_current()

    def simplify_polygons(self) -> None:
        self._run_polygon_processing(
            "Fixing self intersecting polygons…",
            lambda polylines, _scale: PolyProcessor.split_self_intersections(polylines),
        )

    def close_polygons(self) -> None:
        self._run_polygon_processing(
            "Closing polygons…",
            lambda polylines, _scale: PolyProcessor.close_open_polylines(polylines),
        )

    def clip_all(self) -> None:
        self._run_polygon_processing(
            "Clipping polygons…",
            lambda polylines, scale: PolyProcessor.split_intersections_between_polygons(
                polylines, scale
            ),
        )

    def _run_polygon_processing(self, message: str, func) -> None:
        if not self.model or not self.model.polylines:
            messagebox.showinfo("Geometry Operation", "No geometry loaded.")
            return
        polylines = self.model.polylines
        scale = int(self.model.scale) if self.model.scale else 1
        self._show_spinner(message)
        threading.Thread(
            target=self._polygon_processing_worker,
            args=(func, polylines, scale),
            daemon=True,
        ).start()

    def _polygon_processing_worker(self, func, polylines, scale: int) -> None:
        try:
            result = func(polylines, scale)
        except Exception as e:
            self.after(0, lambda: self._polygon_processing_failed(e))
            return
        self.after(0, lambda: self._polygon_processing_done(result))

    def _polygon_processing_done(self, polylines: List[PolylineInt]) -> None:
        self._hide_spinner()
        if not self.model:
            return
        self.model = GeometryInt(polylines, self.model.points, self.model.scale)
        self.selected_polygons = []
        self.generated_paths = []
        self.show_generated_paths.set(True)
        self.menubar.files_dirty = True
        self._refresh_tree()
        self.update_properties()
        self.redraw_all()

    def _polygon_processing_failed(self, err: Exception) -> None:
        self._hide_spinner()
        messagebox.showerror("Geometry Operation Failed", f"{err}")

    def _format_svg_number(self, value: float) -> str:
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"

    def _write_svg(self, path: str):
        if not self.model:
            return
        minx, miny, maxx, maxy = self.model.bounds()
        scale = self.model.scale or 1
        width = max((maxx - minx) / scale, 1e-6)
        height = max((maxy - miny) / scale, 1e-6)
        view_minx = minx / scale
        view_miny = -maxy / scale

        def fmt(val: float) -> str:
            return self._format_svg_number(val)

        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="{fmt(view_minx)} {fmt(view_miny)} {fmt(width)} {fmt(height)}" '
                f'width="{fmt(width)}" height="{fmt(height)}">'
            ),
        ]

        for pl in self.model.polylines:
            if len(pl.points) < 2:
                continue
            pts = [f"{fmt(p.x / scale)} {fmt(-p.y / scale)}" for p in pl.points]
            d = "M " + " L ".join(pts)
            lines.append(f'  <path d="{d}" fill="none" stroke="{SVG_STROKE_COLOR}" />')
        lines.append("</svg>")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def save_svg_as(self):
        if not self.model:
            messagebox.showinfo("Save SVG", "No geometry loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Save SVG As",
            defaultextension=".svg",
            filetypes=[("SVG Files", "*.svg"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            self._write_svg(path)
        except Exception as e:
            messagebox.showerror("Save failed", f"{path}\n\n{e}")
            return
        self.menubar.files_dirty = False

    def generate_paths_for_selection(self, append: bool = False) -> None:
        if not self.selected_polygons:
            messagebox.showinfo("Generate Paths", "No polygons selected.")
            return

        hatch_angle = float(self.hatch_angle_deg)
        hatch_spacing = max(1e-6, float(self.hatch_spacing_px))
        scale = int(self.model.scale) if self.model and self.model.scale else 1
        selection = []
        for entry in self.selected_polygons:
            polygon = entry["polygon"]
            holes = entry["holes"]
            selection.append(
                {
                    "polygon_index": polygon["index"],
                    "polygon_points": [(p.x, p.y) for p in polygon["points"]],
                    "holes": [[(p.x, p.y) for p in hole["points"]] for hole in holes],
                }
            )

        self._show_spinner("Generating paths…")
        threading.Thread(
            target=self._generate_paths_worker,
            args=(selection, hatch_angle, hatch_spacing, scale, append),
            daemon=True,
        ).start()

    def _generate_paths_worker(
        self,
        selection,
        hatch_angle: float,
        hatch_spacing: float,
        scale: int,
        append: bool,
    ):
        try:
            paths = []
            for entry in selection:
                polygon_points = [PointInt(x, y) for x, y in entry["polygon_points"]]
                holes = [[PointInt(x, y) for x, y in hole] for hole in entry["holes"]]
                primary = self._hatch_lines_for_polygon(
                    polygon_points,
                    holes,
                    hatch_angle,
                    hatch_spacing,
                    scale,
                )
                secondary = self._hatch_lines_for_polygon(
                    polygon_points,
                    holes,
                    hatch_angle + 90.0,
                    hatch_spacing,
                    scale,
                )
                boundary_outer = self._polygon_boundary_segments(polygon_points, scale)
                boundary_children = [
                    {
                        "key": "boundary_outer",
                        "name": "Boundary (outer)",
                        "visible": True,
                    }
                ]
                lines = {
                    "primary": primary,
                    "secondary": secondary,
                    "boundary_outer": boundary_outer,
                }
                for hole_index, hole in enumerate(holes, start=1):
                    key = f"boundary_hole_{hole_index}"
                    lines[key] = self._polygon_boundary_segments(hole, scale)
                    boundary_children.append(
                        {
                            "key": key,
                            "name": f"Boundary (hole {hole_index})",
                            "visible": True,
                        }
                    )
                children = [
                    {"key": "primary", "name": "Fill (primary)", "visible": True},
                    {"key": "secondary", "name": "Fill (secondary)", "visible": True},
                ]
                children.extend(boundary_children)
                paths.append(
                    {
                        "polygon_index": entry["polygon_index"],
                        "hatch_angle_deg": hatch_angle,
                        "hatch_spacing": hatch_spacing,
                        "visible": True,
                        "lines": lines,
                        "children": children,
                    }
                )
        except Exception as e:
            self.after(0, self._generate_paths_failed, e)
            return

        self.after(0, self._generate_paths_done, paths, append)

    def _generate_paths_done(self, paths, append: bool = False):
        self._hide_spinner()
        if append:
            self.generated_paths.extend(paths)
        else:
            self.generated_paths = paths
        self.menubar.files_dirty = True
        self._refresh_tree()
        self.update_properties()
        self.fit_current()

    def _generate_paths_failed(self, err: Exception):
        self._hide_spinner()
        messagebox.showerror("Generate Paths Failed", f"{err}")

    @staticmethod
    def _frange(start: float, end: float, step: float):
        value = start
        while value <= end + 1e-6:
            yield value
            value += step

    @staticmethod
    def _line_segment_intersection(direction, normal, offset, seg_a, seg_b):
        dx, dy = direction
        nx, ny = normal
        ax, ay = seg_a
        bx, by = seg_b
        sx = bx - ax
        sy = by - ay

        denom = nx * sx + ny * sy
        if abs(denom) < 1e-9:
            return None

        t = (offset - (nx * ax + ny * ay)) / denom
        if t < 0.0 or t > 1.0:
            return None

        x = ax + t * sx
        y = ay + t * sy
        line_t = x * dx + y * dy
        return line_t

    @staticmethod
    def _polygon_line_intervals(points, direction, normal, offset):
        intersections = []
        for i in range(len(points)):
            p0 = points[i]
            p1 = points[(i + 1) % len(points)]
            hit = AppView._line_segment_intersection(direction, normal, offset, p0, p1)
            if hit is not None:
                intersections.append(hit)

        if len(intersections) < 2:
            return []

        intersections.sort()
        deduped = []
        for t in intersections:
            if not deduped or abs(t - deduped[-1]) > 1e-6:
                deduped.append(t)

        intervals = []
        for i in range(0, len(deduped) - 1, 2):
            t0 = deduped[i]
            t1 = deduped[i + 1]
            if t1 > t0 + 1e-9:
                intervals.append((t0, t1))
        return intervals

    @staticmethod
    def _build_edge_cache(points, direction, normal):
        dx, dy = direction
        nx, ny = normal
        edges = []
        for i in range(len(points)):
            ax, ay = points[i]
            bx, by = points[(i + 1) % len(points)]
            sx = bx - ax
            sy = by - ay
            denom = nx * sx + ny * sy
            if abs(denom) < 1e-9:
                continue
            base = nx * ax + ny * ay
            adot = ax * dx + ay * dy
            sdot = sx * dx + sy * dy
            edges.append((base, denom, adot, sdot))
        return edges

    @staticmethod
    def _polygon_line_intervals_cached(edges, offset):
        intersections = []
        for base, denom, adot, sdot in edges:
            t = (offset - base) / denom
            if t < 0.0 or t > 1.0:
                continue
            intersections.append(adot + t * sdot)

        if len(intersections) < 2:
            return []

        intersections.sort()
        deduped = []
        for t in intersections:
            if not deduped or abs(t - deduped[-1]) > 1e-6:
                deduped.append(t)

        intervals = []
        for i in range(0, len(deduped) - 1, 2):
            t0 = deduped[i]
            t1 = deduped[i + 1]
            if t1 > t0 + 1e-9:
                intervals.append((t0, t1))
        return intervals

    @staticmethod
    def _merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda item: item[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + 1e-6:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _subtract_intervals(base_intervals, cut_intervals):
        if not cut_intervals:
            return list(base_intervals)
        result = []
        for base_start, base_end in base_intervals:
            cur = base_start
            for cut_start, cut_end in cut_intervals:
                if cut_end <= cur:
                    continue
                if cut_start >= base_end:
                    break
                if cut_start > cur:
                    result.append((cur, min(cut_start, base_end)))
                cur = max(cur, cut_end)
                if cur >= base_end:
                    break
            if cur < base_end:
                result.append((cur, base_end))
        return result

    @staticmethod
    def _polygon_boundary_segments(points: List[PointInt], scale: int):
        if len(points) < 2:
            return []
        segments = []
        count = len(points)
        for i in range(count):
            p0 = points[i]
            p1 = points[(i + 1) % count]
            segments.append(
                [
                    [p0.x / scale, p0.y / scale],
                    [p1.x / scale, p1.y / scale],
                ]
            )
        return segments

    def _hatch_lines_for_polygon(
        self,
        polygon_points: List[PointInt],
        holes: List[List[PointInt]],
        angle_deg: float,
        spacing_world: float,
        scale: int,
    ):
        if len(polygon_points) < 3:
            return []

        spacing = max(1.0, spacing_world * scale)
        direction = (
            math.cos(math.radians(angle_deg)),
            math.sin(math.radians(angle_deg)),
        )
        normal = (-direction[1], direction[0])

        points = [(p.x, p.y) for p in polygon_points]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        corners = [
            (minx, miny),
            (minx, maxy),
            (maxx, miny),
            (maxx, maxy),
        ]
        offsets = [corner[0] * normal[0] + corner[1] * normal[1] for corner in corners]
        min_o, max_o = min(offsets), max(offsets)

        start = math.floor(min_o / spacing) * spacing
        end = math.ceil(max_o / spacing) * spacing

        hole_points = [[(p.x, p.y) for p in hole] for hole in holes]
        outer_edges = self._build_edge_cache(points, direction, normal)
        hole_edges = [
            self._build_edge_cache(hole, direction, normal) for hole in hole_points
        ]
        lines = []

        steps = int(math.floor((end - start) / spacing + 1.0 + 1e-6))
        for step in range(steps):
            offset = start + step * spacing
            outer_intervals = self._polygon_line_intervals_cached(outer_edges, offset)
            if not outer_intervals:
                continue
            hole_intervals = []
            for edges in hole_edges:
                hole_intervals.extend(
                    self._polygon_line_intervals_cached(edges, offset)
                )
            hole_intervals = self._merge_intervals(hole_intervals)
            final_intervals = self._subtract_intervals(outer_intervals, hole_intervals)

            for t0, t1 in final_intervals:
                p0 = (
                    direction[0] * t0 + normal[0] * offset,
                    direction[1] * t0 + normal[1] * offset,
                )
                p1 = (
                    direction[0] * t1 + normal[0] * offset,
                    direction[1] * t1 + normal[1] * offset,
                )
                lines.append(
                    [
                        [p0[0] / scale, p0[1] / scale],
                        [p1[0] / scale, p1[1] / scale],
                    ]
                )

        return lines

    def clear_generated_paths(self) -> None:
        if not self.generated_paths:
            return
        self.generated_paths = []
        self.show_generated_paths.set(True)
        self.menubar.files_dirty = True
        self._refresh_tree()
        self.update_properties()
        self.redraw_all()
