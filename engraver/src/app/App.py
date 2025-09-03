from __future__ import annotations
import json
import math
from pathlib import Path
from typing import List, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from FileBrowser import FileBrowser
from geometry.PointInt import PointInt
from geometry.PolylineInt import PolylineInt
from geometry.GeometryInt import GeometryInt
from svg.SvgConverter import SvgConverter
from view.TopDownView import TopDownView
from view.View3D import View3D


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Polyline Viewer: Tk IDE-ish")
        self.geometry("1200x800")
        self.minsize(900, 600)
        try:
            self.tk.call("tk", "scaling", 1.2)  # slightly larger UI if supported
        except Exception:
            pass

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Shared state
        self.model: Optional[GeometryInt] = None
        self.source_label_var = tk.StringVar(value="No file loaded")

        # Layout: PanedWindow with left browser and right notebook
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, width=280)
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        # Left: file browser
        self.browser = FileBrowser(left, self)
        self.browser.pack(fill=tk.BOTH, expand=True)

        # Right: toolbar + notebook
        topbar = ttk.Frame(right)
        topbar.pack(fill=tk.X)
        ttk.Label(topbar, textvariable=self.source_label_var).pack(side=tk.LEFT, padx=8, pady=6)
        ttk.Button(topbar, text="Open File", command=self.open_file_dialog).pack(side=tk.RIGHT, padx=6)
        ttk.Button(topbar, text="New 3D Tab", command=lambda: self.add_view("3D")).pack(side=tk.RIGHT)
        ttk.Button(topbar, text="New 2D Tab", command=lambda: self.add_view("2D")).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Default tabs
        frame = self.add_view("2D")
        # self.add_view("3D")

        # Load a tiny demo so it's not heartbreakingly empty
        self.load_demo_geometry()

        # We want view to fit on inital demo geometry
        self.after_idle(lambda: setattr(frame, "fit_to_view_pending", True))

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
                pts = [App._as_point(p) for p in pts_raw]
                if len(pts) >= 2:
                    pls.append(PolylineInt(pts))

        # Case 2: root-level "points": [ [ [x,y], ... ], [ ... ] ]
        elif isinstance(data.get("points"), list):
            for poly in data.get("points", []):
                if isinstance(poly, list):
                    pts = [App._as_point(p) for p in poly]
                    if len(pts) >= 2:
                        pls.append(PolylineInt(pts))

        return GeometryInt(pls, scale)

    # Geometry management
    def set_geometry(self, geom: GeometryInt, source: str = ""):
        self.model = geom
        self.source_label_var.set(source or "(in-memory geometry)")
        self.redraw_all()

    def redraw_all(self):
        for i in range(self.notebook.index("end")):
            widget = self.notebook.nametowidget(self.notebook.tabs()[i])
            if hasattr(widget, "redraw"):
                widget.redraw()

    def open_file_dialog(self):
        path = filedialog.askopenfilename(
            title="Open geometry JSON",
            filetypes=[("SVG Files", "*.svg"), ("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return

        try:
            ext = Path(path).suffix.lower()
            if ext == ".json":
                geom = App.load_geometry_from_json(Path(path))
            elif ext == ".svg":
                geom = SvgConverter.svg_to_geometry_int(path, scale=10000, tol=0.25)
            else:
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {path}:\n{e}")
            return
        self.set_geometry(geom, source=str(path))

    def add_view(self, kind: str):
        if kind == "2D":
            frame = TopDownView(self.notebook, self)
            title = "Top-down 2D"
        elif kind == "3D":
            frame = View3D(self.notebook, self)
            title = "3D Wireframe"
        else:
            raise ValueError(f"Unknown view kind: {kind}")

        self.notebook.add(frame, text=title)
        self.notebook.select(frame)
        self.notebook.update_idletasks()          # ensure geometry is computed
        frame.after_idle(frame.fit_to_view)        # run when widget is idle/mapped
        return frame

    def fit_current(self):
        cur = self.notebook.select()
        if not cur:
            return
        widget = self.nametowidget(cur)
        if hasattr(widget, "fit_to_view"):
            widget.fit_to_view()

    def load_demo_geometry(self):
        # Simple rectangle + diagonal in 1000-scale units
        pl1 = PolylineInt(points=[
            PointInt(0000, 0000), PointInt(8000, 0000), PointInt(8000, 8000),
            PointInt(6000, 8000), PointInt(6000, 2000), PointInt(2000, 2000),
            PointInt(2000, 8000), PointInt(0000, 8000), PointInt(0000, 0000)],
            simplify_tolerance=5)
        geom = GeometryInt(polylines=[pl1], scale=1000)
        self.set_geometry(geom, source="<demo>")
