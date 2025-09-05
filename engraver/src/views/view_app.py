from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import List, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from geometry.PointInt import PointInt
from geometry.PolylineInt import PolylineInt
from geometry.GeometryInt import GeometryInt
from svg.SvgConverter import SvgConverter

from views.view_2d import View2D
from views.view_3d import View3D
from views.menubar import Menubar


class AppView(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Polygon Engraver")
        self.geometry("1200x800")
        self.minsize(900, 600)
        try:
            self.tk.call("tk", "scaling", 1.2)  # slightly larger UI if supported
        except Exception:
            pass

        # Shared state
        self.model: Optional[GeometryInt] = None
        self.source_label_var = tk.StringVar(value="No file loaded")

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Create main menubar
        self.menubar = Menubar(self)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Default tabs
        self.add_view("3D")
        frame = self.add_view("2D")

        self.maximize()

        # Load a tiny demo so it's not heartbreakingly empty
        self.load_demo_geometry()

        # We want view to fit on inital demo geometry
        self.after_idle(lambda: setattr(frame, "fit_to_view_pending", True))

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
        frame.after_idle(frame.fit_to_view)  # run when widget is idle/mapped
        return frame

    def fit_current(self):
        cur = self.notebook.select()
        if not cur:
            return
        widget = self.nametowidget(cur)
        if hasattr(widget, "fit_to_view"):
            widget.fit_to_view()

    def reset_current(self):
        cur = self.notebook.select()
        if not cur:
            return
        widget = self.nametowidget(cur)
        if hasattr(widget, "reset_view"):
            widget.reset_view()

    def load_demo_geometry(self):
        # Simple rectangle + diagonal in 1000-scale units
        polyline = PolylineInt(
            points=[
                PointInt(0000, 0000),
                PointInt(8000, 0000),
                PointInt(8000, 8000),
                PointInt(6000, 8000),
                PointInt(6000, 2000),
                PointInt(2000, 2000),
                PointInt(2000, 8000),
                PointInt(0000, 8000),
                PointInt(0000, 0000),
            ],
            simplify_tolerance=5,
        )

        points = [
            PointInt(1000, 4000),
            PointInt(4000, 2000),
            PointInt(2000, 2000),
            PointInt(4000, 6000),
            PointInt(9000, 4000),
            PointInt(-2000, 1000),
        ]

        geom = GeometryInt(polylines=[polyline], points=points, scale=1000)
        self.set_geometry(geom, source="<demo>")
