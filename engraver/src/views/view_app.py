from __future__ import annotations
import json
from pathlib import Path
import sys
import threading
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
from views.view_spinner import Spinner


class AppView(tk.Tk):
    def __init__(self):
        super().__init__()
        self.spinner: tk.Toplevel | None = None
        self.title("Polygon Engraver")
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

        # Shared state
        self.model: Optional[GeometryInt] = None
        self.source_label_var = tk.StringVar(value="No file loaded")
        self.source_path: Optional[str] = None

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
        self.add_view("2D")

        self.maximize()

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
        for i in range(self.notebook.index("end")):
            widget = self.notebook.nametowidget(self.notebook.tabs()[i])
            if isinstance(widget, View3D):
                widget.fit_to_view_pending = True
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
        self.spinner.overrideredirect(True)

        frame = tk.Frame(self.spinner, bg="#B0E0F3", padx=15, pady=15)
        frame.pack(fill="both", expand=True)

        style = ttk.Style(self.spinner)
        style.configure(
            "Blue.Horizontal.Bar",
            troughcolor=frame.cget("bg"),  # background area
            background="#B0E0F3",  # moving bar color
        )

        ttk.Label(
            frame,
            text=message,
            background=frame.cget("bg"),
            font=("Helvetica", 24, "bold"),
        ).pack(padx=0, pady=0)

        sp = Spinner(frame, size=64, thickness=6, color="#00428d", bg=frame.cget("bg"))
        sp.pack()
        sp.start()
        self._center_in_parent(self.spinner)

    def open_svg_async(self, path: str):
        self._show_spinner("Processing…")
        threading.Thread(
            target=self._load_svg_worker, args=(path,), daemon=True
        ).start()

    def _load_svg_worker(self, path: str):
        try:
            geom = SvgConverter.svg_to_geometry_int(path, scale=10000, tol=0.25)
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

    def centre_to_origin(self):
        if not self.model:
            messagebox.showinfo("Centre to Origin", "No geometry loaded.")
            return
        minx, miny, maxx, maxy = self.model.bounds()

        if minx == 0 and miny == 0:
            return

        midx = minx + (maxx - minx) / 2.0
        midy = miny + (maxy - miny) / 2.0

        new_polylines: List[PolylineInt] = []

        for pl in self.model.polylines:
            moved = [PointInt(p.x - midx, p.y - midy) for p in pl.points]
            new_polylines.append(
                PolylineInt(moved, simplify_tolerance=pl.simplify_tolerance)
            )
        new_points = [PointInt(p.x - midx, p.y - midy) for p in self.model.points]
        self.model = GeometryInt(new_polylines, new_points, self.model.scale)
        self.menubar.files_dirty = True
        self.redraw_all()

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
            lines.append(f'  <path d="{d}" fill="none" stroke="#000" />')
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
