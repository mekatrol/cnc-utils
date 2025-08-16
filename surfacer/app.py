"""
grbl_surfacers.py

A cross-platform (Windows/Linux) Python 3.12+ GUI app to generate GRBL-compatible surfacing G-code
and preview the 2D/3D toolpath, including cutting moves and rapid (safe Z) movements.

Dependencies:
- numpy
- matplotlib
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List
import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - ensures 3D support is loaded

matplotlib.use("TkAgg")


def frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    v = start
    eps = step * 1e-8
    while v <= stop + eps:
        vals.append(round(v, 10))
        v += step
    return vals


class SurfacerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("GRBL Surfacer")

        frm = ttk.Frame(root, padding=8)
        frm.grid(row=0, column=0, sticky="nsew")

        self.width = tk.DoubleVar(value=100.0)
        self.length = tk.DoubleVar(value=100.0)
        self.bit_dia = tk.DoubleVar(value=25.4)
        self.step_over = tk.DoubleVar(value=25.4 / 2)
        self.step_down = tk.DoubleVar(value=0.1)
        self.layers = tk.IntVar(value=1)
        self.safe_z = tk.DoubleVar(value=10.0)
        self.cut_feed = tk.DoubleVar(value=1000.0)
        self.plunge_feed = tk.DoubleVar(value=300.0)
        self.start_x = tk.DoubleVar(value=0.0)
        self.start_y = tk.DoubleVar(value=0.0)

        row = 0

        def add_label_entry(label, var):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(frm, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=4, pady=2)
            row += 1

        add_label_entry("Width (mm)", self.width)
        add_label_entry("Length (mm)", self.length)
        add_label_entry("Bit diameter (mm)", self.bit_dia)
        add_label_entry("Step over (mm)", self.step_over)
        add_label_entry("Step down (mm)", self.step_down)
        add_label_entry("Number of layers", self.layers)
        add_label_entry("Safe Z (mm)", self.safe_z)
        add_label_entry("Cut feed (mm/min)", self.cut_feed)
        add_label_entry("Plunge feed (mm/min)", self.plunge_feed)
        add_label_entry("Start X (mm)", self.start_x)
        add_label_entry("Start Y (mm)", self.start_y)

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=8)
        ttk.Button(btn_frame, text="Preview 3D", command=self.preview3d).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="Preview 2D", command=self.preview2d).grid(row=0, column=1, padx=4)
        ttk.Button(btn_frame, text="Export G-code", command=self.export_gcode).grid(row=0, column=2, padx=4)
        ttk.Button(btn_frame, text="Quit", command=root.quit).grid(row=0, column=3, padx=4)

        fig = Figure(figsize=(6, 6))
        self.ax = fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

    def compute_toolpaths(self):
        W = self.width.get()
        L = self.length.get()
        # bit = self.bit_dia.get()  # currently unused but kept for future logic
        step = self.step_over.get()
        step_down = self.step_down.get()
        layers = self.layers.get()

        y_positions = frange(0.0, L, step)
        if y_positions[-1] < L - 1e-6:
            y_positions.append(L)

        toolpaths = []
        for layer in range(layers):
            depth = (layer + 1) * step_down
            for i, y in enumerate(y_positions):
                if i % 2 == 0:
                    xs = [0.0, W]
                else:
                    xs = [W, 0.0]
                ys = [y, y]
                toolpaths.append((xs, ys, depth))
        return toolpaths

    def _apply_equal_mm_scale(self):
        """
        Make 1 mm look the same in X, Y, and Z using set_box_aspect
        WITHOUT changing the current limits (so your custom Z padding stays).
        """
        try:
            x0, x1 = self.ax.get_xlim3d()
            y0, y1 = self.ax.get_ylim3d()
            z0, z1 = self.ax.get_zlim3d()
            xr = abs(x1 - x0)
            yr = abs(y1 - y0)
            zr = abs(z1 - z0)
            # Match the box to the data ranges so a unit in each axis renders equally.
            self.ax.set_box_aspect((xr, yr, zr))
        except Exception:
            # Older Matplotlib: silently skip if not available.
            pass

    def preview3d(self):
        try:
            tps = self.compute_toolpaths()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Rebuild as a 3D axes
        self.ax.clear()
        self.ax.remove()
        self.ax = self.canvas.figure.add_subplot(111, projection="3d")

        W = self.width.get()
        L = self.length.get()
        safe_z = self.safe_z.get()

        # draw stock boundary (at Z=0 plane)
        self.ax.plot([0, W, W, 0, 0], [0, 0, L, L, 0], [0, 0, 0, 0, 0], "k-")

        prev_end = None
        prev_depth = None
        for xs, ys, d in tps:
            depth = -abs(d)
            if prev_end is not None:
                # rapid move up to safe_z, then over, then plunge
                self.ax.plot([prev_end[0], prev_end[0]],
                             [prev_end[1], prev_end[1]],
                             [prev_depth,  safe_z], "r--", alpha=0.5)
                self.ax.plot([prev_end[0], xs[0]],
                             [prev_end[1], ys[0]],
                             [safe_z,      safe_z], "r--", alpha=0.5)
                self.ax.plot([xs[0], xs[0]],
                             [ys[0], ys[0]],
                             [safe_z, depth], "r--", alpha=0.5)
            else:
                # first plunge only
                self.ax.plot([xs[0], xs[0]],
                             [ys[0], ys[0]],
                             [safe_z, depth], "r--", alpha=0.5)

            # cutting move at depth
            self.ax.plot(xs, ys, [depth] * len(xs), "b-")
            prev_end = (xs[-1], ys[-1])
            prev_depth = depth

        # Compute Z bounds from deepest cut to safe height, then pad ±10 mm
        z_min = -abs(self.layers.get() * self.step_down.get())
        z_max = safe_z
        self.ax.set_xlim(-W * 0.05, W * 1.05)
        self.ax.set_ylim(-L * 0.05, L * 1.05)
        self.ax.set_zlim(z_min - 10, z_max + 10)

        # Make mm equal in X, Y, Z without changing limits
        self._apply_equal_mm_scale()

        # Nice default view
        self.ax.view_init(elev=25, azim=-60)

        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        self.ax.set_title("3D Toolpath preview (blue=cut, red=rapid)")

        self.canvas.draw()

    def preview2d(self):
        try:
            tps = self.compute_toolpaths()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.ax.clear()
        W = self.width.get()
        L = self.length.get()

        self.ax.plot([0, W, W, 0, 0], [0, 0, L, L, 0], "k-")

        prev_end = None
        for xs, ys, d in tps:
            if prev_end is not None:
                self.ax.plot([prev_end[0], xs[0]], [prev_end[1], ys[0]], "r--", alpha=0.5)
            self.ax.plot(xs, ys, "b-")
            prev_end = (xs[-1], ys[-1])

        self.ax.set_xlim(-W * 0.05, W * 1.05)
        self.ax.set_ylim(-L * 0.05, L * 1.05)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_title("2D Toolpath preview (blue=cut, red=rapid @ safe Z)")
        self.canvas.draw()

    def export_gcode(self):
        try:
            tps = self.compute_toolpaths()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        startx = self.start_x.get()
        starty = self.start_y.get()
        safe_z = self.safe_z.get()
        cut_feed = self.cut_feed.get()
        plunge_feed = self.plunge_feed.get()

        fn = filedialog.asksaveasfilename(
            defaultextension=".gcode", filetypes=[("G-code", "*.gcode")]
        )
        if not fn:
            return

        with open(fn, "w") as fh:
            fh.write("(GRBL Surfacer)\nG21\nG90\nG94\n")
            fh.write(f"G0 Z{safe_z:.3f}\n")

            prev_end = None
            for xs, ys, depth in tps:
                if prev_end is not None:
                    fh.write(f"G0 Z{safe_z:.3f}\n")
                    fh.write(f"G0 X{startx+xs[0]:.3f} Y{starty+ys[0]:.3f}\n")
                else:
                    fh.write(f"G0 X{startx+xs[0]:.3f} Y{starty+ys[0]:.3f}\n")
                fh.write(f"G1 Z{-abs(depth):.3f} F{plunge_feed:.0f}\n")
                for x, y in zip(xs, ys):
                    fh.write(f"G1 X{startx+x:.3f} Y{starty+y:.3f} F{cut_feed:.0f}\n")
                prev_end = (xs[-1], ys[-1])
            fh.write(f"G0 Z{safe_z:.3f}\nM2\n")
        messagebox.showinfo("Done", f"G-code exported to {fn}")


def main():
    root = tk.Tk()
    app = SurfacerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
