"""
grbl_surfacers.py

A cross-platform (Windows/Linux) Python 3.12+ GUI app to generate GRBL-compatible surfacing G-code
and preview the 2D toolpath, including cutting moves and rapid (safe Z) movements.

Dependencies:
- numpy
- matplotlib

"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Tuple
import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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
        self.bit_dia = tk.DoubleVar(value=6.0)
        self.step_over = tk.DoubleVar(value=3.0)
        self.step_down = tk.DoubleVar(value=1.0)
        self.layers = tk.IntVar(value=1)
        self.safe_z = tk.DoubleVar(value=5.0)
        self.cut_feed = tk.DoubleVar(value=800.0)
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
        ttk.Button(btn_frame, text="Preview Toolpath", command=self.preview).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="Export G-code", command=self.export_gcode).grid(row=0, column=1, padx=4)
        ttk.Button(btn_frame, text="Quit", command=root.quit).grid(row=0, column=2, padx=4)

        fig = Figure(figsize=(6, 6))
        self.ax = fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

    def compute_toolpaths(self):
        W = self.width.get()
        L = self.length.get()
        bit = self.bit_dia.get()
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

    def preview(self):
        try:
            tps = self.compute_toolpaths()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.ax.clear()
        W = self.width.get()
        L = self.length.get()
        safe_z = self.safe_z.get()

        self.ax.plot([0, W, W, 0, 0], [0, 0, L, L, 0], 'k-')

        prev_end = None
        for xs, ys, d in tps:
            if prev_end is not None:
                self.ax.plot([prev_end[0], xs[0]], [prev_end[1], ys[0]], 'r--', alpha=0.5)
            self.ax.plot(xs, ys, 'b-')
            prev_end = (xs[-1], ys[-1])

        self.ax.set_xlim(-W*0.05, W*1.05)
        self.ax.set_ylim(-L*0.05, L*1.05)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_title('Toolpath preview (blue=cut, red=rapid @ safe Z)')
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

        fn = filedialog.asksaveasfilename(defaultextension='.gcode', filetypes=[('G-code', '*.gcode')])
        if not fn:
            return

        with open(fn, 'w') as fh:
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

if __name__ == '__main__':
    main()
