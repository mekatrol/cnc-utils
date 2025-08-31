import tkinter as tk
from tkinter import ttk


class BaseView(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master)
        self.app = app
        self.canvas = tk.Canvas(self, background="#111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Expose>", lambda e: self.redraw())
        # Common mouse bindings are set per subclass

    def _on_resize(self, event):
        self.redraw()

    def redraw(self):
        raise NotImplementedError
