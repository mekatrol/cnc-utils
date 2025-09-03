from __future__ import annotations
from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk

if TYPE_CHECKING:
    # Import only for type checking; does not run at runtime
    from app.App import App  # adjust module path


class BaseView(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master)
        self.app = app
        self.canvas = tk.Canvas(self, background="#111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Expose>", lambda e: self.redraw())
        self.fit_to_view_pending = False

    def _on_resize(self, event):
        self.redraw()

    def redraw(self):
        raise NotImplementedError
