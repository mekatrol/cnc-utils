# views/menubar.py
from __future__ import annotations
import sys
import tkinter as tk
from tkinter import messagebox
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from views.view_app import AppView


class Menubar(tk.Menu):
    def __init__(self, app: "AppView"):
        super().__init__(app)
        self.app = app
        self.files_dirty = False

        app.option_add("*tearOff", False)

        # attach menubar
        app.config(menu=self)

        filem = tk.Menu(self)
        viewm = tk.Menu(self)
        self.add_cascade(label="File", menu=filem)
        self.add_cascade(label="View", menu=viewm)

        # Platform accelerators
        if sys.platform == "darwin":
            open_accel, open_bind = "Command-O", "<Command-o>"
            exit_accel, exit_bind = "Command-Q", "<Command-q>"
        else:
            open_accel, open_bind = "Ctrl-O", "<Control-o>"
            exit_accel, exit_bind = "Alt-X", "<Alt-x>"

        # File
        filem.add_command(
            label="Open", accelerator=open_accel, command=app.open_file_dialog
        )
        filem.add_command(label="Save SVG As...", command=app.save_svg_as)
        filem.add_separator()
        filem.add_command(label="Exit", accelerator=exit_accel, command=self.on_exit)

        # View
        viewm.add_command(label="Fit", command=app.fit_current)
        viewm.add_command(
            label="Fit Including Origin", command=app.fit_current_including_origin
        )

        # Key bindings
        app.bind_all(open_bind, lambda e: app.open_file_dialog())
        app.bind_all(exit_bind, lambda e: self.on_exit())

    def save_file(self):
        self.files_dirty = False

    def on_exit(self):
        if self.files_dirty:
            ans = messagebox.askyesnocancel(
                "Unsaved changes", "Save changes before exit?"
            )
            if ans is None:
                return
            if ans:
                self.save_file()
        self.app.destroy()
