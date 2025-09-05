from __future__ import annotations
import sys
from tkinter import messagebox
from typing import TYPE_CHECKING
import tkinter as tk

if TYPE_CHECKING:
    # Import only for type checking; does not run at runtime
    from views.view_app import AppView  # adjust module path


class Menubar(tk.Frame):
    def __init__(self, app: "AppView"):
        self.app = app

        super().__init__(app)

        self.pack(fill=tk.BOTH, expand=True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # No files are dirty by default
        self.files_dirty = False

        # Disable menu tear off
        self.app.option_add("*tearOff", tk.FALSE)

        self.create_menubar()

    def create_menubar(self) -> None:
        self.menubar = tk.Menu(self.app)
        self.app.config(menu=self.menubar)

        self.file_menu = tk.Menu(self.menubar)
        self.view_menu = tk.Menu(self.menubar)
        self.menubar.add_cascade(menu=self.file_menu, label="File")
        self.menubar.add_cascade(menu=self.view_menu, label="View")

        self.add_file_menu_items()
        self.add_view_menu_items()

    def add_view_menu_items(self):
        self.view_menu.add_command(
            label="Fit",
            command=self.app.fit_current,
        )

        self.view_menu.add_command(
            label="Reset",
            command=self.app.reset_current,
        )

    def add_file_menu_items(self):
        self.file_menu.add_command(
            label="Open",
            accelerator="Ctrl-O",
            state=tk.ACTIVE,
            command=self.app.open_file_dialog,
        )

        self.file_menu.add_separator()

        if sys.platform == "darwin":  # macOS
            accel = "Command-X"  # ⌘X
            bind = "<Command-x>"
        else:  # Windows, Linux
            accel = "Alt-X"
            bind = "<Alt-x>"

        self.file_menu.add_command(
            label="Exit", accelerator=accel, command=self.on_exit
        )

        self.app.bind_all(bind, lambda e: self.on_exit())

    def save_file(self):
        self.files_dirty = False

    def on_exit(self):
        if self.files_dirty:
            ans = messagebox.askyesnocancel(
                "Unsaved changes", "Save changes before exit?"
            )
            if ans is None:
                # Cancel selected
                return
            if ans:
                # Save before exit
                self.save_file()

        # Exit app
        self.app.destroy()
