from __future__ import annotations
import sys
from tkinter import messagebox
from typing import TYPE_CHECKING
import tkinter as tk

if TYPE_CHECKING:
    # Import only for type checking; does not run at runtime
    from views.view_app import AppView  # adjust module path


class Menubar(tk.Frame):
    def __init__(self, root: "AppView"):
        self.root = root

        super().__init__(root)

        self.main_frame = self
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # No files are dirty by default
        self.files_dirty = False

        # Disable menu tear off
        self.root.option_add("*tearOff", tk.FALSE)

        self.create_menubar()

    def create_menubar(self) -> None:
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.file_menu = tk.Menu(self.menubar)
        self.edit_menu = tk.Menu(self.menubar)
        self.menubar.add_cascade(menu=self.file_menu, label="File")
        self.menubar.add_cascade(menu=self.edit_menu, label="Edit")

        self.add_file_menu_items()

    def add_file_menu_items(self):
        self.file_menu.add_command(
            label="Open",
            accelerator="Ctrl-O",
            state=tk.ACTIVE,
            command=self.root.open_file_dialog,
        )

        if sys.platform == "darwin":  # macOS
            accel = "Command-X"  # ⌘X
            bind = "<Command-x>"
        else:  # Windows, Linux
            accel = "Alt-X"
            bind = "<Alt-x>"

        # in your menu
        self.file_menu.add_command(
            label="Exit", accelerator=accel, command=self.on_exit
        )

        # bind actual key sequence
        self.root.bind_all(bind, lambda e: self.on_exit())

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
        self.root.destroy()
