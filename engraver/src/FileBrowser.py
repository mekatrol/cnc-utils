from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class FileBrowser(ttk.Frame):
    def __init__(self, master, app: "App"):
        super().__init__(master)
        self.app = app
        self.root_dir: Optional[Path] = None

        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Open Folder", command=self.choose_folder).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(toolbar, text="New 2D Tab", command=lambda: self.app.add_view("2D")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="New 3D Tab", command=lambda: self.app.add_view("3D")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fit", command=self.app.fit_current).pack(side=tk.LEFT, padx=2)

        self.tree = ttk.Treeview(self, columns=("name",), show="tree")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_open_selected)

    def choose_folder(self):
        chosen = filedialog.askdirectory()
        if not chosen:
            return
        self.root_dir = Path(chosen)
        self.populate()

    def populate(self):
        self.tree.delete(*self.tree.get_children(""))
        if not self.root_dir:
            return
        root_id = self.tree.insert("", tk.END, text=str(self.root_dir), open=True)
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Only show json files
            rel = Path(dirpath).relative_to(self.root_dir)
            parent_id = root_id if rel == Path('.') else self._ensure_parents(root_id, rel)
            for name in sorted(filenames):
                if name.lower().endswith(".json"):
                    full = Path(dirpath) / name
                    self.tree.insert(parent_id, tk.END, text=name, values=(str(full),))

    def _ensure_parents(self, root_id, rel: Path):
        # Ensure intermediate directory nodes exist
        current_id = root_id
        built_path = Path()
        for part in rel.parts:
            built_path = built_path / part
            # Find child named part
            found = None
            for child in self.tree.get_children(current_id):
                if self.tree.item(child, "text") == part:
                    found = child
                    break
            if not found:
                found = self.tree.insert(current_id, tk.END, text=part, open=True)
            current_id = found
        return current_id

    def on_open_selected(self, event=None):
        item = self.tree.focus()
        if not item:
            return
        # Files store full path in values[0]
        vals = self.tree.item(item, "values")
        if not vals:
            return
        path = Path(vals[0])
        try:
            geom = load_geometry_from_json(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {path}:\n{e}")
            return
        self.app.set_geometry(geom, source=str(path))
