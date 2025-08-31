from __future__ import annotations

import sys
import tkinter as tk
from app.App import App


def maximize(win: tk.Tk):
    win.update_idletasks()
    # 1) Try native “zoomed” (Windows, many X11)
    try:
        win.state("zoomed")
        return
    except tk.TclError:
        pass
    # 2) Some X11 WMs expose -zoomed
    try:
        win.attributes("-zoomed", True)
        return
    except tk.TclError:
        pass
    # 3) macOS fallback (fullscreen) or generic geometry fill
    if sys.platform == "darwin":
        win.attributes("-fullscreen", True)  # Esc to exit if you add a binding
    else:
        win.geometry(f"{win.winfo_screenwidth()}x{win.winfo_screenheight()}+0+0")


def main():
    app = App()
    maximize(app)
    app.mainloop()


if __name__ == "__main__":
    main()
