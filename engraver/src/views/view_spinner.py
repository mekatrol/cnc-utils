import tkinter as tk

from views.view_constants import SPINNER_DEFAULT_COLOR


class Spinner(tk.Canvas):
    def __init__(
        self,
        master,
        size=64,
        thickness=6,
        color=SPINNER_DEFAULT_COLOR,
        speed=6,
        **kw,
    ):
        super().__init__(
            master, width=size, height=size, highlightthickness=0, bg=kw.get("bg")
        )
        self.size, self.thickness, self.color, self.speed = (
            size,
            thickness,
            color,
            speed,
        )
        m = thickness // 2 + 2
        self._arc = self.create_arc(
            m,
            m,
            size - m,
            size - m,
            start=0,
            extent=90,
            style="arc",
            outline=color,
            width=thickness,
        )
        self._angle = 0
        self._job = None

    def start(self):
        if self._job:
            return
        self._tick()

    def stop(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None

    def _tick(self):
        self._angle = (self._angle + self.speed) % 360
        self.itemconfigure(self._arc, start=self._angle)
        self._job = self.after(16, self._tick)  # ~60 FPS
