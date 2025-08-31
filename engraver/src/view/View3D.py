from __future__ import annotations
from typing import TYPE_CHECKING
import math
from typing import Tuple
from src.geometry.GeoUtil import GeoUtil
from view.colors import COLORS
from view.BaseView import BaseView

if TYPE_CHECKING:
    # Import only for type checking; does not run at runtime
    from app.App import App  # adjust module path


class View3D(BaseView):
    def __init__(self, master, app: "App"):
        super().__init__(master, app)
        # Camera/world params
        self.yaw = math.radians(35)
        self.pitch = math.radians(25)
        self.distance = 10.0  # camera distance along +Z
        self.zoom = 50.0      # pixel scale (screen pixels per world unit at z≈0)
        self.pan = [0.0, 0.0]  # screen-space pan in pixels

        # Mouse state
        self._rotating = False
        self._panning = False
        self._last = (0, 0)

        # Bindings
        self.canvas.bind("<ButtonPress-1>", self._on_press_left)
        self.canvas.bind("<ButtonPress-3>", self._on_press_right)
        self.canvas.bind("<B1-Motion>", self._on_drag_left)
        self.canvas.bind("<B3-Motion>", self._on_drag_right)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1))   # Linux
        self.canvas.bind("<Button-5>", lambda e: self._zoom(-1))  # Linux
        self.canvas.bind_all("f", lambda e: self.fit_to_view())
        self.canvas.bind_all("F", lambda e: self.fit_to_view())
        self.canvas.bind_all("r", lambda e: self.reset_view())
        self.canvas.bind_all("R", lambda e: self.reset_view())

    def reset_view(self):
        self.yaw = math.radians(35)
        self.pitch = math.radians(25)
        self.distance = 10.0
        self.zoom = 50.0
        self.pan = [0.0, 0.0]
        self.redraw()

    def fit_to_view(self):
        # Compute bounding box in world units
        minx, miny, maxx, maxy = GeoUtil.world_bounds(self.app.model)
        dx = maxx - minx or 1.0
        dy = maxy - miny or 1.0
        size = max(dx, dy)
        # Adjust camera distance so model fits nicely
        self.distance = size * 1.5 if size > 0 else 10.0
        # Center the model on screen (pivot at screen center)
        self.pan = [0.0, 0.0]
        self.yaw = math.radians(35)
        self.pitch = math.radians(25)

        self.redraw()

    # Events
    def _on_press_left(self, event):
        self._rotating = True
        self._last = (event.x, event.y)

    def _on_press_right(self, event):
        self._panning = True
        self._last = (event.x, event.y)

    def _on_drag_left(self, event):
        if not self._rotating:
            return
        dx = event.x - self._last[0]
        dy = event.y - self._last[1]
        self.yaw += dx * 0.01
        self.pitch += dy * 0.01
        # Clamp pitch to avoid flipping
        self.pitch = max(-math.pi/2 + 0.05, min(math.pi/2 - 0.05, self.pitch))
        self._last = (event.x, event.y)
        self.redraw()

    def _on_drag_right(self, event):
        if not self._panning:
            return
        dx = event.x - self._last[0]
        dy = event.y - self._last[1]
        self.pan[0] += dx
        self.pan[1] += dy
        self._last = (event.x, event.y)
        self.redraw()

    def _on_release(self, event):
        self._rotating = False
        self._panning = False

    def _on_wheel(self, event):
        self._zoom(1 if event.delta > 0 else -1)

    def _zoom(self, direction):
        factor = 0.9 if direction > 0 else 1/0.9
        self.distance *= factor
        self.distance = max(0.1, min(1e6, self.distance))
        self.redraw()

    # Math helpers
    def _project_point(self, x: float, y: float, z: float, w: int, h: int) -> Tuple[float, float, float]:
        """
        Project a point already expressed RELATIVE TO THE PIVOT (model center)
        using yaw/pitch rotations, simple perspective, then screen-space pan.
        """
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)

        # Rotate: yaw (Y axis) then pitch (X axis)
        x1 = x * cy + z * sy
        z1 = -x * sy + z * cy
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp

        camz = self.distance
        denom = camz + z2
        if denom <= 1e-6:
            denom = 1e-6
        scale = self.zoom * (camz / denom)

        xs = x1 * scale + w * 0.5 + self.pan[0]
        ys = -y2 * scale + h * 0.5 + self.pan[1]
        return xs, ys, z2

    # Drawing
    def redraw(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()

        # Compute pivot at model center in world units (bbox center)
        minx, miny, maxx, maxy = GeoUtil.world_bounds(self.app.model)
        cx = (minx + maxx) * 0.5
        cy = (miny + maxy) * 0.5

        # Grid plane (drawn around pivot)
        self._draw_grid_3d(w, h)

        g = self.app.model
        if not g or not g.polylines:
            c.create_text(w // 2, h // 2, text="No geometry loaded", fill="#888")
            return
        s = g.scale if g.scale else 1

        # Draw lines RELATIVE to pivot so rotations happen about model center
        for i, pl in enumerate(g.polylines):
            if len(pl.pts) < 2:
                continue
            last_pt = None
            for p in pl.pts:
                x_rel = p.x / s - cx
                y_rel = p.y / s - cy
                xs, ys, _ = self._project_point(x_rel, y_rel, 0.0, w, h)
                if last_pt is not None:
                    color = COLORS[i % len(COLORS)]
                    c.create_line(last_pt[0], last_pt[1], xs, ys, fill=color, width=1.5)
                last_pt = (xs, ys)

    def _draw_grid_3d(self, w: int, h: int):
        # Simple world XY grid centered on the pivot (origin after centering)
        c = self.canvas
        c.create_rectangle(0, 0, w, h, fill="#111", outline="")

        minx, miny, maxx, maxy = GeoUtil.world_bounds(self.app.model)
        dx = max(maxx - minx, 1.0)
        dy = max(maxy - miny, 1.0)
        size = max(dx, dy)
        half = max(5.0, size)
        step = self._nice_step(size / 10.0)

        # Draw grid lines around origin (pivot)
        x = -half
        while x <= half:
            p1 = self._project_point(x, -half, 0.0, w, h)
            p2 = self._project_point(x,  half, 0.0, w, h)
            c.create_line(p1[0], p1[1], p2[0], p2[1], fill="#1a1a1a")
            x += step

        y = -half
        while y <= half:
            p1 = self._project_point(-half, y, 0.0, w, h)
            p2 = self._project_point(half, y, 0.0, w, h)
            c.create_line(p1[0], p1[1], p2[0], p2[1], fill="#1a1a1a")
            y += step

        # Axes at the pivot
        ox, oy, _ = self._project_point(0.0, 0.0, 0.0, w, h)
        xx, xy, _ = self._project_point(step, 0.0, 0.0, w, h)
        yx, yy, _ = self._project_point(0.0, step, 0.0, w, h)
        c.create_line(ox, oy, xx, xy, fill="#444", width=2)  # X
        c.create_line(ox, oy, yx, yy, fill="#444", width=2)  # Y

    @staticmethod
    def _nice_step(raw: float) -> float:
        if raw <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(raw))
        for m in (1, 2, 5, 10):
            if raw <= m * mag:
                return m * mag
        return 10 * mag
