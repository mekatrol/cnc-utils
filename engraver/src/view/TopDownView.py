

from geometry.GeometryUtils import GeometryUtils
from view.BaseView import BaseView


class TopDownView(BaseView):
    def __init__(self, master, app: "App"):
        super().__init__(master, app)
        self.zoom = 1.0  # pixels per world unit
        self.offset = [0.0, 0.0]  # screen offset in pixels
        self.grid_step = 10.0  # world units
        # Mouse state
        self._dragging = False
        self._drag_last = (0, 0)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        # Linux/Mac wheel events
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e, -1))
        self.canvas.bind_all("f", lambda e: self.fit_to_view())
        self.canvas.bind_all("F", lambda e: self.fit_to_view())
        self.canvas.bind_all("r", lambda e: self.reset_view())
        self.canvas.bind_all("R", lambda e: self.reset_view())

    # View control
    def fit_to_view(self):
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        minx, miny, maxx, maxy = GeometryUtils.world_bounds(self.app.model)
        dx = maxx - minx or 1.0
        dy = maxy - miny or 1.0
        # Leave a little padding
        scale_x = (w * 0.9) / dx
        scale_y = (h * 0.9) / dy
        self.zoom = min(scale_x, scale_y)
        # Center
        cx = (minx + maxx) * 0.5
        cy = (miny + maxy) * 0.5
        self.offset = [w * 0.5 - cx * self.zoom, h * 0.5 + cy * self.zoom]
        self.redraw()

    def reset_view(self):
        self.zoom = 1.0
        self.offset = [self.canvas.winfo_width() * 0.5, self.canvas.winfo_height() * 0.5]
        self.redraw()

    # Events
    def _on_press(self, event):
        self._dragging = True
        self._drag_last = (event.x, event.y)

    def _on_drag(self, event):
        if not self._dragging:
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self.offset[0] += dx
        self.offset[1] += dy
        self._drag_last = (event.x, event.y)
        self.redraw()

    def _on_release(self, event):
        self._dragging = False

    def _on_wheel(self, event):
        direction = 1 if event.delta > 0 else -1
        self._zoom_at(event, direction)

    def _zoom_at(self, event, direction):
        factor = 1.1 if direction > 0 else 1.0 / 1.1
        # Zoom around cursor
        x, y = event.x, event.y
        old_zoom = self.zoom
        self.zoom *= factor
        # Adjust offset so the point under the cursor stays put
        self.offset[0] = x - (x - self.offset[0]) * (self.zoom / old_zoom)
        self.offset[1] = y - (y - self.offset[1]) * (self.zoom / old_zoom)
        self.redraw()

    # Drawing
    def redraw(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        # Grid
        self._draw_grid(w, h)
        # Geometry
        g = self.app.model
        if not g or not g.polylines:
            self._draw_center_cross(w, h)
            return
        s = g.scale if g.scale else 1
        for pl in g.polylines:
            if len(pl.pts) < 2:
                continue
            coords = []
            last = None
            for p in pl.pts:
                xw, yw = p.x / s, p.y / s
                xs = xw * self.zoom + self.offset[0]
                ys = -yw * self.zoom + self.offset[1]
                if last is not None:
                    coords.extend([last[0], last[1], xs, ys])
                last = (xs, ys)
            # Batch draw this polyline as many segments
            if coords:
                c.create_line(*coords, fill="#8fd3ff", width=1.5)

    def _draw_grid(self, w: int, h: int):
        c = self.canvas
        # Pick grid spacing in pixels based on zoom
        world_step = self.grid_step
        px_step = max(25, world_step * self.zoom)
        # Convert to nearest nice pixel step
        # Keep it simple: use px_step as-is
        # Find world origin on screen
        # Draw axes
        c.create_line(0, self.offset[1], w, self.offset[1], fill="#333")
        c.create_line(self.offset[0], 0, self.offset[0], h, fill="#333")
        # Light grid lines
        # Horizontal
        y = self.offset[1] % px_step
        while y < h:
            c.create_line(0, y, w, y, fill="#1a1a1a")
            y += px_step
        # Vertical
        x = self.offset[0] % px_step
        while x < w:
            c.create_line(x, 0, x, h, fill="#1a1a1a")
            x += px_step

    def _draw_center_cross(self, w: int, h: int):
        c = self.canvas
        c.create_text(w // 2, h // 2, text="No geometry loaded", fill="#888")
        c.create_line(w//2 - 10, h//2, w//2 + 10, h//2, fill="#444")
        c.create_line(w//2, h//2 - 10, w//2, h//2 + 10, fill="#444")
