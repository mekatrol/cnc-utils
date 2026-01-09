from __future__ import annotations
import math
from typing import TYPE_CHECKING
from colors import COLORS
from geometry.GeoUtil import GeoUtil
from geometry.PointInPolygonResult import PointInPolygonResult
from geometry.PointInt import PointInt
from views.view_base import BaseView
from views.view_constants import (
    AXIS_X_COLOR,
    AXIS_Y_COLOR,
    BACKGROUND_COLOR,
    EMPTY_TEXT_COLOR,
    FILL_COLOR_2D,
    GRID_AXIS_COLOR,
    GRID_CENTER_CROSS_COLOR,
    GRID_LINE_COLOR,
    HATCH_COLOR,
    ORIGIN_BALL_COLOR,
)

if TYPE_CHECKING:
    # Import only for type checking; does not run at runtime
    from views.view_app import AppView


class View2D(BaseView):
    def __init__(self, master, app: "AppView"):
        super().__init__(master, app)
        self.zoom = 1.0  # pixels per world unit
        self.offset = [0.0, 0.0]  # screen offset in pixels
        self.grid_step = 10.0  # world units
        
        # Mouse state
        self._dragging = False
        self._drag_last = (0, 0)
        self._drag_moved = False
        self._press_pos = (0, 0)
        
        # Selection state
        self._selected_polygon_solid_fill = True
        self._selected_polygons = []
        self.hatch_angle_deg = 45.0
        self.hatch_spacing_px = 8.0
        self.hatch_color = HATCH_COLOR
        self.fill_color = FILL_COLOR_2D
        self.fill_stipple = "gray12"

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
        self.fit_to_view_pending = False
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        minx, miny, maxx, maxy = GeoUtil.world_bounds(self.app.model)
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
        self.offset = [
            self.canvas.winfo_width() * 0.5,
            self.canvas.winfo_height() * 0.5,
        ]
        self.redraw()

    # Events
    def _on_press(self, event):
        self._dragging = True
        self._drag_last = (event.x, event.y)
        self._drag_moved = False
        self._press_pos = (event.x, event.y)

    def _on_drag(self, event):
        if not self._dragging:
            return
        if not self._drag_moved:
            dx0 = event.x - self._press_pos[0]
            dy0 = event.y - self._press_pos[1]
            if abs(dx0) + abs(dy0) < 3:
                return
            self._drag_moved = True
            self._drag_last = (event.x, event.y)
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self.offset[0] += dx
        self.offset[1] += dy
        self._drag_last = (event.x, event.y)
        self.redraw()

    def _on_release(self, event):
        self._dragging = False
        if not self._drag_moved:
            self._select_polygon(event.x, event.y)
            self.redraw()

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
        if self.fit_to_view_pending:
            self.fit_to_view()
            return

        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()
        # Grid
        self._draw_grid(w, h)
        self._draw_axis_gizmo(w, h)
        self._draw_selection()
        # Geometry
        g = self.app.model
        if not g or not g.polylines:
            self._draw_center_cross(w, h)
            return
        s = g.scale if g.scale else 1

        # Draw geometry polylines
        for i, polyline in enumerate(g.polylines):
            # Must be at least 2 points to have any line segments
            if len(polyline.points) < 2:
                continue

            coords = []
            last = None

            for poly_point in polyline.points:
                xw, yw = poly_point.x / s, poly_point.y / s
                xs = xw * self.zoom + self.offset[0]
                ys = -yw * self.zoom + self.offset[1]
                if last is not None:
                    coords.extend([last[0], last[1], xs, ys])
                last = (xs, ys)

            # Batch draw this polyline as many segments
            if coords:
                color = COLORS[i % len(COLORS)]
                c.create_line(*coords, fill=color, width=1.5)

        # Points
        for i, pt in enumerate(g.points):
            xw, yw = pt.x / s, pt.y / s
            xs = xw * self.zoom + self.offset[0]
            ys = -yw * self.zoom + self.offset[1]
            r = 3  # screen pixels
            color = COLORS[(i + len(COLORS) >> 1) % len(COLORS)]
            c.create_oval(xs - r, ys - r, xs + r, ys + r, outline=color, fill=color)

    def _draw_axis_gizmo(self, w: int, h: int) -> None:
        # Fixed-size axis arrows anchored at the world origin.
        ox = 0.0 * self.zoom + self.offset[0]
        oy = -0.0 * self.zoom + self.offset[1]
        axis_len = 45.0
        r = 5.0
        self.canvas.create_line(
            ox, oy, ox + axis_len, oy, fill=AXIS_X_COLOR, width=2, arrow="last"
        )
        self.canvas.create_line(
            ox, oy, ox, oy - axis_len, fill=AXIS_Y_COLOR, width=2, arrow="last"
        )
        self.canvas.create_oval(
            ox - r, oy - r, ox + r, oy + r, fill=ORIGIN_BALL_COLOR, outline=""
        )

    def _screen_to_world(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.offset[0]) / self.zoom, -(y - self.offset[1]) / self.zoom

    def _screen_to_pointint(self, x: float, y: float, scale: int) -> PointInt:
        xw, yw = self._screen_to_world(x, y)
        return PointInt(
            GeoUtil.float_to_int(xw, scale),
            GeoUtil.float_to_int(yw, scale),
        )

    def _collect_polygons(self):
        g = self.app.model
        if not g or not g.polylines:
            return []
        polygons = []
        for idx, polyline in enumerate(g.polylines):
            points = polyline.points
            if len(points) < 3:
                continue
            if len(points) >= 2 and points[0] == points[-1]:
                points = points[:-1]
            if len(points) < 3:
                continue
            polygons.append({"index": idx, "points": points})
        return polygons

    def _select_polygon(self, x: float, y: float) -> None:
        g = self.app.model
        if not g or not g.polylines:
            self._selected_polygons = []
            return

        query = self._screen_to_pointint(x, y, g.scale or 1)
        polygons = self._collect_polygons()
        containing = []
        for poly in polygons:
            result = GeoUtil.point_in_polygon(query, poly["points"])
            if result in (
                PointInPolygonResult.Inside,
                PointInPolygonResult.Edge,
                PointInPolygonResult.Vertex,
            ):
                containing.append(poly)

        if not containing:
            self._selected_polygons = []
            return

        selected = min(
            containing,
            key=lambda poly: abs(GeoUtil.area(poly["points"])),
        )

        holes = []
        for poly in polygons:
            if poly is selected:
                continue
            result = GeoUtil.point_in_polygon(poly["points"][0], selected["points"])
            if result == PointInPolygonResult.Inside:
                holes.append(poly)

        selected_entry = {"polygon": selected, "holes": holes}
        existing_idx = next(
            (idx for idx, entry in enumerate(self._selected_polygons)
             if entry["polygon"]["index"] == selected["index"]),
            None,
        )
        if existing_idx is None:
            self._selected_polygons.append(selected_entry)
        else:
            self._selected_polygons.pop(existing_idx)

    def _draw_selection(self) -> None:
        if not self._selected_polygons:
            return
        g = self.app.model
        if not g:
            return
        s = g.scale or 1
        c = self.canvas
        bg = c.cget("background") or BACKGROUND_COLOR

        def to_screen(points):
            coords = []
            for pt in points:
                xw, yw = pt.x / s, pt.y / s
                xs = xw * self.zoom + self.offset[0]
                ys = -yw * self.zoom + self.offset[1]
                coords.extend([xs, ys])
            return coords

        def to_screen_points(points):
            coords = []
            for pt in points:
                xw, yw = pt.x / s, pt.y / s
                xs = xw * self.zoom + self.offset[0]
                ys = -yw * self.zoom + self.offset[1]
                coords.append((xs, ys))
            return coords

        selected_indices = {
            entry["polygon"]["index"] for entry in self._selected_polygons
        }

        for entry in self._selected_polygons:
            selected_polygon = entry["polygon"]
            selected_holes = entry["holes"]

            if self._selected_polygon_solid_fill:
                coords = to_screen(selected_polygon["points"])
                if coords:
                    c.create_polygon(
                        *coords,
                        fill=self.fill_color,
                        outline="",
                        stipple=self.fill_stipple,
                    )

                for hole in selected_holes:
                    if hole["index"] in selected_indices:
                        continue
                    hole_coords = to_screen(hole["points"])
                    if hole_coords:
                        c.create_polygon(*hole_coords, fill=bg, outline="")
            else:
                polygon_points = to_screen_points(selected_polygon["points"])
                if polygon_points:
                    self._draw_hatch_polygon(polygon_points)

                for hole in selected_holes:
                    if hole["index"] in selected_indices:
                        continue
                    hole_points = to_screen_points(hole["points"])
                    if hole_points:
                        hole_coords = [coord for pt in hole_points for coord in pt]
                        c.create_polygon(*hole_coords, fill=bg, outline="")

    def _draw_hatch_polygon(self, polygon_points) -> None:
        if len(polygon_points) < 3:
            return
        self._draw_hatch_lines(polygon_points, self.hatch_angle_deg)
        self._draw_hatch_lines(polygon_points, self.hatch_angle_deg + 90.0)

    def _draw_hatch_lines(self, polygon_points, angle_deg: float) -> None:
        c = self.canvas
        angle = math.radians(angle_deg)
        dx = math.cos(angle)
        dy = math.sin(angle)
        nx = -dy
        ny = dx

        xs = [p[0] for p in polygon_points]
        ys = [p[1] for p in polygon_points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        corners = [
            (minx, miny),
            (minx, maxy),
            (maxx, miny),
            (maxx, maxy),
        ]
        offsets = [corner[0] * nx + corner[1] * ny for corner in corners]
        min_o, max_o = min(offsets), max(offsets)

        spacing = max(1.0, float(self.hatch_spacing_px))
        start = math.floor(min_o / spacing) * spacing
        end = math.ceil(max_o / spacing) * spacing

        for offset in self._frange(start, end, spacing):
            intersections = []
            for i in range(len(polygon_points)):
                p0 = polygon_points[i]
                p1 = polygon_points[(i + 1) % len(polygon_points)]
                hit = self._line_segment_intersection((dx, dy), (nx, ny), offset, p0, p1)
                if hit is not None:
                    intersections.append(hit)

            if len(intersections) < 2:
                continue

            intersections.sort(key=lambda item: item[0])
            deduped = []
            for t, pt in intersections:
                if not deduped or abs(t - deduped[-1][0]) > 1e-6:
                    deduped.append((t, pt))

            for i in range(0, len(deduped) - 1, 2):
                p0 = deduped[i][1]
                p1 = deduped[i + 1][1]
                c.create_line(
                    p0[0],
                    p0[1],
                    p1[0],
                    p1[1],
                    fill=self.hatch_color,
                    width=1.0,
                )

    @staticmethod
    def _line_segment_intersection(direction, normal, offset, seg_a, seg_b):
        dx, dy = direction
        nx, ny = normal
        ax, ay = seg_a
        bx, by = seg_b
        sx = bx - ax
        sy = by - ay

        denom = nx * sx + ny * sy
        if abs(denom) < 1e-9:
            return None

        t = (offset - (nx * ax + ny * ay)) / denom
        if t < 0.0 or t > 1.0:
            return None

        x = ax + t * sx
        y = ay + t * sy
        line_t = x * dx + y * dy
        return line_t, (x, y)

    @staticmethod
    def _frange(start: float, end: float, step: float):
        value = start
        while value <= end + 1e-6:
            yield value
            value += step

    def _draw_grid(self, w: int, h: int):
        c = self.canvas
        # Pick grid spacing in pixels based on zoom
        world_step = self.grid_step
        px_step = max(25, world_step * self.zoom)
        # Convert to nearest nice pixel step
        # Keep it simple: use px_step as-is
        # Find world origin on screen
        # Draw axes
        c.create_line(0, self.offset[1], w, self.offset[1], fill=GRID_AXIS_COLOR)
        c.create_line(self.offset[0], 0, self.offset[0], h, fill=GRID_AXIS_COLOR)
        # Light grid lines
        # Horizontal
        y = self.offset[1] % px_step
        while y < h:
            c.create_line(0, y, w, y, fill=GRID_LINE_COLOR)
            y += px_step
        # Vertical
        x = self.offset[0] % px_step
        while x < w:
            c.create_line(x, 0, x, h, fill=GRID_LINE_COLOR)
            x += px_step

    def _draw_center_cross(self, w: int, h: int):
        c = self.canvas
        c.create_text(w // 2, h // 2, text="No geometry loaded", fill=EMPTY_TEXT_COLOR)
        c.create_line(
            w // 2 - 10, h // 2, w // 2 + 10, h // 2, fill=GRID_CENTER_CROSS_COLOR
        )
        c.create_line(
            w // 2, h // 2 - 10, w // 2, h // 2 + 10, fill=GRID_CENTER_CROSS_COLOR
        )
