from __future__ import annotations
from typing import TYPE_CHECKING
from geometry.GeoUtil import GeoUtil
from geometry.PointInPolygonResult import PointInPolygonResult
from geometry.PointInt import PointInt
from views.view_base import BaseView
from views.view_constants import (
    AXIS_X_COLOR,
    AXIS_Y_COLOR,
    BACKGROUND_COLOR,
    BOUNDARY_COLOR,
    COLORS,
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
        self.hatch_color = HATCH_COLOR
        self.fill_color = FILL_COLOR_2D
        self.fill_stipple = "gray12"
        self.hover_fill_color = FILL_COLOR_2D
        self.hover_fill_stipple = ""
        self._hovered_polygon = None

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        # Linux/Mac wheel events
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e, -1))
        self.canvas.bind_all("f", lambda e: self.fit_to_view())
        self.canvas.bind_all("F", lambda e: self.fit_to_view())
        self.canvas.bind_all("r", lambda e: self.reset_view())
        self.canvas.bind_all("R", lambda e: self.reset_view())

    @property
    def hatch_spacing_px(self) -> float:
        return max(1.0, float(self.app.hatch_spacing_px) * self.zoom)

    # View control
    def fit_to_view(self, include_origin: bool = False):
        self.fit_to_view_pending = False
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        minx, miny, maxx, maxy = GeoUtil.world_bounds(self.app.model)
        if include_origin:
            minx = min(minx, 0.0)
            miny = min(miny, 0.0)
            maxx = max(maxx, 0.0)
            maxy = max(maxy, 0.0)
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
        self._rebuild_selected_polygons()
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
            self.app.redraw_all()

    def _on_motion(self, event):
        if self._dragging:
            return
        hovered = self._hover_polygon(event.x, event.y)
        if hovered is None and self._hovered_polygon is None:
            return
        if (
            hovered is not None
            and self._hovered_polygon is not None
            and hovered["polygon"]["index"] == self._hovered_polygon["polygon"]["index"]
        ):
            return
        self._hovered_polygon = hovered
        self.redraw()

    def _on_leave(self, event):
        if self._hovered_polygon is None:
            return
        self._hovered_polygon = None
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
        self._draw_hover()
        self._draw_selection()
        # Geometry
        g = self.app.model
        if not g or not g.polylines:
            self._draw_center_cross(w, h)
            self._draw_axis_gizmo(w, h)
            return
        s = g.scale if g.scale else 1

        show_geometry = True
        show_geometry_var = getattr(self.app, "show_geometry", None)
        if show_geometry_var is not None:
            try:
                show_geometry = show_geometry_var.get()
            except Exception:
                show_geometry = bool(show_geometry_var)

        if show_geometry:
            # Draw geometry polylines
            selected_indices = set(self.app.selected_edge_polygons)
            selected_line_indices = set(self.app.selected_polylines)
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
                    is_polygon = (
                        len(polyline.points) >= 3
                        and polyline.points[0] == polyline.points[-1]
                    )
                    is_selected = (
                        i in selected_indices if is_polygon else i in selected_line_indices
                    )
                    width = 1.5 * 2.0 if is_selected else 1.5
                    c.create_line(*coords, fill=color, width=width)

            show_degenerate = True
            show_degenerate_var = getattr(self.app, "show_degenerate", None)
            if show_degenerate_var is not None:
                try:
                    show_degenerate = show_degenerate_var.get()
                except Exception:
                    show_degenerate = bool(show_degenerate_var)
            degenerates = getattr(g, "degenerate_polylines", [])
            if show_degenerate and degenerates:
                for polyline in degenerates:
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
                    if coords:
                        c.create_line(*coords, fill="#777777", width=1.0, dash=(3, 3))

        self._draw_generated_paths()

        if show_geometry:
            # Points
            for i, pt in enumerate(g.points):
                xw, yw = pt.x / s, pt.y / s
                xs = xw * self.zoom + self.offset[0]
                ys = -yw * self.zoom + self.offset[1]
                r = 3  # screen pixels
                color = COLORS[(i + len(COLORS) >> 1) % len(COLORS)]
                c.create_oval(
                    xs - r, ys - r, xs + r, ys + r, outline=color, fill=color
                )
        self._draw_axis_gizmo(w, h)

    def _draw_generated_paths(self) -> None:
        show_paths = getattr(self.app, "show_generated_paths", None)
        if show_paths is not None:
            try:
                if not show_paths.get():
                    return
            except Exception:
                if not show_paths:
                    return
        paths = getattr(self.app, "generated_paths", [])
        if not paths:
            return
        c = self.canvas
        for entry in paths:
            if not entry.get("visible", True):
                continue
            lines = entry.get("lines", {})
            children = entry.get("children", [])
            if children:
                for child in children:
                    if not isinstance(child, dict):
                        continue
                    if not child.get("visible", True):
                        continue
                    key = child.get("key")
                    if not key:
                        continue
                    color = (
                        BOUNDARY_COLOR if str(key).startswith("boundary") else HATCH_COLOR
                    )
                    width = 1.5 if color == BOUNDARY_COLOR else 1.0
                    for segment in lines.get(key, []):
                        if len(segment) < 2:
                            continue
                        (x0, y0), (x1, y1) = segment[0], segment[1]
                        xs0 = x0 * self.zoom + self.offset[0]
                        ys0 = -y0 * self.zoom + self.offset[1]
                        xs1 = x1 * self.zoom + self.offset[0]
                        ys1 = -y1 * self.zoom + self.offset[1]
                        c.create_line(xs0, ys0, xs1, ys1, fill=color, width=width)
                continue
            for key in ("primary", "secondary"):
                for segment in lines.get(key, []):
                    if len(segment) < 2:
                        continue
                    (x0, y0), (x1, y1) = segment[0], segment[1]
                    xs0 = x0 * self.zoom + self.offset[0]
                    ys0 = -y0 * self.zoom + self.offset[1]
                    xs1 = x1 * self.zoom + self.offset[0]
                    ys1 = -y1 * self.zoom + self.offset[1]
                    c.create_line(xs0, ys0, xs1, ys1, fill=HATCH_COLOR, width=1.0)
            for segment in lines.get("boundary", []):
                if len(segment) < 2:
                    continue
                (x0, y0), (x1, y1) = segment[0], segment[1]
                xs0 = x0 * self.zoom + self.offset[0]
                ys0 = -y0 * self.zoom + self.offset[1]
                xs1 = x1 * self.zoom + self.offset[0]
                ys1 = -y1 * self.zoom + self.offset[1]
                c.create_line(xs0, ys0, xs1, ys1, fill=BOUNDARY_COLOR, width=1.5)

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

    def _select_polygon(self, x: float, y: float) -> None:
        show_geometry = getattr(self.app, "show_geometry", None)
        if show_geometry is not None:
            try:
                if not show_geometry.get():
                    self.app.selected_polygons = []
                    self.app.update_properties()
                    return
            except Exception:
                if not show_geometry:
                    self.app.selected_polygons = []
                    self.app.update_properties()
                    return
        g = self.app.model
        if not g or not g.polylines:
            self.app.selected_polygons = []
            self.app.update_properties()
            return

        polygons = self._collect_polygons()
        s = g.scale or 1

        def toggle_line_selection(index):
            if index in self.app.selected_polylines:
                self.app.selected_polylines.remove(index)
            else:
                self.app.selected_polylines.append(index)

        def toggle_selection(selected):
            selected_entry = self._build_polygon_entry(selected, polygons)
            existing_idx = next(
                (
                    idx
                    for idx, entry in enumerate(self.app.selected_polygons)
                    if entry["polygon"]["index"] == selected["index"]
                ),
                None,
            )
            if existing_idx is None:
                self.app.selected_polygons.append(selected_entry)
            else:
                self.app.selected_polygons.pop(existing_idx)
            self.app.update_properties()

        def toggle_edge_selection(selected):
            idx = selected["index"]
            if idx in self.app.selected_edge_polygons:
                self.app.selected_edge_polygons.remove(idx)
            else:
                self.app.selected_edge_polygons.append(idx)

        edge_hit = self._find_edge_hit_polygon(x, y, polygons, s)
        if edge_hit is not None:
            toggle_edge_selection(edge_hit)
            return

        line_hit = self._find_line_hit_polyline(x, y, g.polylines, s)
        if line_hit is not None:
            toggle_line_selection(line_hit)
            return

        query = self._screen_to_pointint(x, y, s)
        selected = None
        selected_area = None
        qx, qy = query.x, query.y
        for poly in polygons:
            bbox = poly.get("bbox")
            if bbox is None:
                bbox = self._polygon_bbox(poly["points"])
            minx, miny, maxx, maxy = bbox
            if qx < minx or qx > maxx or qy < miny or qy > maxy:
                continue
            result = GeoUtil.point_in_polygon(query, poly["points"])
            if result in (
                PointInPolygonResult.Inside,
                PointInPolygonResult.Edge,
                PointInPolygonResult.Vertex,
            ):
                area = abs(poly["area"]) if "area" in poly else abs(
                    GeoUtil.area(poly["points"])
                )
                if selected is None or area < selected_area:
                    selected = poly
                    selected_area = area

        if not selected:
            self.app.selected_polygons = []
            self.app.update_properties()
            return
        toggle_selection(selected)

    def _build_polygon_entry(self, selected, polygons):
        holes = []
        selected_bbox = selected.get("bbox")
        for poly in polygons:
            if poly is selected:
                continue
            if selected_bbox and poly.get("bbox"):
                sminx, sminy, smaxx, smaxy = selected_bbox
                pminx, pminy, pmaxx, pmaxy = poly["bbox"]
                if pminx < sminx or pmaxx > smaxx or pminy < sminy or pmaxy > smaxy:
                    continue
            result = GeoUtil.point_in_polygon(poly["points"][0], selected["points"])
            if result == PointInPolygonResult.Inside:
                holes.append(poly)
        return {"polygon": selected, "holes": holes}

    def _hover_polygon(self, x: float, y: float):
        show_geometry = getattr(self.app, "show_geometry", None)
        if show_geometry is not None:
            try:
                if not show_geometry.get():
                    return None
            except Exception:
                if not show_geometry:
                    return None
        g = self.app.model
        if not g or not g.polylines:
            return None

        polygons = self._collect_polygons()
        s = g.scale or 1
        query = self._screen_to_pointint(x, y, s)
        selected = None
        selected_area = None
        qx, qy = query.x, query.y
        for poly in polygons:
            bbox = poly.get("bbox")
            if bbox is None:
                bbox = self._polygon_bbox(poly["points"])
            minx, miny, maxx, maxy = bbox
            if qx < minx or qx > maxx or qy < miny or qy > maxy:
                continue
            result = GeoUtil.point_in_polygon(query, poly["points"])
            if result in (
                PointInPolygonResult.Inside,
                PointInPolygonResult.Edge,
                PointInPolygonResult.Vertex,
            ):
                area = abs(poly["area"]) if "area" in poly else abs(
                    GeoUtil.area(poly["points"])
                )
                if selected is None or area < selected_area:
                    selected = poly
                    selected_area = area

        if not selected:
            return None
        return self._build_polygon_entry(selected, polygons)

    @staticmethod
    def _polygon_bbox(points) -> tuple[int, int, int, int]:
        minx = min(pt.x for pt in points)
        maxx = max(pt.x for pt in points)
        miny = min(pt.y for pt in points)
        maxy = max(pt.y for pt in points)
        return minx, miny, maxx, maxy

    @staticmethod
    def _point_segment_distance_sq(point, a, b) -> float:
        px, py = point
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if dx == 0.0 and dy == 0.0:
            return (px - ax) ** 2 + (py - ay) ** 2
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        if t <= 0.0:
            cx, cy = ax, ay
        elif t >= 1.0:
            cx, cy = bx, by
        else:
            cx = ax + t * dx
            cy = ay + t * dy
        return (px - cx) ** 2 + (py - cy) ** 2

    def _find_edge_hit_polygon(self, x: float, y: float, polygons, scale: int):
        edge_tol = 5.0
        edge_tol_sq = edge_tol * edge_tol
        best_poly = None
        best_dist = None

        for poly in polygons:
            screen_points = []
            minx = None
            maxx = None
            miny = None
            maxy = None
            for pt in poly["points"]:
                xw, yw = pt.x / scale, pt.y / scale
                xs = xw * self.zoom + self.offset[0]
                ys = -yw * self.zoom + self.offset[1]
                screen_points.append((xs, ys))
                if minx is None:
                    minx = maxx = xs
                    miny = maxy = ys
                else:
                    minx = min(minx, xs)
                    maxx = max(maxx, xs)
                    miny = min(miny, ys)
                    maxy = max(maxy, ys)

            if minx is None:
                continue
            if (
                x < minx - edge_tol
                or x > maxx + edge_tol
                or y < miny - edge_tol
                or y > maxy + edge_tol
            ):
                continue

            count = len(screen_points)
            for i in range(count):
                a = screen_points[i]
                b = screen_points[(i + 1) % count]
                dist_sq = self._point_segment_distance_sq((x, y), a, b)
                if dist_sq <= edge_tol_sq and (
                    best_dist is None or dist_sq < best_dist
                ):
                    best_dist = dist_sq
                    best_poly = poly

        return best_poly

    def _find_line_hit_polyline(self, x: float, y: float, polylines, scale: int):
        edge_tol = 5.0
        edge_tol_sq = edge_tol * edge_tol
        best_index = None
        best_dist = None

        for idx, polyline in enumerate(polylines):
            points = polyline.points
            if len(points) < 2:
                continue
            if len(points) >= 3 and points[0] == points[-1]:
                continue

            screen_points = []
            minx = None
            maxx = None
            miny = None
            maxy = None
            for pt in points:
                xw, yw = pt.x / scale, pt.y / scale
                xs = xw * self.zoom + self.offset[0]
                ys = -yw * self.zoom + self.offset[1]
                screen_points.append((xs, ys))
                if minx is None:
                    minx = maxx = xs
                    miny = maxy = ys
                else:
                    minx = min(minx, xs)
                    maxx = max(maxx, xs)
                    miny = min(miny, ys)
                    maxy = max(maxy, ys)

            if minx is None:
                continue
            if (
                x < minx - edge_tol
                or x > maxx + edge_tol
                or y < miny - edge_tol
                or y > maxy + edge_tol
            ):
                continue

            for i in range(len(screen_points) - 1):
                a = screen_points[i]
                b = screen_points[i + 1]
                dist_sq = self._point_segment_distance_sq((x, y), a, b)
                if dist_sq <= edge_tol_sq and (
                    best_dist is None or dist_sq < best_dist
                ):
                    best_dist = dist_sq
                    best_index = idx

        return best_index

    def _draw_selection(self) -> None:
        show_geometry = getattr(self.app, "show_geometry", None)
        if show_geometry is not None:
            try:
                if not show_geometry.get():
                    return
            except Exception:
                if not show_geometry:
                    return
        if not self.app.selected_polygons:
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
            entry["polygon"]["index"] for entry in self.app.selected_polygons
        }

        for entry in self.app.selected_polygons:
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

    def _draw_hover(self) -> None:
        show_geometry = getattr(self.app, "show_geometry", None)
        if show_geometry is not None:
            try:
                if not show_geometry.get():
                    return
            except Exception:
                if not show_geometry:
                    return
        if not self._hovered_polygon:
            return
        g = self.app.model
        if not g:
            return
        s = g.scale or 1
        c = self.canvas
        bg = c.cget("background") or BACKGROUND_COLOR

        selected_indices = {
            entry["polygon"]["index"] for entry in self.app.selected_polygons
        }
        hover_polygon = self._hovered_polygon["polygon"]
        if hover_polygon["index"] in selected_indices:
            return

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

        hover_holes = self._hovered_polygon["holes"]
        if self._selected_polygon_solid_fill:
            coords = to_screen(hover_polygon["points"])
            if coords:
                c.create_polygon(
                    *coords,
                    fill=self.hover_fill_color,
                    outline="",
                    stipple=self.hover_fill_stipple,
                )
            for hole in hover_holes:
                if hole["index"] in selected_indices:
                    continue
                hole_coords = to_screen(hole["points"])
                if hole_coords:
                    c.create_polygon(*hole_coords, fill=bg, outline="")
        else:
            polygon_points = to_screen_points(hover_polygon["points"])
            if polygon_points:
                self._draw_hatch_polygon(polygon_points)
            for hole in hover_holes:
                if hole["index"] in selected_indices:
                    continue
                hole_points = to_screen_points(hole["points"])
                if hole_points:
                    hole_coords = [coord for pt in hole_points for coord in pt]
                    c.create_polygon(*hole_coords, fill=bg, outline="")

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
