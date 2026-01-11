from __future__ import annotations
import math
from typing import TYPE_CHECKING, Tuple
from geometry.GeoUtil import GeoUtil
from geometry.PointInt import PointInt
from geometry.PointInPolygonResult import PointInPolygonResult
from views.view_base import BaseView
from views.view_constants import (
    AXIS_X_COLOR,
    AXIS_X_DIM_COLOR,
    AXIS_Y_COLOR,
    AXIS_Y_DIM_COLOR,
    AXIS_Z_COLOR,
    AXIS_Z_DIM_COLOR,
    BACKGROUND_COLOR,
    BOUNDARY_COLOR,
    COLORS,
    EMPTY_TEXT_COLOR,
    FILL_COLOR_3D,
    GRID_LINE_COLOR,
    HATCH_COLOR,
    ORIGIN_BALL_COLOR,
    ORIGIN_BALL_DIM_COLOR,
)


if TYPE_CHECKING:
    # Import only for type checking; does not run at runtime
    from views.view_app import AppView


class View3D(BaseView):
    def __init__(self, master, app: "AppView"):
        super().__init__(master, app)
        # Camera/world params
        self.yaw = math.radians(0)
        self.pitch = math.radians(0)
        self.zoom = 50.0  # pixel scale (screen pixels per world unit at z≈0)
        self.pan = [0.0, 0.0]  # screen-space pan in pixels
        self._pivot_center: tuple[float, float] | None = None

        # Mouse state
        self._rotating = False
        self._panning = False
        self._last = (0, 0)
        self._drag_moved = False
        self._press_pos = (0, 0)

        # Selection state
        self._selected_polygon_solid_fill = True
        self.hatch_color = HATCH_COLOR
        self.fill_color = FILL_COLOR_3D
        self.fill_stipple = "gray12"

        # Bindings
        self.canvas.bind("<ButtonPress-1>", self._on_press_left)
        self.canvas.bind("<ButtonPress-3>", self._on_press_right)
        self.canvas.bind("<B1-Motion>", self._on_drag_left)
        self.canvas.bind("<B3-Motion>", self._on_drag_right)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(1))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self._zoom(-1))  # Linux
        self.canvas.bind_all("f", lambda e: self.fit_to_view())
        self.canvas.bind_all("F", lambda e: self.fit_to_view())

    @property
    def hatch_spacing_px(self) -> float:
        return max(1.0, float(self.app.hatch_spacing_px) * self.zoom)

    def fit_to_view(self, include_origin: bool = False):
        # Compute bounding box in world units and scale to canvas size
        self.fit_to_view_pending = False
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        if w <= 1 or h <= 1:
            self.fit_to_view_pending = True
            return
        minx, miny, maxx, maxy = GeoUtil.world_bounds(self.app.model)
        if include_origin:
            minx = min(minx, 0.0)
            miny = min(miny, 0.0)
            maxx = max(maxx, 0.0)
            maxy = max(maxy, 0.0)
        dx = maxx - minx or 1.0
        dy = maxy - miny or 1.0
        # Fit geometry to the visible area with a little padding
        scale_x = (w * 0.9) / dx
        scale_y = (h * 0.9) / dy
        self.zoom = min(scale_x, scale_y)
        # Center the model on screen (pivot at screen center)
        self.pan = [0.0, 0.0]
        self.yaw = math.radians(0)
        self.pitch = math.radians(0)
        self._pivot_center = ((minx + maxx) * 0.5, (miny + maxy) * 0.5)

        self._rebuild_selected_polygons()
        self.redraw()

    def _get_pivot_center(self) -> tuple[float, float]:
        if self._pivot_center is not None:
            return self._pivot_center
        g = self.app.model
        if not g or not g.polylines:
            return 0.0, 0.0
        minx, miny, maxx, maxy = GeoUtil.world_bounds(g)
        return (minx + maxx) * 0.5, (miny + maxy) * 0.5

    # Events
    def _on_press_left(self, event):
        self._rotating = True
        self._last = (event.x, event.y)
        self._drag_moved = False
        self._press_pos = (event.x, event.y)

    def _on_press_right(self, event):
        self._panning = True
        self._last = (event.x, event.y)

    def _on_drag_left(self, event):
        if not self._rotating:
            return
        if not self._drag_moved:
            dx0 = event.x - self._press_pos[0]
            dy0 = event.y - self._press_pos[1]
            if abs(dx0) + abs(dy0) < 3:
                return
            self._drag_moved = True
            self._last = (event.x, event.y)
            return
        dx = event.x - self._last[0]
        dy = event.y - self._last[1]
        self.yaw += dx * 0.01
        self.pitch += dy * 0.01
        # Clamp pitch to avoid flipping
        self.pitch = max(-math.pi / 2 + 0.05, min(math.pi / 2 - 0.05, self.pitch))
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
        was_rotating = self._rotating
        self._rotating = False
        self._panning = False
        if was_rotating and not self._drag_moved:
            self._select_polygon(event.x, event.y)
            self.app.redraw_all()

    def _on_wheel(self, event):
        self._zoom(1 if event.delta > 0 else -1)

    def _zoom(self, direction):
        factor = 1.1 if direction > 0 else 1.0 / 1.1
        self.zoom *= factor
        self.zoom = max(1e-3, min(1e6, self.zoom))
        self.redraw()

    # Math helpers
    def _project_point(
        self, x: float, y: float, z: float, w: int, h: int
    ) -> Tuple[float, float, float]:
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

        scale = self.zoom

        xs = x1 * scale + w * 0.5 + self.pan[0]
        ys = -y2 * scale + h * 0.5 + self.pan[1]
        return xs, ys, z2

    # Drawing
    def redraw(self):
        if self.fit_to_view_pending:
            self.fit_to_view()
            if self.fit_to_view_pending:
                return
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        h = c.winfo_height()

        # Compute pivot at model center in world units (bbox center)
        cx, cy = self._get_pivot_center()

        # Grid plane (drawn around pivot)
        self._draw_grid_3d(w, h)
        self._draw_axis_gizmo(w, h, cx, cy)
        self._draw_selection(w, h, cx, cy)

        g = self.app.model
        if not g or not g.polylines:
            c.create_text(
                w // 2, h // 2, text="No geometry loaded", fill=EMPTY_TEXT_COLOR
            )
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
            # Draw lines RELATIVE to pivot so rotations happen about model center
            selected_indices = set(self.app.selected_edge_polygons)
            selected_line_indices = set(self.app.selected_polylines)
            for i, pl in enumerate(g.polylines):
                if len(pl.points) < 2:
                    continue
                last_pt = None
                for p in pl.points:
                    x_rel = p.x / s - cx
                    y_rel = p.y / s - cy
                    xs, ys, _ = self._project_point(x_rel, y_rel, 0.0, w, h)
                    if last_pt is not None:
                        color = COLORS[i % len(COLORS)]
                        is_polygon = (
                            len(pl.points) >= 3 and pl.points[0] == pl.points[-1]
                        )
                        is_selected = (
                            i in selected_indices
                            if is_polygon
                            else i in selected_line_indices
                        )
                        width = 1.5 * 2.0 if is_selected else 1.5
                        c.create_line(
                            last_pt[0], last_pt[1], xs, ys, fill=color, width=width
                        )
                    last_pt = (xs, ys)

            show_degenerate = True
            show_degenerate_var = getattr(self.app, "show_degenerate", None)
            if show_degenerate_var is not None:
                try:
                    show_degenerate = show_degenerate_var.get()
                except Exception:
                    show_degenerate = bool(show_degenerate_var)
            degenerates = getattr(g, "degenerate_polylines", [])
            if show_degenerate and degenerates:
                for pl in degenerates:
                    if len(pl.points) < 2:
                        continue
                    last_pt = None
                    for p in pl.points:
                        x_rel = p.x / s - cx
                        y_rel = p.y / s - cy
                        xs, ys, _ = self._project_point(x_rel, y_rel, 0.0, w, h)
                        if last_pt is not None:
                            c.create_line(
                                last_pt[0],
                                last_pt[1],
                                xs,
                                ys,
                                fill="#777777",
                                width=1.0,
                                dash=(3, 3),
                            )
                        last_pt = (xs, ys)

        self._draw_generated_paths(w, h, cx, cy)

    def _draw_generated_paths(self, w: int, h: int, cx: float, cy: float) -> None:
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
                        x_rel0 = x0 - cx
                        y_rel0 = y0 - cy
                        x_rel1 = x1 - cx
                        y_rel1 = y1 - cy
                        xs0, ys0, _ = self._project_point(x_rel0, y_rel0, 0.0, w, h)
                        xs1, ys1, _ = self._project_point(x_rel1, y_rel1, 0.0, w, h)
                        c.create_line(xs0, ys0, xs1, ys1, fill=color, width=width)
                continue
            for key in ("primary", "secondary"):
                for segment in lines.get(key, []):
                    if len(segment) < 2:
                        continue
                    (x0, y0), (x1, y1) = segment[0], segment[1]
                    x_rel0 = x0 - cx
                    y_rel0 = y0 - cy
                    x_rel1 = x1 - cx
                    y_rel1 = y1 - cy
                    xs0, ys0, _ = self._project_point(x_rel0, y_rel0, 0.0, w, h)
                    xs1, ys1, _ = self._project_point(x_rel1, y_rel1, 0.0, w, h)
                    c.create_line(xs0, ys0, xs1, ys1, fill=HATCH_COLOR, width=1.0)
            for segment in lines.get("boundary", []):
                if len(segment) < 2:
                    continue
                (x0, y0), (x1, y1) = segment[0], segment[1]
                x_rel0 = x0 - cx
                y_rel0 = y0 - cy
                x_rel1 = x1 - cx
                y_rel1 = y1 - cy
                xs0, ys0, _ = self._project_point(x_rel0, y_rel0, 0.0, w, h)
                xs1, ys1, _ = self._project_point(x_rel1, y_rel1, 0.0, w, h)
                c.create_line(xs0, ys0, xs1, ys1, fill=BOUNDARY_COLOR, width=1.5)

    def _project_polygon(
        self, points, w: int, h: int, cx: float, cy: float, scale: int
    ):
        coords = []
        for pt in points:
            x_rel = pt.x / scale - cx
            y_rel = pt.y / scale - cy
            xs, ys, _ = self._project_point(x_rel, y_rel, 0.0, w, h)
            coords.append((xs, ys))
        return coords

    def _screen_to_world(
        self, x: float, y: float, center_x: float, center_y: float
    ) -> tuple[float, float] | None:
        w = self.canvas.winfo_width() or 1
        h = self.canvas.winfo_height() or 1
        x1 = (x - w * 0.5 - self.pan[0]) / self.zoom
        y2 = -(y - h * 0.5 - self.pan[1]) / self.zoom

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)

        # Solve for world x,y on z=0 plane directly from orthographic projection.
        if abs(cos_yaw) < 1e-6 or abs(cos_pitch) < 1e-6:
            return None
        x_rel = x1 / cos_yaw
        y_rel = (y2 - sin_pitch * sin_yaw * x_rel) / cos_pitch
        return x_rel + center_x, y_rel + center_y

    def _screen_to_pointint(
        self, x: float, y: float, scale: int, cx: float, cy: float
    ) -> "PointInt" | None:
        world = self._screen_to_world(x, y, cx, cy)
        if world is None:
            return None
        xw, yw = world
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

        s = g.scale if g.scale else 1
        cx, cy = self._get_pivot_center()
        polygons = self._collect_polygons()

        def toggle_line_selection(index):
            if index in self.app.selected_polylines:
                self.app.selected_polylines.remove(index)
            else:
                self.app.selected_polylines.append(index)

        def toggle_edge_selection(selected):
            idx = selected["index"]
            if idx in self.app.selected_edge_polygons:
                self.app.selected_edge_polygons.remove(idx)
            else:
                self.app.selected_edge_polygons.append(idx)

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        edge_hit = self._find_edge_hit_polygon(x, y, polygons, w, h, cx, cy, s)
        if edge_hit is not None:
            toggle_edge_selection(edge_hit)
            return

        line_hit = self._find_line_hit_polyline(x, y, g.polylines, w, h, cx, cy, s)
        if line_hit is not None:
            toggle_line_selection(line_hit)
            return

        containing = []
        query = self._screen_to_pointint(x, y, s, cx, cy)
        if query is not None:
            for poly in polygons:
                result = GeoUtil.point_in_polygon(query, poly["points"])
                if result in (
                    PointInPolygonResult.Inside,
                    PointInPolygonResult.Edge,
                    PointInPolygonResult.Vertex,
                ):
                    containing.append(poly)
        else:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w <= 1 or h <= 1:
                return
            for poly in polygons:
                proj = self._project_polygon(poly["points"], w, h, cx, cy, s)
                if self._point_in_polygon_screen((x, y), proj):
                    containing.append(poly)

        if not containing:
            self.app.selected_polygons = []
            self.app.update_properties()
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

    @staticmethod
    def _point_on_segment(point, a, b, eps: float = 1e-6) -> bool:
        px, py = point
        ax, ay = a
        bx, by = b
        cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        if abs(cross) > eps:
            return False
        dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
        if dot < -eps:
            return False
        seg_len = (bx - ax) * (bx - ax) + (by - ay) * (by - ay)
        return dot <= seg_len + eps

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

    def _point_in_polygon_screen(self, point, polygon_points) -> bool:
        if not polygon_points:
            return False
        x, y = point
        inside = False
        count = len(polygon_points)
        for i in range(count):
            p1 = polygon_points[i]
            p2 = polygon_points[(i + 1) % count]
            if self._point_on_segment(point, p1, p2):
                return True
            x1, y1 = p1
            x2, y2 = p2
            if (y1 > y) != (y2 > y):
                x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x_intersect > x:
                    inside = not inside
        return inside

    def _find_edge_hit_polygon(
        self, x: float, y: float, polygons, w: int, h: int, cx: float, cy: float, scale: int
    ):
        edge_tol = 5.0
        edge_tol_sq = edge_tol * edge_tol
        best_poly = None
        best_dist = None

        for poly in polygons:
            screen_points = self._project_polygon(poly["points"], w, h, cx, cy, scale)
            if not screen_points:
                continue
            xs = [pt[0] for pt in screen_points]
            ys = [pt[1] for pt in screen_points]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
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

    def _find_line_hit_polyline(
        self, x: float, y: float, polylines, w: int, h: int, cx: float, cy: float, scale: int
    ):
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
            for pt in points:
                x_rel = pt.x / scale - cx
                y_rel = pt.y / scale - cy
                xs, ys, _ = self._project_point(x_rel, y_rel, 0.0, w, h)
                screen_points.append((xs, ys))

            xs = [pt[0] for pt in screen_points]
            ys = [pt[1] for pt in screen_points]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
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

    def _draw_selection(self, w: int, h: int, cx: float, cy: float) -> None:
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
        s = g.scale if g.scale else 1
        c = self.canvas
        bg = c.cget("background")

        selected_indices = {
            entry["polygon"]["index"] for entry in self.app.selected_polygons
        }

        for entry in self.app.selected_polygons:
            selected_polygon = entry["polygon"]
            selected_holes = entry["holes"]
            polygon_points = self._project_polygon(
                selected_polygon["points"], w, h, cx, cy, s
            )
            if not polygon_points:
                continue

            if self._selected_polygon_solid_fill:
                coords = [coord for pt in polygon_points for coord in pt]
                c.create_polygon(
                    *coords,
                    fill=self.fill_color,
                    outline="",
                    stipple=self.fill_stipple,
                )
            else:
                self._draw_hatch_polygon(polygon_points)

            for hole in selected_holes:
                if hole["index"] in selected_indices:
                    continue
                hole_points = self._project_polygon(hole["points"], w, h, cx, cy, s)
                if hole_points:
                    hole_coords = [coord for pt in hole_points for coord in pt]
                    c.create_polygon(*hole_coords, fill=bg, outline="")

    def _draw_grid_3d(self, w: int, h: int):
        # Simple world XY grid centered on the pivot (origin after centering)
        c = self.canvas
        c.create_rectangle(0, 0, w, h, fill=BACKGROUND_COLOR, outline="")

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
            p2 = self._project_point(x, half, 0.0, w, h)
            c.create_line(p1[0], p1[1], p2[0], p2[1], fill=GRID_LINE_COLOR)
            x += step

        y = -half
        while y <= half:
            p1 = self._project_point(-half, y, 0.0, w, h)
            p2 = self._project_point(half, y, 0.0, w, h)
            c.create_line(p1[0], p1[1], p2[0], p2[1], fill=GRID_LINE_COLOR)
            y += step

        # Axes at the pivot
        ox, oy, _ = self._project_point(0.0, 0.0, 0.0, w, h)
        axis_len = step * 0.3
        xx, xy, _ = self._project_point(axis_len, 0.0, 0.0, w, h)
        yx, yy, _ = self._project_point(0.0, axis_len, 0.0, w, h)
        zx, zy, _ = self._project_point(0.0, 0.0, axis_len, w, h)

        # Simulate 50% opacity over the dark background by pre-blending colors.
        c.create_line(ox, oy, xx, xy, fill=AXIS_X_DIM_COLOR, width=2, arrow="last")  # X
        c.create_line(ox, oy, yx, yy, fill=AXIS_Y_DIM_COLOR, width=2, arrow="last")  # Y
        c.create_line(ox, oy, zx, zy, fill=AXIS_Z_DIM_COLOR, width=2, arrow="last")  # Z
        c.create_oval(
            ox - 4,
            oy - 4,
            ox + 4,
            oy + 4,
            fill=ORIGIN_BALL_DIM_COLOR,
            outline="",
        )

    def _draw_axis_gizmo(self, w: int, h: int, cx: float, cy: float) -> None:
        # Draw fixed-size axis arrows at the true world origin.
        ox, oy, _ = self._project_point(-cx, -cy, 0.0, w, h)
        axis_len = 45.0
        r = 5.0

        def axis_dir(x: float, y: float, z: float) -> tuple[float, float]:
            cyaw = math.cos(self.yaw)
            syaw = math.sin(self.yaw)
            cpitch = math.cos(self.pitch)
            spitch = math.sin(self.pitch)
            x1 = x * cyaw + z * syaw
            z1 = -x * syaw + z * cyaw
            y2 = y * cpitch - z1 * spitch
            dx = x1
            dy = -y2
            norm = math.hypot(dx, dy) or 1.0
            return dx / norm, dy / norm

        axes = [
            (1.0, 0.0, 0.0, AXIS_X_COLOR),
            (0.0, 1.0, 0.0, AXIS_Y_COLOR),
            (0.0, 0.0, 1.0, AXIS_Z_COLOR),
        ]
        for ax, ay, az, color in axes:
            dx, dy = axis_dir(ax, ay, az)
            x2 = ox + dx * axis_len
            y2 = oy + dy * axis_len
            self.canvas.create_line(ox, oy, x2, y2, fill=color, width=2, arrow="last")

        # Origin marker
        self.canvas.create_oval(
            ox - r, oy - r, ox + r, oy + r, fill=ORIGIN_BALL_COLOR, outline=""
        )

    @staticmethod
    def _nice_step(raw: float) -> float:
        if raw <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(raw))
        for m in (1, 2, 5, 10):
            if raw <= m * mag:
                return m * mag
        return 10 * mag
