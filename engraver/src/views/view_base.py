from __future__ import annotations
import math
from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk
from geometry.GeoUtil import GeoUtil
from geometry.PointInPolygonResult import PointInPolygonResult
from views.view_constants import BACKGROUND_COLOR

if TYPE_CHECKING:
    # Import only for type checking; does not run at runtime
    from views.view_app import AppView


class BaseView(ttk.Frame):
    def __init__(self, master, app: "AppView"):
        super().__init__(master)
        self.app = app
        self.canvas = tk.Canvas(
            self, background=BACKGROUND_COLOR, highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Expose>", lambda e: self.redraw())
        self.fit_to_view_pending = False

    @property
    def hatch_angle_deg(self) -> float:
        return getattr(self.app, "hatch_angle_deg", 45.0)

    @property
    def hatch_spacing_px(self) -> float:
        return getattr(self.app, "hatch_spacing_px", 8.0)

    def _on_resize(self, event):
        self.redraw()

    def redraw(self):
        raise NotImplementedError

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

    def _rebuild_selected_polygons(self) -> None:
        if not self.app.selected_polygons:
            return
        polygons = self._collect_polygons()
        if not polygons:
            self.app.selected_polygons = []
            return
        polygons_by_index = {poly["index"]: poly for poly in polygons}
        rebuilt = []
        for entry in self.app.selected_polygons:
            selected = polygons_by_index.get(entry["polygon"]["index"])
            if not selected:
                continue
            holes = []
            for poly in polygons:
                if poly is selected:
                    continue
                result = GeoUtil.point_in_polygon(poly["points"][0], selected["points"])
                if result == PointInPolygonResult.Inside:
                    holes.append(poly)
            rebuilt.append({"polygon": selected, "holes": holes})
        self.app.selected_polygons = rebuilt

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
                hit = self._line_segment_intersection(
                    (dx, dy), (nx, ny), offset, p0, p1
                )
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
