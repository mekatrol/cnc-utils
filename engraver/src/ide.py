from pathlib import Path
import sys
import json
from typing import List

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QAction, QPen, QBrush, QColor, QTransform, QPainterPath, QPainter
from PySide6.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene,
    QGraphicsPathItem, QGraphicsEllipseItem,
    QFileDialog, QMainWindow
)

from geometry.GeometryInt import GeometryInt
from geometry.PointInt import PointInt
from geometry.PolylineInt import PolylineInt
from svg.SvgConverter import SvgConverter


class View(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # pan with mouse
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

    def wheelEvent(self, e):
        s = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.scale(s, s)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.view = View(self.scene)
        self.setCentralWidget(self.view)

        act = QAction("Open…", self)
        act.triggered.connect(self.open_file)
        self.menuBar().addMenu("&File").addAction(act)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open geometry JSON/SVG",
                                              filter="SVG (*.svg);;JSON (*.json);;All (*.*)")
        if not path:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # hourglass
        try:
            self.scene.clear()
            if path.lower().endswith(".json"):
                self.geom = Main.load_geometry_from_json(Path(path))
            else:
                self.geom = SvgConverter.svg_to_geometry_int(path, scale=10000, tol=0.25)

            self.fit_scene(self.geom)
            self.load_geometry(self.geom)
        finally:
            QApplication.restoreOverrideCursor()

    def load_geometry(self, g: GeometryInt):
        scale = getattr(g, "scale", None) or (g.get("scale", 1) if isinstance(g, dict) else 1)
        colors = ["#2aaaff", "#ff6b6b", "#51cf66", "#fab005", "#845ef7"]

        polys = g.polylines

        for i, pl in enumerate(polys):
            pts = pl.points
            if len(pts) < 1:
                continue
            path = QPainterPath()
            x0, y0 = pts[0]
            path.moveTo(QPointF(x0 / scale, -y0 / scale))
            for x, y in pts[1:]:
                path.lineTo(QPointF(x / scale, -y / scale))
            item = QGraphicsPathItem(path)
            item.setPen(QPen(QColor(colors[i % len(colors)]), 1.5))
            self.scene.addItem(item)

        pts = g.points

        r = 3
        for i, (x, y) in enumerate(pts):
            cx, cy = x / scale, -y / scale
            dot = QGraphicsEllipseItem(cx - r, cy - r, 2 * r, 2 * r)
            color = QColor(colors[i % len(colors)])
            dot.setPen(QPen(color))
            dot.setBrush(QBrush(color))
            self.scene.addItem(dot)

    def fit_scene(self, g: GeometryInt):
        bounds = g.bounds()
        rect = QRectF(bounds[0], bounds[1], bounds[2], bounds[3])
        # rect = self.scene.itemsBoundingRect()

        if rect.isNull():
            rect = QRectF(-50, -50, 100, 100)
        self.view.setTransform(QTransform())  # reset zoom
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    @staticmethod
    def _as_point(pt) -> PointInt:
        if isinstance(pt, dict) and "x" in pt and "y" in pt:
            return PointInt(int(pt["x"]), int(pt["y"]))
        if (
            isinstance(pt, (list, tuple))
            and len(pt) >= 2
            and all(isinstance(v, (int, float)) for v in pt[:2])
        ):
            return PointInt(int(pt[0]), int(pt[1]))

        raise ValueError(f"Unsupported point format: {pt!r}")

    @staticmethod
    def load_geometry_from_json(path: Path) -> GeometryInt:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scale = int(data.get("scale", 1) or 1)

        pls: List[PolylineInt] = []

        # Case 1: standard "polylines": [{"pts": [...]}, ...]
        if isinstance(data.get("polylines"), list):
            for pl in data.get("polylines", []):
                pts_raw = pl.get("pts", []) if isinstance(pl, dict) else []
                pts = [Main._as_point(p) for p in pts_raw]
                if len(pts) >= 2:
                    pls.append(PolylineInt(pts))

        # Case 2: root-level "points": [ [ [x,y], ... ], [ ... ] ]
        elif isinstance(data.get("points"), list):
            for poly in data.get("points", []):
                if isinstance(poly, list):
                    pts = [Main._as_point(p) for p in poly]
                    if len(pts) >= 2:
                        pls.append(PolylineInt(pts))

        return GeometryInt(pls, [], scale)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Main()
    w.resize(1000, 700)
    w.showMaximized()
    sys.exit(app.exec())
