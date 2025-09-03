from enum import IntEnum
from typing import Iterable, List, Tuple
import pyclipper
from geometry.GeometryInt import GeometryInt
from geometry.PointInt import PointInt
from geometry.PolylineInt import PolylineInt


Path = List[Tuple[int, int]]
Paths = List[Path]


class ClipOp(IntEnum):
    INTERSECTION = 0
    UNION = 1
    DIFFERENCE = 2
    XOR = 3


class FillRule(IntEnum):
    EVENODD = 0
    NONZERO = 1
    POSITIVE = 2
    NEGATIVE = 3


class VattiClipper:
    @staticmethod
    def _to_paths(polys: GeometryInt) -> Paths:
        out: Paths = []
        for poly in polys.polylines:
            pts = poly.points
            if not pts:
                continue
            # drop duplicate closing vertex if present
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            path = [(p.x, p.y) for p in pts]
            if len(path) >= 3:
                out.append(path)
        return out

    @staticmethod
    def paths_to_geometry_int(paths: Iterable[Iterable[Tuple[int, int]]]) -> GeometryInt:
        polylines: List[PolylineInt] = []

        for path in paths:
            points = [PointInt(int(x), int(y)) for (x, y) in path]

            # Drop
            if len(points) >= 2 and points[0] == points[-1]:
                points = points[:-1]

            if points:
                polylines.append(PolylineInt(points=points))

        geo = GeometryInt(polylines=polylines, points=[])

        geo.simplify()

        return geo

    @staticmethod
    def clip_polygons(subjects: GeometryInt,
                      clips: GeometryInt,
                      op: ClipOp,
                      fill_rule: FillRule = FillRule.EVENODD) -> GeometryInt:

        subj = VattiClipper._to_paths(subjects)
        clip = VattiClipper._to_paths(clips)
        pc = pyclipper.Pyclipper()  # type: ignore

        if subj:
            pc.AddPaths(subj, pyclipper.PT_SUBJECT, True)  # type: ignore # closed

        if clip:
            pc.AddPaths(clip, pyclipper.PT_CLIP, True)     # type: ignore # closed

        sol = pc.Execute(int(op), int(fill_rule), int(fill_rule))

        return VattiClipper.paths_to_geometry_int(sol)

    @staticmethod
    def clip_polygons_tree(subjects: GeometryInt,
                           clips: GeometryInt,
                           op: ClipOp,
                           fill_rule: FillRule = FillRule.EVENODD):

        subj = VattiClipper._to_paths(subjects)
        clip = VattiClipper._to_paths(clips)
        pc = pyclipper.Pyclipper()  # type: ignore

        if subj:
            pc.AddPaths(subj, pyclipper.PT_SUBJECT, True)  # type: ignore

        if clip:
            pc.AddPaths(clip, pyclipper.PT_CLIP, True)  # type: ignore

        tree = pc.Execute2(int(op), int(fill_rule), int(fill_rule))  # PolyTree

        def walk(node):
            items = []
            for child in node.Childs:
                poly = PolylineInt(points=[PointInt(x, y) for (x, y) in child.Contour])
                items.append({"polyline": poly, "is_hole": child.IsHole})
                items.extend(walk(child))
            return items

        return walk(tree)
