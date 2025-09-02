from enum import Enum, auto
from typing import List
from geometry.PointInt import PointInt


class PolyRelation(Enum):
    DISJOINT = auto()      # no touching
    INTERSECT = auto()     # edges cross or touch (incl. tangency/overlap)
    A_INSIDE_B = auto()
    B_INSIDE_A = auto()


def _bbox(poly: List[PointInt]):
    xs = [p.x for p in poly]
    ys = [p.y for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _orient(a: PointInt, b: PointInt, c: PointInt) -> int:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)  # 2*area, integer


def _on_seg(a: PointInt, b: PointInt, c: PointInt) -> bool:
    return (min(a.x, b.x) <= c.x <= max(a.x, b.x) and
            min(a.y, b.y) <= c.y <= max(a.y, b.y))


def _seg_intersect(a: PointInt, b: PointInt, c: PointInt, d: PointInt) -> bool:
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True  # proper crossing
    # touching or collinear overlap
    if o1 == 0 and _on_seg(a, b, c):
        return True
    if o2 == 0 and _on_seg(a, b, d):
        return True
    if o3 == 0 and _on_seg(c, d, a):
        return True
    if o4 == 0 and _on_seg(c, d, b):
        return True
    return False


def _edges(poly: List[PointInt]):
    n = len(poly)
    last = n - 1
    # treat as closed ring; ignore duplicate last==first if present
    m = n - 1 if n >= 2 and poly[0] == poly[-1] else n
    for i in range(m):
        a = poly[i]
        b = poly[(i + 1) % m]
        if a != b:
            yield a, b


class PIP(Enum):
    OUT = 0
    IN = 1
    EDGE = 2
    VERTEX = 3


def _point_in_polygon(q: PointInt, poly: List[PointInt]) -> PIP:
    # Ray-cast on +x and -x to detect edge vs inside robustly (integer math)
    n = len(poly)
    if n == 0:
        return PIP.OUT
    # shift by q to test against origin
    px = [p.x - q.x for p in poly]
    py = [p.y - q.y for p in poly]
    r = l = 0
    for i in range(n):
        if px[i] == 0 and py[i] == 0:
            return PIP.VERTEX
        j = (i - 1) % n
        if (py[i] > 0) != (py[j] > 0):
            num = px[i] * py[j] - px[j] * py[i]
            den = py[j] - py[i]
            x = num / den
            if x > 0:
                r += 1
        if (py[i] < 0) != (py[j] < 0):
            num = px[i] * py[j] - px[j] * py[i]
            den = py[j] - py[i]
            x = num / den
            if x < 0:
                l += 1
    if (r & 1) != (l & 1):
        return PIP.EDGE
    return PIP.IN if (r & 1) else PIP.OUT


def classify_polygons(A: List[PointInt], B: List[PointInt]) -> PolyRelation:
    if len(A) < 3 or len(B) < 3:
        # degenerate: fall back to segment tests only
        for a1, a2 in _edges(A):
            for b1, b2 in _edges(B):
                if _seg_intersect(a1, a2, b1, b2):
                    return PolyRelation.INTERSECT
        return PolyRelation.DISJOINT

    # 1) bbox reject
    ax0, ay0, ax1, ay1 = _bbox(A)
    bx0, by0, bx1, by1 = _bbox(B)
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return PolyRelation.DISJOINT

    # 2) edge intersection (including touching)
    for a1, a2 in _edges(A):
        for b1, b2 in _edges(B):
            if _seg_intersect(a1, a2, b1, b2):
                return PolyRelation.INTERSECT

    # 3) containment (touch counts as intersect)
    pipAB = _point_in_polygon(A[0], B)
    if pipAB == PIP.IN:
        return PolyRelation.A_INSIDE_B
    if pipAB in (PIP.EDGE, PIP.VERTEX):
        return PolyRelation.INTERSECT

    pipBA = _point_in_polygon(B[0], A)
    if pipBA == PIP.IN:
        return PolyRelation.B_INSIDE_A
    if pipBA in (PIP.EDGE, PIP.VERTEX):
        return PolyRelation.INTERSECT

    return PolyRelation.DISJOINT
