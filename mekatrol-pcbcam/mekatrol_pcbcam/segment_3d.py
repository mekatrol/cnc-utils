from dataclasses import dataclass

from .point_3d import Point3D


@dataclass(frozen=True)
class Segment3D:
    start: Point3D
    end: Point3D
    rapid: bool
    line_number: int
    source: str
