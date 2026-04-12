from dataclasses import dataclass

from .point_3d import Point3D


@dataclass(frozen=True)
class ToolpathStats:
    min_point: Point3D
    max_point: Point3D
    segment_count: int
    rapid_count: int
    cut_count: int
    path_length: float
