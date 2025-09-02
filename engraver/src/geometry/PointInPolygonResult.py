from enum import Enum, auto


class PointInPolygonResult(Enum):
    Outside = auto()
    Inside = auto()
    Edge = auto()
    Vertex = auto()
