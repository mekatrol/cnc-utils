import math
import re
import numpy as np
from typing import Tuple, Optional


class Affine2D:
    """Simple 2D affine transform: 3x3 homogeneous matrix.

    Stored as numpy array with shape (3,3). Points are column vectors [x,y,1]^T.
    """

    def __init__(self, m: Optional[np.ndarray] = None):
        if m is None:
            self.m = np.eye(3, dtype=float)
        else:
            self.m = np.array(m, dtype=float).reshape(3, 3)

    def __matmul__(self, other: "Affine2D") -> "Affine2D":
        return Affine2D(self.m @ other.m)

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        v = np.array([x, y, 1.0], dtype=float)
        res = self.m @ v
        return (res[0], res[1])

    @staticmethod
    def identity() -> "Affine2D":
        return Affine2D()

    @staticmethod
    def translation(tx: float, ty: float) -> "Affine2D":
        m = np.eye(3)
        m[0, 2] = tx
        m[1, 2] = ty
        return Affine2D(m)

    @staticmethod
    def scale(sx: float, sy: Optional[float] = None) -> "Affine2D":
        if sy is None:
            sy = sx
        m = np.eye(3)
        m[0, 0] = sx
        m[1, 1] = sy
        return Affine2D(m)

    @staticmethod
    def rotation_deg(angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> "Affine2D":
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        R = np.array(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0, 0, 1]], dtype=float)
        T1 = Affine2D.translation(-cx, -cy).m
        T2 = Affine2D.translation(cx, cy).m
        return Affine2D(T2 @ R @ T1)

    @staticmethod
    def matrix(a: float, b: float, c: float, d: float, e: float, f: float) -> "Affine2D":
        m = np.array([[a, c, e], [b, d, f], [0, 0, 1]],
                     dtype=float)  # SVG's (a b c d e f)
        return Affine2D(m)

    @staticmethod
    def from_svg_transform(transform_str: str) -> "Affine2D":
        """Parse a subset of SVG transform strings."""
        if not transform_str:
            return Affine2D.identity()

        tok_re = re.compile(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)")
        transform = Affine2D.identity()
        for name, args in tok_re.findall(transform_str):
            parts = [float(p) for p in re.split(r"[ ,]+", args.strip()) if p]
            if name == "matrix" and len(parts) == 6:
                T = Affine2D.matrix(*parts)
            elif name == "translate" and len(parts) in (1, 2):
                tx = parts[0]
                ty = parts[1] if len(parts) == 2 else 0.0
                T = Affine2D.translation(tx, ty)
            elif name == "scale" and len(parts) in (1, 2):
                sx = parts[0]
                sy = parts[1] if len(parts) == 2 else None
                T = Affine2D.scale(sx, sy)
            elif name == "rotate" and len(parts) in (1, 3):
                angle = parts[0]
                if len(parts) == 3:
                    T = Affine2D.rotation_deg(angle, parts[1], parts[2])
                else:
                    T = Affine2D.rotation_deg(angle)
            else:
                continue
            transform = transform @ T
        return transform
