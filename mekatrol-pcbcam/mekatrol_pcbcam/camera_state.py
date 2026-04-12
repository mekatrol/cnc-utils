from dataclasses import dataclass
import math


@dataclass
class CameraState:
    yaw: float = math.radians(35.0)
    pitch: float = math.radians(-25.0)
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
