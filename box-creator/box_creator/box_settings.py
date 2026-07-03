from dataclasses import dataclass


@dataclass
class BoxSettings:
    job_name: str = "finger-box"
    box_kind: str = "box"
    size_x: float = 160.0
    size_y: float = 100.0
    size_z: float = 70.0
    material_thickness: float = 6.0
    stock_width: float = 600.0
    stock_height: float = 400.0
    bit_diameter: float = 3.175
    finger_width: float = 12.0
    include_tabs: bool = True
    tab_width: float = 8.0
    tab_height: float = 1.5
    relief_diameter: float = 3.175
    cut_depth_step: float = 1.5
    safe_height: float = 5.0
    surface_height: float = 0.5
    feed_rate: int = 650
    plunge_rate: int = 180
    spindle_speed: int = 18000
    layout_gap: float = 14.0

    @property
    def final_cut_depth(self) -> float:
        return -(self.material_thickness + 0.35)
