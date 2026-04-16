from __future__ import annotations

ORIGIN_TOP_LEFT = "top_left"
ORIGIN_TOP_RIGHT = "top_right"
ORIGIN_BOTTOM_LEFT = "bottom_left"
ORIGIN_BOTTOM_RIGHT = "bottom_right"
ORIGIN_CENTER = "center"
DEFAULT_NC_ORIGIN = ORIGIN_BOTTOM_LEFT
VALID_NC_ORIGINS = {
    ORIGIN_TOP_LEFT,
    ORIGIN_TOP_RIGHT,
    ORIGIN_BOTTOM_LEFT,
    ORIGIN_BOTTOM_RIGHT,
    ORIGIN_CENTER,
}
NC_ORIGIN_LABELS: dict[str, str] = {
    ORIGIN_TOP_LEFT: "Top Left",
    ORIGIN_TOP_RIGHT: "Top Right",
    ORIGIN_BOTTOM_LEFT: "Bottom Left",
    ORIGIN_BOTTOM_RIGHT: "Bottom Right",
    ORIGIN_CENTER: "Center",
}


def normalize_nc_origin(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in VALID_NC_ORIGINS:
        return normalized
    return DEFAULT_NC_ORIGIN


def legacy_origin_point_for_bounds(
    bounds: tuple[float, float, float, float],
    origin: str,
) -> tuple[float, float]:
    normalized = normalize_nc_origin(origin)
    x_min, x_max, y_min, y_max = bounds
    if normalized == ORIGIN_TOP_LEFT:
        return x_min, y_max
    if normalized == ORIGIN_TOP_RIGHT:
        return x_max, y_max
    if normalized == ORIGIN_BOTTOM_RIGHT:
        return x_max, y_min
    if normalized == ORIGIN_CENTER:
        return (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
    return x_min, y_min


def format_origin_point(point: tuple[float, float]) -> str:
    return f"({point[0]:.3f} mm, {point[1]:.3f} mm)"
