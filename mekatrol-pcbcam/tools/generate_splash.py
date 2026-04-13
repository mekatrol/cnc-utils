from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH = 1280
HEIGHT = 720
BLUE = "#0067cf"
GREEN = "#3dd30b"
WHITE = "#ffffff"
TRANSPARENT = (0, 0, 0, 0)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "splash-legacy.png"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_BOLD, size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_board_icon(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    ox, oy = origin
    body = [
        (ox + 0, oy + 40),
        (ox + 0, oy + 220),
        (ox + 60, oy + 280),
        (ox + 220, oy + 280),
        (ox + 220, oy + 210),
        (ox + 285, oy + 210),
        (ox + 285, oy + 95),
        (ox + 185, oy + 0),
        (ox + 45, oy + 0),
    ]
    draw.polygon(body, fill=GREEN, outline=BLUE, width=8)

    pads = [
        (ox + 30, oy + 28, 26),
        (ox + 138, oy + 58, 24),
        (ox + 206, oy + 104, 28),
        (ox + 38, oy + 136, 22),
    ]
    for px, py, radius in pads:
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=WHITE, outline=BLUE, width=6)
        inner = max(6, radius // 3)
        draw.ellipse((px - inner, py - inner, px + inner, py + inner), fill=BLUE)

    hub_center = (ox + 135, oy + 154)
    draw.line((ox + 30, oy + 28, ox + 104, oy + 102), fill=BLUE, width=14)
    draw.line((ox + 138, oy + 58, ox + 138, oy + 128), fill=BLUE, width=14)
    draw.line((ox + 206, oy + 104, ox + 164, oy + 138), fill=BLUE, width=14)
    draw.line((ox + 38, oy + 136, ox + 105, oy + 188), fill=BLUE, width=14)

    draw.ellipse((hub_center[0] - 56, hub_center[1] - 56, hub_center[0] + 56, hub_center[1] + 56), fill=WHITE, outline=BLUE, width=8)
    draw.ellipse((hub_center[0] - 24, hub_center[1] - 24, hub_center[0] + 24, hub_center[1] + 24), fill=BLUE)

    for offset in (168, 200, 232):
        draw.line((hub_center[0] + 56, offset + oy - 40, ox + 284, offset + oy - 40), fill=BLUE, width=12)


def _draw_glow(base: Image.Image) -> Image.Image:
    glow = base.filter(ImageFilter.GaussianBlur(radius=6))
    faded = Image.new("RGBA", base.size, TRANSPARENT)
    faded.alpha_composite(glow)
    return faded


def main() -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), TRANSPARENT)
    glow_layer = Image.new("RGBA", (WIDTH, HEIGHT), TRANSPARENT)
    glow_draw = ImageDraw.Draw(glow_layer)
    main_draw = ImageDraw.Draw(image)

    _draw_board_icon(glow_draw, (145, 205))
    _draw_board_icon(main_draw, (145, 205))

    title_font = _font(116)
    subtitle_font = _font(150)

    glow_draw.text((445, 182), "MEKATROL", font=title_font, fill=BLUE)
    glow_draw.text((495, 328), "PCBCAM", font=subtitle_font, fill=GREEN)
    main_draw.text((445, 182), "MEKATROL", font=title_font, fill=BLUE)
    main_draw.text((495, 328), "PCBCAM", font=subtitle_font, fill=GREEN)

    image = Image.alpha_composite(_draw_glow(glow_layer), image)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)


if __name__ == "__main__":
    main()
