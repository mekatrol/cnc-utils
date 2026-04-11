# mekatrol-pcbcam

`mekatrol-pcbcam` is a Qt6 desktop viewer for CNC `.nc` / G-code files. The initial skeleton focuses on:

- loading generated `.nc` files
- viewing motion in a simple interactive 3D viewport
- providing a branded splash screen during application startup

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Controls

- Left mouse drag: orbit
- Right mouse drag: pan
- Mouse wheel: zoom
- `F`: fit loaded toolpath to view
