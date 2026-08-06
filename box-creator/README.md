# box-creator

`box-creator` is a PySide6 desktop wizard for generating GRBL-compatible NC
files for finger-jointed boxes and drawer trays.

It creates a flat cutting layout from outside `X`, `Y`, and `Z` dimensions,
material thickness, stock size, cutter diameter, finger width, tab settings,
and inside corner preview settings. The wizard can preview the parts in their
cutting layout or as an assembled box.

Each wizard step drives a contextual preview: box dimensions, cutter geometry,
finger joints, holding tab placement, final layout, or generated job summary.

The Generate step can build NC in memory and load it straight into a simulator
view without saving a file. Run, pause, or restart the toolpath overlay to watch
the cutter position advance in real time from the NC feed rates. The Simulation
speed slider remains live while running and changes simulated feed, spindle, and
playback speed from 10% to 400%.

Projects can be saved and reopened as `.boxcreator.json` files. File dialogs
remember their last project-open and file-save directories using the same
Mekatrol application config location pattern as `mekatrol-pcbcam`. The main
window also remembers its last screen, size, position, and maximized state.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Current CAM behavior

- Units are millimetres.
- Output uses `G21`, `G90`, `G94`, spindle start, profile passes, spindle stop,
  and `M30`.
- Finger height is always the material thickness.
- Parts are packed onto the configured stock size. When they do not all fit on
  one piece, the layout creates additional stock sheets.
- Holding tabs are only placed on the outer bounding edges of each flat panel,
  with up to two tabs per edge.
- New projects default to a drawer tray with no top panel and 4 mm holding tabs.
- Inside corners are cut as part of the same profile contour passes as the
  edges; NC output does not add separate plunge-style corner pocket operations.
- The default settings are suitable starting values for a Shapeoko 4 Pro, while
  the generated file remains plain GRBL-style G-code.
- Profile cuts use cutter-centre coordinates offset outward by half the edge-cut
  bit diameter.
- Multi-sheet output uses local coordinates for each stock sheet and inserts an
  `M0` pause before sheet 2 and later sheets so the next piece of stock can be
  loaded.
- The built-in simulator reads the generated NC text, maps stock-sheet comments
  back onto the preview layout, and times rapid and feed moves from the
  programmed rates.
