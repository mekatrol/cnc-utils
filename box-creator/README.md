# box-creator

`box-creator` is a PySide6 desktop wizard for generating GRBL-compatible NC
files for finger-jointed boxes and drawer trays.

It creates a flat cutting layout from outside `X`, `Y`, and `Z` dimensions,
material thickness, cutter diameter, finger width, tab settings, and inside
corner-relief size. The wizard can preview the parts in their cutting layout or
as an assembled box.

Each wizard step drives a contextual preview: box dimensions, cutter and relief
geometry, finger joints, holding tab placement, final layout, or generated job
summary.

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
- Holding tabs are only placed on the outer bounding edges of each flat panel.
- Corner relief pockets are cut at inset finger-joint corners and default to
  the edge-cut bit diameter. In the flat preview, the dashed outline is the
  square nominal edge before relief cutting, and the clipped orange arcs show
  the material removed by the relief bit.
- The default settings are suitable starting values for a Shapeoko 4 Pro, while
  the generated file remains plain GRBL-style G-code.
