# gerber2nc

Adapted from: [Gerber2nc Matthias Wandel <April 2026>](<https://github.com/Matthias-Wandel/Gerber2nc>)

Python tool for converting a simple one-sided KiCad PCB export into CNC G-code.

This project is adapted from Matthias Wandel's `Gerber2nc` and is currently structured as a small Python package with a script entrypoint, a local `.venv`, and VS Code debug configuration.

## What It Does

The tool:

- reads KiCad-generated Gerber copper data from the front copper layer
- reads KiCad edge cuts to outline the board
- reads plated through-hole drill data
- computes isolation milling toolpaths around traces and pads
- previews the board and toolpaths in a Tkinter window
- writes G-code for trace isolation, edge marking, peck drilling, and optional hole enlargement milling
- can optionally write a second G-code file with height-map-based Z compensation applied

## Current Assumptions And Limitations

The code is intended for simple boards and currently assumes:

- single-sided front copper only
- traces, pads, and holes only
- no rotated pads
- KiCad-style file naming
- a simple edge-cut outline flow expected by the parser

Optional files:

- missing edge-cuts data is allowed
- missing drill data is allowed

## Expected Input Files

The script expects a KiCad project base path without suffixes, for example:

```text
/path/to/triac_fet_module/triac_fet_module
```

It then derives input file names using the board name from that base path and opens them from the Gerber directory:

- `<gerber_dir>/<project_name>-F_Cu.gbr`
- `<gerber_dir>/<project_name>-Edge_Cuts.gbr`
- `<gerber_dir>/<project_name>-PTH.drl`

Example:

```text
project:    ../hardware/triac_fet_module/triac_fet_module
gerber_dir: gerber
```

Resolved inputs:

- `../hardware/triac_fet_module/gerber/triac_fet_module-F_Cu.gbr`
- `../hardware/triac_fet_module/gerber/triac_fet_module-Edge_Cuts.gbr`
- `../hardware/triac_fet_module/gerber/triac_fet_module-PTH.drl`

## Output

Generated G-code is written to the configured output directory, which defaults to `nc/`.

Example output:

```text
nc/triac_fet_module.nc
```

The output directory is created automatically if it does not already exist.

## Machining Config

The YAML config now has separate sections for:

- `milling`: isolation-routing settings
- `drilling`: hole drilling and hole enlargement settings
- `height_map`: optional post-processing settings for Z compensation from a probed surface map
- `visualisation`: preview colours

In this `cnc-utils` copy, `visualisation.preview_enabled` defaults to `false` so
the tool can run in headless or remote debug sessions without a Tk display. Set
it to `true` when you want the preview window.

The `drilling` section controls:

- available drill diameters the planner may choose from
- PCB thickness and breakthrough depth
- peck drilling depth and chip-clear retract height
- mill diameter, mill depth step, and radial enlargement step for oversize holes

Hole planning works like this:

1. For each Excellon hole diameter, the program picks the largest configured drill bit that does not exceed the requested hole size.
2. If that drill exactly matches the requested hole, the hole is finished by peck drilling alone.
3. If the drill is smaller than the requested hole, the code drills first and then enlarges the hole with the configured milling cutter in radial passes at each Z depth until the finished diameter is reached.

## Height-Map Compensation

The recommended integration is a post-process, not an in-generator coordinate tweak:

- `gerber2nc.py` always writes the original NC program first
- when `height_map.enabled: true`, it then writes a second file with `height_map.output_suffix` added before `.nc`
- the original file remains available for comparison in NC viewers

The height-map CSV is expected to come from the `height-map` probing tool and contain:

```text
ix,iy,x_mm,y_mm,z_mm
```

Compensation behavior:

- bilinear interpolation is used between sampled probe points
- XY cutting moves are split into shorter segments so the generated Z can follow the interpolated surface
- only moves at or below `height_map.max_compensated_z` are adjusted, so high clearance rapids remain unchanged

Example config:

```yaml
height_map:
  enabled: true
  input: "../height-map/heightmap.csv"
  output_suffix: ".height-adjusted"
  max_compensated_z: 1.0
  max_segment_length: 1.0
```

In the `cnc-utils` repository layout, that default `input` path points at the sibling
[`height-map`](../height-map) project.

If your normal output is `nc/board.nc`, the compensated file becomes:

```text
nc/board.height-adjusted.nc
```

## Installation

Create and activate the local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dependencies

Runtime dependency:

- `shapely`

Standard library modules used include:

- `argparse`
- `pathlib`
- `re`
- `tkinter`

## Command Line Usage

```bash
python gerber2nc.py project [gerber_dir] [outname] [nc_dir]
```

Arguments:

- `project`: KiCad project base path without suffixes
- `gerber_dir`: Gerber subdirectory under the project path, default `gerber`
- `outname`: output base filename without `.nc`, defaults to the project name
- `nc_dir`: output directory for generated G-code, default `nc`

Example:

```bash
python gerber2nc.py \
  ../hardware/triac_fet_module/triac_fet_module \
  gerber \
  triac_fet_module \
  nc
```

If you want to keep the default output filename but override only the output directory, you currently need to pass an empty `outname` placeholder:

```bash
python gerber2nc.py \
  ../hardware/triac_fet_module/triac_fet_module \
  gerber \
  "" \
  nc
```

To post-process any existing NC file manually:

```bash
python apply_height_map.py nc/board.nc ../height-map/heightmap.csv
```

Or write to a specific destination:

```bash
python apply_height_map.py nc/board.nc ../height-map/heightmap.csv nc/board.probed.nc
```

Paths read from `gerber2nc.yaml` are resolved relative to that YAML file, so the
default `nc` output directory and sibling `../height-map/heightmap.csv` reference
work regardless of the shell directory you launch from.

## VS Code Debugging

The workspace includes:

- `.vscode/settings.json` to use `.venv/bin/python`
- `.vscode/launch.json` to debug `gerber2nc.py` with `F5`

The debug configuration prompts for:

- KiCad project base path
- Gerber subdirectory
- output directory

Current defaults:

- project: `../hardware/triac_fet_module/triac_fet_module`
- gerber dir: `gerber`
- nc dir: `nc`

## Project Layout

```text
gerber2nc/
├── gerber2nc.py
├── apply_height_map.py
├── gerber2nc_lib/
│   ├── board_context.py
│   ├── drillfile_parser.py
│   ├── gcode_generator.py
│   ├── gerber_edge_cuts_parser.py
│   ├── gerber_traces_parser.py
│   ├── height_map_compensation.py
│   ├── output_visualizer.py
│   └── shapely_bases.py
├── .vscode/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Module Overview

- `gerber2nc.py`: CLI entrypoint and application flow
- `board_context.py`: shared board bounds and naming context
- `gerber_traces_parser.py`: parses front copper Gerber traces and pads
- `height_map_compensation.py`: loads probe CSV data and applies interpolated Z compensation to G-code
- `gerber_edge_cuts_parser.py`: parses edge-cuts geometry
- `drillfile_parser.py`: parses Excellon drill data
- `shapely_bases.py`: builds buffered geometry and isolation toolpaths
- `output_visualizer.py`: renders a Tkinter preview window
- `gcode_generator.py`: emits the final `.nc` file

## Linting

Ruff is configured in `pyproject.toml`.

Current local ignore:

- `F541` for f-strings without placeholders

## Notes

- The preview window must be closed before G-code generation completes.
- `nc/`, `.venv/`, Python cache files, and other local artifacts are excluded in this project's `.gitignore`.
