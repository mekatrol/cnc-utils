#!/usr/bin/python3
#
# Adapted from: Matthias Wandel August 2025 (https://github.com/Matthias-Wandel/Gerber2nc)
"""CLI entry point for converting simple KiCad Gerber exports into CNC G-code.

This script is the high-level manufacturing pipeline:
1. Find the copper, board-outline, and plated-hole files exported by KiCad.
2. Parse them into geometry that describes where copper must be preserved,
   where the board edge is, and where drill hits belong.
3. Re-base everything into a local machine-friendly coordinate system.
4. Expand the copper features outward to generate isolation-routing passes.
5. Show a preview so the operator can catch import mistakes before cutting.
6. Emit G-code that first isolates traces, then marks the board outline, and
   finally drills the holes.

The comments in this file focus on the PCB fabrication flow rather than on
Python syntax so a reader can understand why each stage exists in the job.
"""

from pathlib import Path

import argparse
import yaml

from gerber2nc_lib.board_context import BoardContext
from gerber2nc_lib.drillfile_parser import DrillfileParser
from gerber2nc_lib.gcode_generator import GcodeGenerator
from gerber2nc_lib.gerber_edge_cuts_parser import GerberEdgeCutsParser
from gerber2nc_lib.gerber_traces_parser import GerberTracesParser
from gerber2nc_lib.height_map_compensation import (
    apply_height_map_to_gcode,
    build_adjusted_output_path,
    parse_compensation_config,
    validate_compensation_config,
)
from gerber2nc_lib.output_visualizer import OutputVisualizer
from gerber2nc_lib.shapely_bases import ShapelyBases


DEFAULT_CONFIG_PATH = Path(__file__).with_name("gerber2nc.yaml")
DEFAULT_CONFIG = {
    "input": {
        "project": "../hardware/triac_fet_module/triac_fet_module",
        "gerber_dir": "gerber",
    },
    "output": {
        "outname": "",
        "nc_dir": "nc",
    },
    "milling": {
        "isolation_offset_distance": 0.1,
        "isolation_num_passes": 3,
        "isolation_path_spacing": 0.1,
    },
    "drilling": {
        "available_drill_bits": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        "pcb_thickness": 1.6,
        "breakthrough_depth": 0.2,
        "safe_height": 3.0,
        "start_height": 0.1,
        "spindle_speed": 12000,
        "plunge_feed_rate": 200,
        "peck_depth": 0.6,
        "chip_clear_height": 1.0,
        "mill_tool_diameter": 0.3,
        "mill_feed_rate": 300,
        "mill_plunge_feed_rate": 150,
        "mill_depth_step": 0.4,
        "mill_start_height": 0.1,
        "max_enlarging_step": 0.2,
        "circle_segment_count": 24,
    },
    "height_map": {
        "enabled": False,
        "input": "../height-map/heightmap.csv",
        "output_suffix": ".height-adjusted",
        "max_compensated_z": 1.0,
        "max_segment_length": 1.0,
    },
    "visualisation": {
        "preview_enabled": False,
        "background_with_edgecuts_rgb": "#202020",
        "background_without_edgecuts_rgb": "#FF00FF",
        "edgecuts_fill_rgb": "#FF00FF",
        "edgecuts_outline_rgb": "#FFFF00",
        "trace_copper_rgb": "#662222",
        "region_copper_rgb": "#662222",
        "pad_fill_rgb": "#0000FF",
        "pad_outline_rgb": "#ADD8E6",
        "toolpath_rgb": "#FFFFFF",
        "hole_fill_rgb": "#000000",
        "hole_outline_rgb": "#FFFFFF",
    },
}


def find_gerber_base_names(gerber_base_name: str) -> list[str]:
    # A KiCad project can export several boards into one Gerber directory. If
    # the user points us at the wrong base name, this helper lists the names we
    # can actually see so the failure mode is easier to diagnose.
    gerber_dir = Path(gerber_base_name).parent
    suffixes = ("-F_Cu.gbr", "-B_Cu.gbr", "-Edge_Cuts.gbr", "-PTH.drl")
    base_names: set[str] = set()

    if not gerber_dir.is_dir():
        return []

    for path in gerber_dir.iterdir():
        if not path.is_file():
            continue
        for suffix in suffixes:
            if path.name.endswith(suffix):
                base_names.add(path.name[: -len(suffix)])
                break

    return sorted(base_names)


def load_gerber_traces(gerber_base_name: str, context: BoardContext) -> GerberTracesParser:
    front_copper = Path(gerber_base_name + "-F_Cu.gbr")
    back_copper = Path(gerber_base_name + "-B_Cu.gbr")

    # The current toolpath generator assumes a one-sided workflow: isolate one
    # copper layer, drill holes, and mark the board perimeter. Prefer front
    # copper when present, but allow a back-copper-only project as a fallback.
    if front_copper.exists():
        print(f"Using front copper file '{front_copper}'")
        return GerberTracesParser(str(front_copper), context)

    if back_copper.exists():
        print(f"Using back copper file '{back_copper}'")
        return GerberTracesParser(str(back_copper), context)

    print(
        f"No copper file found at '{front_copper}' or '{back_copper}', "
        "continuing without traces"
    )
    return GerberTracesParser(str(front_copper), context, optional=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make G-code from a simple one-sided KiCad PCB layout by converting "
            "Gerber copper, edge cuts, and drill data into CNC toolpaths."
        )
    )
    parser.add_argument(
        "project",
        nargs="?",
        help=(
            "KiCad project base path without suffixes, for example /path/to/myboard. "
            "Defaults to input.project from the config file."
        ),
    )
    parser.add_argument(
        "gerber_dir",
        nargs="?",
        help=(
            "Gerber subdirectory under the project path. Defaults to "
            "input.gerber_dir from the config file."
        ),
    )
    parser.add_argument(
        "outname",
        nargs="?",
        help=(
            "Optional output base name. Defaults to output.outname from the "
            "config file, then to the project base file name if blank."
        ),
    )
    parser.add_argument(
        "nc_dir",
        nargs="?",
        help=(
            "Output directory for generated G-code. Defaults to output.nc_dir "
            "from the config file."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "YAML config file for machining settings, defaults to "
            f"{DEFAULT_CONFIG_PATH.name} next to this script"
        ),
    )
    return parser.parse_args()


def resolve_project_paths(project_arg: str, gerber_dir: str) -> tuple[str, str, str]:
    # Users may supply either `/path/to/project_dir` or `/path/to/project_base`.
    # Both resolve to the same Gerber naming convention KiCad uses:
    #   <project_dir>/<gerber_dir>/<project_name>-F_Cu.gbr
    #   <project_dir>/<gerber_dir>/<project_name>-Edge_Cuts.gbr
    # etc.
    project_path = Path(project_arg)

    if project_path.is_dir():
        project_dir = project_path
        project_name = project_dir.name
        project_base_name = str(project_dir / project_name)
    else:
        project_base = Path(project_arg)
        project_dir = project_base.parent
        project_name = project_base.name
        project_base_name = str(project_base)

    gerber_base_name = str(project_dir / gerber_dir / project_name)
    return project_base_name, project_name, gerber_base_name


def load_config(config_path: str) -> dict:
    # YAML provides a simple human-editable place to move machine/process
    # settings out of the code. Start with the isolation-routing parameters so
    # the operator can tune copper clearances without editing Python.
    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Config file '{resolved_path}' does not exist. "
            f"Create it from '{DEFAULT_CONFIG_PATH.name}' or pass --config."
        )

    with resolved_path.open("r", encoding="utf-8") as config_file:
        loaded_config = yaml.safe_load(config_file) or {}

    if not isinstance(loaded_config, dict):
        raise ValueError(
            f"Config file '{resolved_path}' must contain a YAML mapping of settings."
        )

    # Merge with defaults so omitted keys remain predictable as more settings
    # move into the config file over time.
    config = {**DEFAULT_CONFIG, **loaded_config}
    if "input" in loaded_config and isinstance(loaded_config["input"], dict):
        config["input"] = {**DEFAULT_CONFIG["input"], **loaded_config["input"]}
    if "output" in loaded_config and isinstance(loaded_config["output"], dict):
        config["output"] = {**DEFAULT_CONFIG["output"], **loaded_config["output"]}
    if "milling" in loaded_config and isinstance(loaded_config["milling"], dict):
        config["milling"] = {**DEFAULT_CONFIG["milling"], **loaded_config["milling"]}
    if "drilling" in loaded_config and isinstance(loaded_config["drilling"], dict):
        config["drilling"] = {
            **DEFAULT_CONFIG["drilling"],
            **loaded_config["drilling"],
        }
    if "height_map" in loaded_config and isinstance(loaded_config["height_map"], dict):
        config["height_map"] = {
            **DEFAULT_CONFIG["height_map"],
            **loaded_config["height_map"],
        }
    if "visualisation" in loaded_config and isinstance(
        loaded_config["visualisation"], dict
    ):
        config["visualisation"] = {
            **DEFAULT_CONFIG["visualisation"],
            **loaded_config["visualisation"],
        }
    validate_config(config, resolved_path)
    return config


def resolve_path_from_config(base_dir: Path, configured_path: str) -> Path:
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def validate_config(config: dict, config_path: Path) -> None:
    # Keep validation explicit and close to the settings for now. This makes
    # the permitted range of each machining parameter obvious and gives clear
    # operator-facing errors when a config is unsafe or nonsensical.
    input_config = config.get("input")
    if not isinstance(input_config, dict):
        raise ValueError(f"'{config_path}': input must be a YAML mapping.")

    project = input_config.get("project")
    if not isinstance(project, str) or not project.strip():
        raise ValueError(
            f"'{config_path}': input.project must be a non-empty project path string."
        )

    gerber_dir = input_config.get("gerber_dir")
    if not isinstance(gerber_dir, str) or not gerber_dir.strip():
        raise ValueError(
            f"'{config_path}': input.gerber_dir must be a non-empty directory name."
        )

    output_config = config.get("output")
    if not isinstance(output_config, dict):
        raise ValueError(f"'{config_path}': output must be a YAML mapping.")

    outname = output_config.get("outname")
    if not isinstance(outname, str):
        raise ValueError(f"'{config_path}': output.outname must be a string.")

    nc_dir = output_config.get("nc_dir")
    if not isinstance(nc_dir, str) or not nc_dir.strip():
        raise ValueError(
            f"'{config_path}': output.nc_dir must be a non-empty directory name."
        )

    milling = config.get("milling")
    if not isinstance(milling, dict):
        raise ValueError(f"'{config_path}': milling must be a YAML mapping.")

    isolation_offset_distance = milling.get("isolation_offset_distance")
    if not isinstance(isolation_offset_distance, (int, float)):
        raise ValueError(
            f"'{config_path}': milling.isolation_offset_distance must be a number in mm."
        )
    if isolation_offset_distance <= 0:
        raise ValueError(
            f"'{config_path}': milling.isolation_offset_distance must be greater than 0."
        )

    isolation_num_passes = milling.get("isolation_num_passes")
    if not isinstance(isolation_num_passes, int) or isinstance(
        isolation_num_passes, bool
    ):
        raise ValueError(
            f"'{config_path}': milling.isolation_num_passes must be an integer."
        )
    if isolation_num_passes < 1:
        raise ValueError(
            f"'{config_path}': milling.isolation_num_passes must be at least 1."
        )

    isolation_path_spacing = milling.get("isolation_path_spacing")
    if not isinstance(isolation_path_spacing, (int, float)):
        raise ValueError(
            f"'{config_path}': milling.isolation_path_spacing must be a number in mm."
        )
    if isolation_path_spacing <= 0:
        raise ValueError(
            f"'{config_path}': milling.isolation_path_spacing must be greater than 0."
        )

    drilling = config.get("drilling")
    if not isinstance(drilling, dict):
        raise ValueError(f"'{config_path}': drilling must be a YAML mapping.")

    available_drill_bits = drilling.get("available_drill_bits")
    if not isinstance(available_drill_bits, list) or not available_drill_bits:
        raise ValueError(
            f"'{config_path}': drilling.available_drill_bits must be a non-empty list of bit diameters in mm."
        )
    for index, diameter in enumerate(available_drill_bits):
        validate_positive_number(
            diameter,
            f"drilling.available_drill_bits[{index}]",
            config_path,
        )

    validate_positive_number(
        drilling.get("pcb_thickness"),
        "drilling.pcb_thickness",
        config_path,
    )
    validate_non_negative_number(
        drilling.get("breakthrough_depth"),
        "drilling.breakthrough_depth",
        config_path,
    )
    validate_positive_number(
        drilling.get("safe_height"),
        "drilling.safe_height",
        config_path,
    )
    validate_non_negative_number(
        drilling.get("start_height"),
        "drilling.start_height",
        config_path,
    )
    validate_positive_number(
        drilling.get("spindle_speed"),
        "drilling.spindle_speed",
        config_path,
    )
    validate_positive_number(
        drilling.get("plunge_feed_rate"),
        "drilling.plunge_feed_rate",
        config_path,
    )
    validate_positive_number(
        drilling.get("peck_depth"),
        "drilling.peck_depth",
        config_path,
    )
    validate_non_negative_number(
        drilling.get("chip_clear_height"),
        "drilling.chip_clear_height",
        config_path,
    )
    validate_positive_number(
        drilling.get("mill_tool_diameter"),
        "drilling.mill_tool_diameter",
        config_path,
    )
    validate_positive_number(
        drilling.get("mill_feed_rate"),
        "drilling.mill_feed_rate",
        config_path,
    )
    validate_positive_number(
        drilling.get("mill_plunge_feed_rate"),
        "drilling.mill_plunge_feed_rate",
        config_path,
    )
    validate_positive_number(
        drilling.get("mill_depth_step"),
        "drilling.mill_depth_step",
        config_path,
    )
    validate_non_negative_number(
        drilling.get("mill_start_height"),
        "drilling.mill_start_height",
        config_path,
    )
    validate_positive_number(
        drilling.get("max_enlarging_step"),
        "drilling.max_enlarging_step",
        config_path,
    )

    circle_segment_count = drilling.get("circle_segment_count")
    if not isinstance(circle_segment_count, int) or isinstance(
        circle_segment_count, bool
    ):
        raise ValueError(
            f"'{config_path}': drilling.circle_segment_count must be an integer."
        )
    if circle_segment_count < 8:
        raise ValueError(
            f"'{config_path}': drilling.circle_segment_count must be at least 8."
        )

    validate_compensation_config(config, config_path)

    visualisation = config.get("visualisation")
    if not isinstance(visualisation, dict):
        raise ValueError(f"'{config_path}': visualisation must be a YAML mapping.")

    preview_enabled = visualisation.get("preview_enabled")
    if not isinstance(preview_enabled, bool):
        raise ValueError(
            f"'{config_path}': visualisation.preview_enabled must be true or false."
        )

    for colour_name in DEFAULT_CONFIG["visualisation"]:
        if colour_name == "preview_enabled":
            continue
        validate_hex_colour(
            visualisation.get(colour_name),
            f"visualisation.{colour_name}",
            config_path,
        )


def validate_hex_colour(value: object, setting_name: str, config_path: Path) -> None:
    # Visualisation colours are stored as hex strings so the config stays
    # compact while still being explicit about the exact rendered colour.
    if not isinstance(value, str):
        raise ValueError(
            f"'{config_path}': {setting_name} must be a string like '#RRGGBB'."
        )
    if len(value) != 7 or not value.startswith("#"):
        raise ValueError(
            f"'{config_path}': {setting_name} must be in '#RRGGBB' format."
        )
    try:
        int(value[1:], 16)
    except ValueError as exc:
        raise ValueError(
            f"'{config_path}': {setting_name} must be in '#RRGGBB' format."
        ) from exc


def validate_positive_number(
    value: object, setting_name: str, config_path: Path
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"'{config_path}': {setting_name} must be a number.")
    if value <= 0:
        raise ValueError(f"'{config_path}': {setting_name} must be greater than 0.")


def validate_non_negative_number(
    value: object, setting_name: str, config_path: Path
) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"'{config_path}': {setting_name} must be a number.")
    if value < 0:
        raise ValueError(
            f"'{config_path}': {setting_name} must be greater than or equal to 0."
        )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config_dir = Path(args.config).resolve().parent
    input_config = config["input"]
    output_config = config["output"]
    milling_config = config["milling"]
    drilling_config = config["drilling"]
    height_map_config = parse_compensation_config(config)
    visualisation_config = config["visualisation"]

    project_arg = (
        args.project
        if args.project is not None
        else str(resolve_path_from_config(config_dir, input_config["project"]))
    )
    gerber_dir = args.gerber_dir or input_config["gerber_dir"]
    configured_outname = output_config["outname"]
    requested_outname = args.outname if args.outname is not None else configured_outname
    nc_dir = (
        args.nc_dir
        if args.nc_dir is not None
        else str(resolve_path_from_config(config_dir, output_config["nc_dir"]))
    )

    # Normalize path separators first so mixed Windows-style input still maps
    # cleanly onto pathlib operations when the script is run on Unix.
    project_base_name, project_name, gerber_base_name = resolve_project_paths(
        project_arg.replace("\\", "/"), gerber_dir
    )
    outname = f"{(requested_outname or project_name)}.nc"

    # BoardContext is the shared "world bounds" object. Every parser expands
    # it as features are discovered, then later stages use those extents to
    # shift the job into a local origin for both previewing and machining.
    context = BoardContext(base_name=project_base_name)

    # Import the three physical feature classes used by this workflow:
    # copper to preserve, board edges to respect, and drill hits to perform.
    gerber_traces = load_gerber_traces(gerber_base_name, context)
    gerber_edgecuts = GerberEdgeCutsParser(gerber_base_name + "-Edge_Cuts.gbr", context)
    drilldata = DrillfileParser(gerber_base_name + "-PTH.drl", context)

    if (
        not gerber_traces.traces
        and not gerber_traces.pads
        and not gerber_edgecuts.outline
        and not drilldata.holes
    ):
        available_bases = find_gerber_base_names(gerber_base_name)
        print(
            "No input geometry was loaded. "
            f"Checked for files based on '{gerber_base_name}'."
        )
        if available_bases:
            print("Available Gerber base names:", ", ".join(available_bases))
        return 1

    # Gerber files usually carry their own absolute project coordinates. CNC
    # jobs are easier to reason about if the lower-left used extent becomes
    # approximately (0, 0), so every geometry source is shifted by the common
    # minimum bounds discovered during parsing.
    gerber_traces.shift(context.x_min, context.y_min)
    gerber_edgecuts.shift(context.x_min, context.y_min)
    drilldata.shift(context.x_min, context.y_min)

    # Isolation routing removes copper around the intended traces instead of
    # "drawing" the traces themselves. The trace parser gives us the copper to
    # keep; Shapely expands that copper footprint outward and uses the expanded
    # boundaries as routing centerlines.
    #
    # `isolation_offset_distance` is the first clearance pass away from the
    # copper edge.
    # `isolation_num_passes` determines how many widening loops to cut.
    # `isolation_path_spacing` sets how much farther out each successive
    # cleanup pass is.
    isolation_offset_distance = float(milling_config["isolation_offset_distance"])
    isolation_num_passes = int(milling_config["isolation_num_passes"])
    isolation_path_spacing = float(milling_config["isolation_path_spacing"])

    shapely_bases = ShapelyBases(gerber_traces)
    trace_mill_geometry = shapely_bases.compute_trace_toolpaths(
        isolation_offset_distance, isolation_num_passes, isolation_path_spacing
    )

    # Preview before cutting. CAM mistakes are expensive once the spindle is in
    # the material, so the Tk window acts as a manual sanity check for layer
    # orientation, missing pads, bad outlines, and obviously wrong offsets.
    if visualisation_config["preview_enabled"]:
        visualizer = OutputVisualizer(project_base_name, context, visualisation_config)
        visualizer.load_trace_geometries(gerber_traces)
        visualizer.load_trace_mill_geometry(trace_mill_geometry)
        visualizer.load_edge_cut_geometry(gerber_edgecuts.outline)
        visualizer.load_holes(drilldata.holes)
        visualizer.create_tkinter_visualization()

    # The generator uses the final board height to park the tool above the
    # upper edge of the rebased workpiece when the program ends.
    board_height = context.y_max - context.y_min
    gcode = GcodeGenerator(board_height, drilling_config)
    output_path = Path(nc_dir) / outname
    gcode.output_gcode(
        str(output_path),
        gerber_edgecuts.outline,
        trace_mill_geometry,
        drilldata.holes,
    )

    if height_map_config.enabled:
        if not height_map_config.map_path.strip():
            raise ValueError(
                "height_map.enabled is true, but height_map.input is empty."
            )
        height_map_path = resolve_path_from_config(
            config_dir, height_map_config.map_path
        )
        adjusted_output_path = build_adjusted_output_path(
            output_path, height_map_config.output_suffix
        )
        apply_height_map_to_gcode(
            output_path,
            adjusted_output_path,
            height_map_path,
            max_compensated_z=height_map_config.max_compensated_z,
            max_segment_length=height_map_config.max_segment_length,
        )
        print(
            "Height-map-adjusted G-code generated in '%s'"
            % adjusted_output_path
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
