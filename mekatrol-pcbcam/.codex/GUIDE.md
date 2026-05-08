# Codex Project Guide

This repository is a Qt6 desktop application for PCB CAM workflows. Keep changes
direct, small, and easy to reason about.

## Cavemane Style

- Use caveman-simple language in code, UI text, commit messages, and comments.
- Prefer boring, explicit control flow over clever abstraction.
- Name things for what they are in the PCB CAM domain.
- Do one thing at a time. Make the next step obvious.
- Comments should explain why something is necessary, not restate the code.
- Avoid speculative framework code, premature extension points, and generic
  helper layers.

## Python Style

- Read and follow `pyproject.toml` before making Python changes.
- Use Ruff as the formatter and linter.
- Preserve Ruff's current formatter behavior:
  - `skip-magic-trailing-comma = true`
  - no magic trailing commas or dangling commas introduced just for formatting
- Unused imports are not allowed.
- Keep one top-level class per file, including dataclasses.
- Do not introduce new Python files that contain multiple classes.
- If an existing file already has more than one class, do not add more classes
  there. Split only when it is needed for the task.
- Prefer typed, narrow functions over large utility functions with loose input.
- Keep file I/O at the edge of the app. Pass parsed values through domain
  objects and simple collections.

## UI Style

- This app uses PySide6/Qt widgets. Follow existing widget construction patterns
  in `mekatrol_pcbcam/main_window.py` and the preview widgets.
- Do not use spin controls for floating-point entry. Use validated text inputs.
- Validate user-entered numeric values before writing them into project state.
- Avoid gradient colors in icons, graphics, and UI styling unless explicitly
  requested.
- Keep controls dense, work-focused, and predictable. This is a CAM tool, not a
  marketing page.
- Keep preview interaction responsive. Expensive parsing or generation should
  not be hidden inside paint paths.
- Use existing theme objects and named theme fields instead of hard-coded colors.

## Architecture

- `main.py` is the entry point.
- `mekatrol_pcbcam/app.py` owns application startup, config loading, theme
  loading, splash behavior, logging, and creation of `MainWindow`.
- `mekatrol_pcbcam/main_window.py` coordinates the wizard workflow, menus,
  sidebar pages, preview widgets, generated outputs, and persistence hooks.
- `mekatrol_pcbcam/pcb_project.py` is the mutable workflow state model. Its step
  constants, completion state, dirty state, selected files, tool choices, layer
  assignments, alignment holes, edge cuts, and generated output paths define the
  project lifecycle.
- Parser modules turn external files into app data:
  - `gerber_file_parser.py`
  - `excellon_file_parser.py`
  - `gcode_parser.py`
- Preview modules render user-facing geometry:
  - `pcb_preview_widget.py`
  - `mirror_preview_widget.py`
  - `viewer.py`
- CAM and validation modules produce or check manufacturing paths:
  - `cam_generator.py`
  - `edge_cut_validator.py`
  - `edge_cut_profile.py`
  - `toolpath_document.py`
- Configuration, theme, and persistence helpers live in focused modules such as
  `app_config.py`, `theme.py`, `theme_info.py`, `tool_library.py`,
  `file_locations.py`, and `ui_save_state.py`.

## Data And Workflow Rules

- Treat `PcbProject` as the source of truth for wizard state.
- When project data changes, invalidate downstream steps from the step where the
  change matters.
- Normalize paths with `Path.resolve()` before storing or comparing selected
  Gerber, drill, tool library, and output paths.
- Keep generated output paths in `project.generated_outputs`.
- Keep layer roles stable:
  - `front_copper`
  - `back_copper`
  - `edges`
- Keep tool roles stable:
  - `drilling`
  - `milling`
  - `v_bits`
- Preserve project file compatibility. If persisted project shape changes,
  update `PcbProject.VERSION` and migration/load handling together.

## Error Handling

- Show actionable UI messages for user-fixable problems.
- Log details useful for diagnosis, but do not make normal user mistakes look
  like crashes.
- Validate malformed Gerber, Excellon, YAML, and G-code input defensively.
- Prefer returning structured results for validation and generation outcomes.

## Assets

- SVG assets live under `assets/`.
- Keep generated splash or icon assets reproducible through scripts under
  `tools/` where practical.
- Do not replace branded assets as part of unrelated code changes.

## Verification

- For Python formatting and lint checks, use Ruff.
- For app smoke testing, run:

```bash
python main.py
```

- Exercise the wizard step touched by the change.
- For parser or generator changes, test with representative Gerber, Excellon,
  or G-code input when available.
- For UI changes, check that text fits, controls do not overlap, and preview
  widgets still respond to normal interactions.
