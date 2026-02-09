#!/usr/bin/env python3
"""
GRBL Z height-map probe for a rectangular area (e.g., Vevor 3018 with probe input)

Goal (your change request):
- Assume the job starts at "world" X=0, Y=0 (at the current location).
- Do a probe at world (0,0) to establish world Z=0 at the probe contact point.
- After that, use world coordinates for all actions:
  - We accomplish this by issuing a temporary coordinate shift: G92 Z0
    (non-persistent; does NOT write EEPROM).
- Also print world coordinates even if GRBL status reports only MPos by maintaining
  a synthetic WCO override derived from MPos at the job start/probe contact.

Notes:
- This does NOT write any GRBL settings (no $10=, etc).
- G92 is temporary (clears on reset, or can be cleared by G92.1).
- Soft-limit clamping remains based on machine coordinates (MPos) and $132.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import serial

# Probe report line example:
#   PRB:123.456,78.900,-12.345:1
PRB_RE = re.compile(r"PRB:([-\d.]+),([-\d.]+),([-\d.]+):([01])")

# Status line examples:
#   <Idle|MPos:0.000,0.000,0.000|FS:0,0>
#   <Idle|WPos:0.000,0.000,0.000|WCO:10.000,20.000,30.000|FS:0,0>
STATUS_MPOS_RE = re.compile(r"(?:^|[|])MPos:([-\d.]+),([-\d.]+),([-\d.]+)(?:[|>]|$)")
STATUS_WPOS_RE = re.compile(r"(?:^|[|])WPos:([-\d.]+),([-\d.]+),([-\d.]+)(?:[|>]|$)")
STATUS_WCO_RE = re.compile(r"(?:^|[|])WCO:([-\d.]+),([-\d.]+),([-\d.]+)(?:[|>]|$)")
STATUS_STATE_RE = re.compile(r"^<([^|>]+)")

# GRBL setting response example:
#   $132=80.000
SETTING_RE = re.compile(r"^\$(\d+)=([-\d.]+)\s*$")

T = TypeVar("T")


@dataclass(frozen=True)
class Result(Generic[T]):
    ok: bool
    value: Optional[T] = None
    error: Optional[str] = None

    @staticmethod
    def success(value: T) -> "Result[T]":
        return Result(ok=True, value=value)

    @staticmethod
    def fail(error: str) -> "Result[T]":
        return Result(ok=False, error=error)


@dataclass(frozen=True)
class ProbeResult:
    prb_x: float
    prb_y: float
    prb_z: float
    success: bool


@dataclass(frozen=True)
class StatusPos:
    mpos: Optional[Tuple[float, float, float]]
    wpos: Optional[Tuple[float, float, float]]
    wco: Optional[Tuple[float, float, float]]


@dataclass(frozen=True)
class SamplePoint:
    ix: int
    iy: int
    x: float
    y: float
    z: float  # Z in "world" coordinates after G92 Z0 (best-effort)


def describe_command(cmd: str) -> List[str]:
    tokens = cmd.strip().split()
    codes: List[str] = []
    for t in tokens:
        if t.startswith(("G", "M", "$")):
            codes.append(t)

    seen = set()
    ordered: List[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    desc = {
        "$H": "($H) GRBL homing cycle.",
        "$X": "($X) GRBL unlock (clears alarm lock).",
        "G90": "(G90) Absolute positioning (active work coordinate system).",
        "G91": "(G91) Relative positioning.",
        "G21": "(G21) Units to millimeters.",
        "G94": "(G94) Feed per minute mode.",
        "G0": "(G0) Rapid move.",
        "G1": "(G1) Linear move at feed rate.",
        "G38.2": "(G38.2) Probe toward target; ALARM if not triggered within distance.",
        "M5": "(M5) Spindle stop.",
        "G92": "(G92) Temporary coordinate offset (non-persistent).",
    }

    out: List[str] = []
    for c in ordered:
        key = c
        if c.startswith("G38.2"):
            key = "G38.2"
        if key in desc:
            out.append(desc[key])
    return out


class Grbl:
    def __init__(self, port: str, baud: int, timeout: float = 2.0) -> None:
        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=timeout,
            write_timeout=timeout,
        )

        # Synthetic WCO override for when status does not include WPos/WCO.
        # Interpreted as: WPos = MPos - _wco_override
        self._wco_override: Optional[Tuple[float, float, float]] = None

    def close(self) -> None:
        try:
            if self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def _readline(self) -> str:
        return self.ser.readline().decode("ascii", errors="replace").strip()

    def wake(self) -> None:
        self.ser.reset_input_buffer()
        self.ser.write(b"\r\n\r\n")
        self.ser.flush()
        time.sleep(0.5)

        t0 = time.time()
        while time.time() - t0 < 0.5:
            line = self._readline()
            if not line:
                break

    def send(self, cmd: str, timeout_s: Optional[float] = None) -> List[str]:
        """
        Send a single line to GRBL and collect all non-ok response lines until 'ok' or an error/alarm.

        Returns: list of intermediate lines (e.g. settings echo, probe report).
        Raises: RuntimeError on 'error:' / 'alarm:', TimeoutError on no completion.
        """
        cmd = cmd.strip()
        if not cmd:
            return []

        overall_timeout = 10.0 if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + overall_timeout

        self.ser.write((cmd + "\n").encode("ascii"))
        self.ser.flush()

        lines: List[str] = []
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out after {overall_timeout:.1f}s waiting for: {cmd!r}"
                )

            line = self._readline()
            if not line:
                continue

            low = line.lower()
            if low == "ok":
                return lines
            if low.startswith("error:") or low.startswith("alarm:"):
                raise RuntimeError(f"{line} (while running {cmd!r})")

            lines.append(line)

    def query_status_line(self, timeout_s: float = 2.0) -> str:
        """
        Send the realtime status query '?' and return the first full status frame '<...>'.
        """
        deadline = time.monotonic() + float(timeout_s)
        self.ser.write(b"?")
        self.ser.flush()

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for status response to '?'")
            line = self._readline()
            if not line:
                continue
            if line.startswith("<") and line.endswith(">"):
                return line

    def get_state(self, timeout_s: float = 2.0) -> str:
        status = self.query_status_line(timeout_s=timeout_s)
        m = STATUS_STATE_RE.search(status)
        if not m:
            raise RuntimeError(f"Could not parse state from status line: {status}")
        return m.group(1)

    def get_status_pos(self, timeout_s: float = 2.0) -> StatusPos:
        status = self.query_status_line(timeout_s=timeout_s)

        mpos: Optional[Tuple[float, float, float]] = None
        wpos: Optional[Tuple[float, float, float]] = None
        wco: Optional[Tuple[float, float, float]] = None

        m = STATUS_MPOS_RE.search(status)
        if m:
            mpos = (float(m.group(1)), float(m.group(2)), float(m.group(3)))

        w = STATUS_WPOS_RE.search(status)
        if w:
            wpos = (float(w.group(1)), float(w.group(2)), float(w.group(3)))

        c = STATUS_WCO_RE.search(status)
        if c:
            wco = (float(c.group(1)), float(c.group(2)), float(c.group(3)))

        return StatusPos(mpos=mpos, wpos=wpos, wco=wco)

    def capture_world_xy_zero_from_current_mpos(
        self, *, status_timeout_s: float = 2.0
    ) -> None:
        """
        Assume the *current physical position* is world X=0, Y=0.
        Store an override so we can display WPos even if status doesn't provide WPos/WCO.
        Z is left unchanged here; we set Z later after the probe contact.
        """
        sp = self.get_status_pos(timeout_s=status_timeout_s)
        if sp.mpos is None:
            raise RuntimeError(
                "Cannot capture world XY=0: status did not include MPos."
            )
        mx, my, _ = sp.mpos

        oz = self._wco_override[2] if self._wco_override is not None else 0.0
        self._wco_override = (mx, my, oz)

    def set_world_z_zero_from_current_mpos(
        self, *, status_timeout_s: float = 2.0
    ) -> None:
        """
        Assume the *current physical position* is world Z=0.
        (We call this at probe contact at world X=0,Y=0.)
        """
        sp = self.get_status_pos(timeout_s=status_timeout_s)
        if sp.mpos is None:
            raise RuntimeError("Cannot set world Z=0: status did not include MPos.")
        mx, my, mz = sp.mpos

        ox = self._wco_override[0] if self._wco_override is not None else mx
        oy = self._wco_override[1] if self._wco_override is not None else my
        self._wco_override = (ox, oy, mz)

    def best_effort_wpos(self, timeout_s: float = 2.0) -> Tuple[float, float, float]:
        """
        Prefer WPos if present; else compute from (MPos+WCO); else compute from (MPos+override).
        """
        sp = self.get_status_pos(timeout_s=timeout_s)
        if sp.wpos is not None:
            return sp.wpos
        if sp.mpos is not None and sp.wco is not None:
            mx, my, mz = sp.mpos
            ox, oy, oz = sp.wco
            return (mx - ox, my - oy, mz - oz)
        if sp.mpos is not None and self._wco_override is not None:
            mx, my, mz = sp.mpos
            ox, oy, oz = self._wco_override
            return (mx - ox, my - oy, mz - oz)
        raise RuntimeError(
            "Status did not contain usable WPos, or (MPos+WCO) to derive it, "
            "and no override has been captured."
        )

    def _format_status_line(self, status: str) -> str:
        m_state = STATUS_STATE_RE.search(status)
        state = m_state.group(1) if m_state else "?"

        w = STATUS_WPOS_RE.search(status)
        if w:
            x, y, z = map(float, w.groups())
            return f"State={state}  WPos X={x:.3f} Y={y:.3f} Z={z:.3f}"

        m = STATUS_MPOS_RE.search(status)
        c = STATUS_WCO_RE.search(status)
        if m and c:
            mx, my, mz = map(float, m.groups())
            ox, oy, oz = map(float, c.groups())
            return (
                f"State={state}  WPos X={mx - ox:.3f} Y={my - oy:.3f} Z={mz - oz:.3f}"
            )

        if m and self._wco_override is not None:
            mx, my, mz = map(float, m.groups())
            ox, oy, oz = self._wco_override
            return (
                f"State={state}  WPos X={mx - ox:.3f} Y={my - oy:.3f} Z={mz - oz:.3f}"
            )

        if m:
            mx, my, mz = map(float, m.groups())
            return f"State={state}  MPos X={mx:.3f} Y={my:.3f} Z={mz:.3f}"

        return f"State={state}"

    def send_and_wait_idle(
        self,
        cmd: str,
        *,
        send_timeout_s: Optional[float] = None,
        idle_timeout_s: float = 30.0,
        poll_s: float = 0.05,
        status_timeout_s: float = 2.0,
    ) -> List[str]:
        lines = self.send(cmd, timeout_s=send_timeout_s)

        deadline = time.monotonic() + float(idle_timeout_s)
        last_print: float = 0.0
        last_line: Optional[str] = None

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for Idle")

            status = self.query_status_line(timeout_s=status_timeout_s)

            m_state = STATUS_STATE_RE.search(status)
            state = m_state.group(1) if m_state else None

            line = self._format_status_line(status)

            now = time.monotonic()
            if line != last_line or (now - last_print) > 0.2:
                print(line)
                last_line = line
                last_print = now

            if state == "Idle":
                return lines

            if state == "Alarm":
                raise RuntimeError("Controller entered ALARM while waiting for Idle")

            time.sleep(poll_s)

    def get_probe_result(self, response_lines: List[str]) -> Optional[ProbeResult]:
        for line in response_lines:
            m = PRB_RE.search(line)
            if m:
                x, y, z, s = m.groups()
                return ProbeResult(float(x), float(y), float(z), s == "1")
        return None

    def read_setting(self, setting_num: int, timeout_s: float = 5.0) -> float:
        """
        Read a single GRBL setting by dumping '$$' and parsing '$<n>=<value>'.
        Compatible with GRBL builds that do not support '$<n>' query.
        """
        lines = self.send("$$", timeout_s=timeout_s)

        for line in lines:
            m = SETTING_RE.match(line.strip())
            if m and int(m.group(1)) == setting_num:
                return float(m.group(2))

        raise RuntimeError(
            f"GRBL did not return ${setting_num} in response to '$$': {lines!r}"
        )

    def run_step(
        self,
        cmd: str,
        context: str,
        timeout_s: Optional[float] = None,
        status_timeout_s: float = 2.0,
        wait_idle: bool = False,
        idle_timeout_s: float = 30.0,
    ) -> Result[List[str]]:
        try:
            # Print current position in "world" if we can (WPos, or derived).
            try:
                wx, wy, wz = self.best_effort_wpos(timeout_s=status_timeout_s)
                print(f"Current WPos: X={wx:.3f} Y={wy:.3f} Z={wz:.3f}")
            except Exception:
                sp = self.get_status_pos(timeout_s=status_timeout_s)
                if sp.mpos is not None:
                    mx, my, mz = sp.mpos
                    print(f"Current MPos: X={mx:.3f} Y={my:.3f} Z={mz:.3f}")

            for d in describe_command(cmd):
                print(d)

            print(f"Executing: {cmd}")

            if wait_idle:
                lines = self.send_and_wait_idle(
                    cmd,
                    send_timeout_s=timeout_s,
                    idle_timeout_s=idle_timeout_s,
                    status_timeout_s=status_timeout_s,
                )
            else:
                lines = self.send(cmd, timeout_s=timeout_s)

            return Result.success(lines)
        except Exception as ex:
            return Result.fail(f"{context}: {ex}")


def build_axis_by_step_distance(
    start: float, length: float, step_distance: float
) -> List[float]:
    if step_distance <= 0:
        raise ValueError("step_distance must be > 0")
    if length < 0:
        raise ValueError("length must be >= 0")

    end = start + length
    if length == 0:
        return [start]

    pts: List[float] = [start]
    i = 1
    while True:
        v = start + i * step_distance
        if v >= end:
            break
        pts.append(v)
        i += 1

    if abs(pts[-1] - end) > 1e-9:
        pts.append(end)
    else:
        pts[-1] = end

    return pts


def clamp_probe_travel_to_soft_limits(
    grbl: Grbl,
    requested_travel_mm: float,
    *,
    margin_mm: float = 0.5,
    status_timeout_s: float = 2.0,
) -> float:
    """
    Clamp a *relative* Z- probe distance (positive mm) so it cannot exceed GRBL soft limits.

    Soft limits are enforced in MACHINE coordinates (MPos).
    Z travel setting is $132 (max Z travel).
    When homed, MPosZ typically 0 at top and min allowed is -$132.
    """
    if requested_travel_mm <= 0:
        raise ValueError("requested_travel_mm must be > 0")

    sp = grbl.get_status_pos(timeout_s=status_timeout_s)
    if sp.mpos is None:
        raise RuntimeError("Status did not include MPos; cannot clamp to soft limits.")
    _, _, mpos_z = sp.mpos

    z_max_travel = grbl.read_setting(132)
    z_min = -z_max_travel

    max_down = mpos_z - z_min
    safe_travel = min(requested_travel_mm, max(0.0, max_down - margin_mm))
    if safe_travel <= 0.0:
        raise RuntimeError(
            f"No safe Z- probing travel available "
            f"(MPosZ={mpos_z:.3f}, z_min={z_min:.3f}, max_down={max_down:.3f})."
        )

    return safe_travel


def probe_height_map(
    grbl: Grbl,
    start_x: float,
    start_y: float,
    width: float,
    height: float,
    step_distance_x: float,
    step_distance_y: float,
    retract_z: float,
    max_probe_travel: float,
    probe_feed: float,
    travel_feed: float,
    settle_s: float,
    unlock: bool,
    home: bool,
) -> Result[List[SamplePoint]]:
    if step_distance_x <= 0 or step_distance_y <= 0:
        return Result.fail("step_distance_x and step_distance_y must be > 0.")
    if width < 0 or height < 0:
        return Result.fail("width and height must be >= 0.")
    if max_probe_travel <= 0:
        return Result.fail("max_probe_travel must be > 0.")
    if probe_feed <= 0 or travel_feed <= 0:
        return Result.fail("probe_feed and travel_feed must be > 0.")

    if unlock:
        r = grbl.run_step("$X", "Unlock ($X)", timeout_s=10.0)
        if not r.ok:
            return Result.fail(r.error or "Unlock failed")

    if home:
        r = grbl.run_step("$H", "Homing ($H)", timeout_s=180.0)
        if not r.ok:
            return Result.fail(r.error or "Homing failed")

    # Modal setup
    for cmd, ctx, tmo in [
        ("G90", "Set absolute mode (G90)", 10.0),
        ("G21", "Set mm units (G21)", 10.0),
        ("G94", "Set feed/min (G94)", 10.0),
        ("M5", "Spindle off (M5)", 10.0),
    ]:
        r = grbl.run_step(cmd, ctx, timeout_s=tmo)
        if not r.ok:
            return Result.fail(r.error or f"{ctx} failed")

    # -------------------------------------------------------------------------
    # Establish "world" frame:
    # 1) Move to world (0,0) using the current coordinate system.
    # 2) Capture XY zero from current MPos so we can DISPLAY world coords.
    # 3) Probe down at (0,0), then:
    #    - set world Z=0 for DISPLAY (override)
    #    - set controller Z=0 for MOTION via G92 Z0 (temporary, non-EEPROM)
    # -------------------------------------------------------------------------

    r = grbl.run_step(
        f"G1 X0 Y0 F{travel_feed:.3f}",
        "Move to world (0, 0) to establish XY origin",
        timeout_s=180.0,
        wait_idle=True,
    )
    if not r.ok:
        return Result.fail(r.error or "Initial XY move to (0,0) failed")

    try:
        grbl.capture_world_xy_zero_from_current_mpos(status_timeout_s=2.0)
        ox, oy, oz = grbl._wco_override or (0.0, 0.0, 0.0)
        print(
            f"World XY origin captured (display override): WCO_override={ox:.3f},{oy:.3f},{oz:.3f}"
        )
    except Exception as ex:
        return Result.fail(f"Failed to capture world XY origin from MPos: {ex}")

    # Pre-probe retract (this Z is interpreted in the current coordinate system)
    r = grbl.run_step(
        f"G0 Z{retract_z:.3f}",
        "Initial retract before Z-zero probe",
        timeout_s=30.0,
        wait_idle=True,
    )
    if not r.ok:
        return Result.fail(r.error or "Initial retract failed")

    if settle_s > 0:
        time.sleep(settle_s)

    print("\n--- Establishing world Z=0 by probing at X=0 Y=0 ---")

    r = grbl.run_step("G91", "Set relative mode (G91) for Z-zero probe", timeout_s=10.0)
    if not r.ok:
        return Result.fail(r.error or "Failed to set G91 for Z-zero probe")

    try:
        safe_travel = clamp_probe_travel_to_soft_limits(
            grbl,
            max_probe_travel,
            margin_mm=0.5,
            status_timeout_s=2.0,
        )
    except Exception as ex:
        return Result.fail(f"Soft-limit clamp failed during Z-zero probe: {ex}")

    probe = grbl.run_step(
        f"G38.2 Z-{safe_travel:.3f} F{probe_feed:.3f}",
        "Probe down to establish Z=0 (G38.2)",
        timeout_s=180.0,
        wait_idle=True,
    )
    if not probe.ok:
        return Result.fail((probe.error or "Z-zero probe failed") + " at X=0 Y=0.")

    prb = grbl.get_probe_result(probe.value or [])
    if prb is None or not prb.success:
        return Result.fail(
            "Z-zero probe did not report success (no PRB:...:1) at X=0 Y=0."
        )

    # Restore absolute mode before applying G92
    r = grbl.run_step(
        "G90", "Restore absolute mode (G90) after Z-zero probe", timeout_s=10.0
    )
    if not r.ok:
        return Result.fail(r.error or "Failed to set G90 after Z-zero probe")

    # Update DISPLAY override: current MPos is world Z=0 at contact
    try:
        grbl.set_world_z_zero_from_current_mpos(status_timeout_s=2.0)
        ox, oy, oz = grbl._wco_override or (0.0, 0.0, 0.0)
        print(
            f"World Z=0 captured (display override): WCO_override={ox:.3f},{oy:.3f},{oz:.3f}"
        )
    except Exception as ex:
        return Result.fail(f"Failed to capture world Z=0 from MPos: {ex}")

    # Make controller MOTION match world Z=0 too (temporary; non-EEPROM).
    # After this, commanding Z values uses your world Z zero at the probe contact point.
    r = grbl.run_step("G92 Z0", "Set temporary world Z0 (G92 Z0)", timeout_s=10.0)
    if not r.ok:
        return Result.fail(r.error or "Failed to set G92 Z0")

    # Now retract in world coordinates (because G92 set Z0)
    r = grbl.run_step(
        f"G0 Z{retract_z:.3f}",
        "Retract after establishing world Z0",
        timeout_s=30.0,
        wait_idle=True,
    )
    if not r.ok:
        return Result.fail(r.error or "Post Z-zero retract failed")

    # -------------------------------------------------------------------------
    # Heightmap scan begins (world coordinates now aligned for motion and display)
    # -------------------------------------------------------------------------

    try:
        xs = build_axis_by_step_distance(start_x, width, step_distance_x)
        ys = build_axis_by_step_distance(start_y, height, step_distance_y)
    except Exception as ex:
        return Result.fail(f"Failed to build probe grid: {ex}")

    print(
        f"Grid: X samples={len(xs)} (step {step_distance_x}mm)  Y samples={len(ys)} (step {step_distance_y}mm)"
    )
    print(
        f"X range: {xs[0]:.3f} .. {xs[-1]:.3f}   Y range: {ys[0]:.3f} .. {ys[-1]:.3f}"
    )

    samples: List[SamplePoint] = []

    # Move to first sample point in world coordinates
    r = grbl.run_step(
        f"G1 X{xs[0]:.3f} Y{ys[0]:.3f} F{travel_feed:.3f}",
        "Move to first XY (world)",
        timeout_s=60.0,
        wait_idle=True,
    )
    if not r.ok:
        return Result.fail(r.error or "Initial move to first XY failed")

    if settle_s > 0:
        time.sleep(settle_s)

    for iy, y in enumerate(ys):
        row = list(enumerate(xs))
        if iy % 2 == 1:
            row = list(reversed(row))

        for ix, x in row:
            print(f"\n--- Sample ix={ix} iy={iy} at X={x:.3f} Y={y:.3f} ---")

            # Retract before XY move (world coordinates)
            r = grbl.run_step(
                f"G0 Z{retract_z:.3f}",
                "Retract to safe Z (world)",
                timeout_s=30.0,
                wait_idle=True,
            )
            if not r.ok:
                return Result.fail(r.error or "Retract failed")

            # XY move (world coordinates)
            r = grbl.run_step(
                f"G1 X{x:.3f} Y{y:.3f} F{travel_feed:.3f}",
                "Move to XY (world)",
                timeout_s=60.0,
                wait_idle=True,
            )
            if not r.ok:
                return Result.fail(r.error or "XY move failed")

            if settle_s > 0:
                time.sleep(settle_s)

            # Probe down relative (clamped to soft limits)
            r = grbl.run_step("G91", "Set relative mode (G91)", timeout_s=10.0)
            if not r.ok:
                return Result.fail(r.error or "Failed to set G91")

            try:
                safe_travel = clamp_probe_travel_to_soft_limits(
                    grbl,
                    max_probe_travel,
                    margin_mm=0.5,
                    status_timeout_s=2.0,
                )
            except Exception as ex:
                return Result.fail(f"Soft-limit clamp failed at ix={ix} iy={iy}: {ex}")

            probe = grbl.run_step(
                f"G38.2 Z-{safe_travel:.3f} F{probe_feed:.3f}",
                "Probe down (G38.2)",
                timeout_s=180.0,
                wait_idle=True,
            )
            if not probe.ok:
                return Result.fail(
                    (probe.error or "Probe failed")
                    + f" at ix={ix} iy={iy} (X={x:.3f}, Y={y:.3f})."
                )

            prb = grbl.get_probe_result(probe.value or [])
            if prb is None or not prb.success:
                return Result.fail(f"Probe did not report success at ix={ix} iy={iy}.")

            # Restore absolute mode
            r = grbl.run_step("G90", "Restore absolute mode (G90)", timeout_s=10.0)
            if not r.ok:
                return Result.fail(r.error or "Failed to set G90")

            # Read current Z in world coordinates (best-effort).
            # Because we used G92 Z0 at the initial contact, Z here should already be in world units.
            try:
                _, _, wz = grbl.best_effort_wpos(timeout_s=2.0)
                measured_z = wz
            except Exception:
                # Fallback: PRB Z may or may not match "world"; keep as a last resort.
                measured_z = prb.prb_z

            # Retract after probe (world)
            r = grbl.run_step(
                f"G0 Z{retract_z:.3f}",
                "Retract after probe (world)",
                timeout_s=30.0,
                wait_idle=True,
            )
            if not r.ok:
                return Result.fail(r.error or "Post-probe retract failed")

            samples.append(SamplePoint(ix=ix, iy=iy, x=x, y=y, z=measured_z))

    return Result.success(samples)


def write_csv(path: str, points: List[SamplePoint]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ix,iy,x_mm,y_mm,z_mm\n")
        for p in points:
            f.write(f"{p.ix},{p.iy},{p.x:.6f},{p.y:.6f},{p.z:.6f}\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GRBL Z height-map probe across a rectangular area (world coordinates via G92 Z0)."
    )
    ap.add_argument(
        "--port", required=True, help="Serial port (e.g. COM3 or /dev/ttyUSB0)"
    )
    ap.add_argument(
        "--baud", type=int, default=115200, help="Baud rate (default: 115200)"
    )

    ap.add_argument(
        "--start-x", type=float, required=True, help="Start X (mm) in world coordinates"
    )
    ap.add_argument(
        "--start-y", type=float, required=True, help="Start Y (mm) in world coordinates"
    )
    ap.add_argument(
        "--width", type=float, required=True, help="Width of area (mm) in +X direction"
    )
    ap.add_argument(
        "--height",
        type=float,
        required=True,
        help="Height of area (mm) in +Y direction",
    )

    ap.add_argument(
        "--step-distance-x",
        type=float,
        required=True,
        help="Step distance along X (mm). End limit (start-x+width) is always included.",
    )
    ap.add_argument(
        "--step-distance-y",
        type=float,
        required=True,
        help="Step distance along Y (mm). End limit (start-y+height) is always included.",
    )

    ap.add_argument(
        "--retract-z",
        type=float,
        required=True,
        help="Safe retract Z (mm) in world coordinates",
    )
    ap.add_argument(
        "--max-probe-travel",
        type=float,
        required=True,
        help="Maximum Z travel (mm) without probe trigger before failing. Clamped to soft limits.",
    )
    ap.add_argument(
        "--probe-feed", type=float, default=60.0, help="Probe feed rate (mm/min)"
    )
    ap.add_argument(
        "--travel-feed", type=float, default=600.0, help="XY travel feed rate (mm/min)"
    )
    ap.add_argument(
        "--settle",
        type=float,
        default=0.1,
        help="Settle time between actions (seconds)",
    )

    ap.add_argument("--unlock", action="store_true", help="Send $X at start")
    ap.add_argument("--home", action="store_true", help="Send $H at start")

    ap.add_argument(
        "--out",
        default="heightmap.csv",
        help="Output CSV path (default: heightmap.csv)",
    )
    ap.add_argument(
        "--out-json", default="", help="Optional output JSON path (default: disabled)"
    )

    args = ap.parse_args()

    grbl: Optional[Grbl] = None
    try:
        grbl = Grbl(args.port, args.baud)
        grbl.wake()

        res = probe_height_map(
            grbl=grbl,
            start_x=args.start_x,
            start_y=args.start_y,
            width=args.width,
            height=args.height,
            step_distance_x=args.step_distance_x,
            step_distance_y=args.step_distance_y,
            retract_z=args.retract_z,
            max_probe_travel=args.max_probe_travel,
            probe_feed=args.probe_feed,
            travel_feed=args.travel_feed,
            settle_s=args.settle,
            unlock=args.unlock,
            home=args.home,
        )
        if not res.ok:
            print(res.error or "Failed", file=sys.stderr)
            return 1

        points = res.value or []
        write_csv(args.out, points)
        print(f"\nWrote CSV: {args.out}")

        steps_x = max((p.ix for p in points), default=-1) + 1
        steps_y = max((p.iy for p in points), default=-1) + 1

        if args.out_json:
            payload = {
                "start_x": args.start_x,
                "start_y": args.start_y,
                "width": args.width,
                "height": args.height,
                "step_distance_x": args.step_distance_x,
                "step_distance_y": args.step_distance_y,
                "steps_x": steps_x,
                "steps_y": steps_y,
                "retract_z": args.retract_z,
                "max_probe_travel": args.max_probe_travel,
                "probe_feed": args.probe_feed,
                "travel_feed": args.travel_feed,
                "points": [p.__dict__ for p in points],
            }
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote JSON: {args.out_json}")

        zs = [p.z for p in points]
        if zs:
            print(
                f"Samples: {len(points)}  Z(min/avg/max): "
                f"{min(zs):.4f} / {sum(zs) / len(zs):.4f} / {max(zs):.4f} (mm)"
            )

        if steps_x and steps_y:
            print(f"Derived grid: steps_x={steps_x} steps_y={steps_y}")

        return 0

    except serial.SerialException as ex:
        print(f"Serial error: {ex}", file=sys.stderr)
        return 1
    except Exception as ex:
        print(f"Unhandled error: {ex}", file=sys.stderr)
        return 1
    finally:
        if grbl is not None:
            try:
                # Best-effort cleanup. If you want to clear the temporary Z offset:
                #   grbl.run_step("G92.1", "Clear G92 offsets (G92.1)", timeout_s=10.0)
                grbl.run_step("$X", "Unlock before final retract ($X)", timeout_s=10.0)

                grbl.run_step(
                    "G53 G0 Z0",
                    "Final retract to machine Z0 (G53)",
                    timeout_s=30.0,
                    wait_idle=True,
                    idle_timeout_s=60.0,
                )

                grbl.run_step("$H", "Return home", timeout_s=180.0)
            except Exception as ex:
                print(
                    f"Warning: failed to retract Z to top before close: {ex}",
                    file=sys.stderr,
                )
            finally:
                grbl.close()


if __name__ == "__main__":
    raise SystemExit(main())
