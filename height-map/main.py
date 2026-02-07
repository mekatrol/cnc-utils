#!/usr/bin/env python3
"""
GRBL Z height-map probe for a rectangular area (e.g., Vevor 3018 with probe input)

- Probes a grid across a specified rectangle in the active work coordinate system (e.g., G54).
- Records Z at each (X,Y) where the probe triggers.
- Outputs CSV (and optional JSON) with the sampled points.

Safety/behavior:
- Moves to a retract height before any XY move (optional; enabled by default here).
- Uses G38.2 with a caller-specified max travel, but clamps the travel so it cannot exceed GRBL soft limits.
- Uses a serpentine scan pattern to reduce rapids.

Soft-limit clamping notes (important):
- GRBL soft limits are enforced in MACHINE coordinates (MPos).
- The Z soft-limit minimum is typically at MPosZ == -$132 when homed at MPosZ == 0.
- Before each probe, this script reads:
  - Current MPosZ via "?" status
  - $132 (max Z travel) via "$132"
- It then computes the maximum safe downward travel and clamps G38.2 distance accordingly.

Typical usage:
  ./z_height_map.py --port /dev/ttyUSB0 --start-x 0 --start-y 0 --width 100 --height 80 --steps-x 11 --steps-y 9 \
    --retract-z 5 --max-probe-travel 15 --probe-feed 60 --travel-feed 600 --out heightmap.csv
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
    z: float  # Z in work coordinates (best-effort; see notes below)


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
        """
        Returns the GRBL state from the status line, e.g. 'Idle', 'Run', 'Hold', 'Alarm'.
        """
        status = self.query_status_line(timeout_s=timeout_s)
        m = STATUS_STATE_RE.search(status)
        if not m:
            raise RuntimeError(f"Could not parse state from status line: {status}")
        return m.group(1)

    def _format_status_line(self, status: str) -> str:
        """
        Extracts state and best-effort position from a raw GRBL status line
        and formats it for logging.
        """
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
        """
        Sends a command (waits for 'ok'), then polls status until the controller
        reports 'Idle', printing live position updates while waiting.

        Note: When the sender is streaming commands quickly, '?' responses can interleave.
        This implementation expects the port is used by this process alone.
        """
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

    def best_effort_wpos(self, timeout_s: float = 2.0) -> Tuple[float, float, float]:
        """
        Prefer WPos if present; else compute WPos from MPos and WCO if both present.
        If neither is possible, raise.
        """
        sp = self.get_status_pos(timeout_s=timeout_s)
        if sp.wpos is not None:
            return sp.wpos
        if sp.mpos is not None and sp.wco is not None:
            mx, my, mz = sp.mpos
            ox, oy, oz = sp.wco
            return (mx - ox, my - oy, mz - oz)
        raise RuntimeError(
            "Status did not contain usable WPos, or (MPos+WCO) to derive it."
        )

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


def linspace(start: float, end: float, steps: int) -> List[float]:
    if steps < 2:
        return [start]
    span = end - start
    return [start + span * (i / (steps - 1)) for i in range(steps)]


def clamp_probe_travel_to_soft_limits(
    grbl: Grbl,
    requested_travel_mm: float,
    *,
    margin_mm: float = 0.5,
    status_timeout_s: float = 2.0,
) -> float:
    """
    Clamp a *relative* Z- probe move distance (positive mm) so it cannot exceed GRBL soft limits.

    Assumptions/behavior:
    - Soft limits are enforced in machine coordinates (MPos).
    - Z travel setting is $132 (max travel in mm).
    - When homed, MPosZ is typically 0 at the top and the lowest allowed is -$132.

    Returns:
      safe_travel_mm (positive). May be less than requested_travel_mm.

    Raises:
      RuntimeError if no safe downward motion is available (already at/below limit).
    """
    if requested_travel_mm <= 0:
        raise ValueError("requested_travel_mm must be > 0")

    # Read current MPosZ
    sp = grbl.get_status_pos(timeout_s=status_timeout_s)
    if sp.mpos is None:
        raise RuntimeError("Status did not include MPos; cannot clamp to soft limits.")
    _, _, mpos_z = sp.mpos

    # Read $132 (Z max travel)
    z_max_travel = grbl.read_setting(132)

    # Compute machine Z minimum allowed by soft limits.
    z_min = -z_max_travel

    # How far can we move downward before crossing the minimum?
    max_down = mpos_z - z_min  # (e.g. if mpos_z=-25 and z_min=-80 => 55mm available)

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
    steps_x: int,
    steps_y: int,
    retract_z: float,
    max_probe_travel: float,
    probe_feed: float,
    travel_feed: float,
    settle_s: float,
    unlock: bool,
    home: bool,
) -> Result[List[SamplePoint]]:
    if steps_x <= 0 or steps_y <= 0:
        return Result.fail("steps_x and steps_y must be positive.")
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

    # Basic modal setup (work coordinates)
    for cmd, ctx, tmo in [
        ("G90", "Set absolute mode (G90)", 10.0),
        ("G21", "Set mm units (G21)", 10.0),
        ("G94", "Set feed/min (G94)", 10.0),
        ("M5", "Spindle off (M5)", 10.0),
    ]:
        r = grbl.run_step(cmd, ctx, timeout_s=tmo)
        if not r.ok:
            return Result.fail(r.error or f"{ctx} failed")

    # Precompute grid (work coordinates)
    xs = linspace(start_x, start_x + width, steps_x)
    ys = linspace(start_y, start_y + height, steps_y)

    samples: List[SamplePoint] = []

    # Move to (0, 0)
    r = grbl.run_step(
        f"G1 X0 Y0 F{travel_feed:.3f}",
        "Move to (0, 0)",
        timeout_s=180.0,
        wait_idle=True,
    )
    if not r.ok:
        return Result.fail(r.error or "Initial XY move failed")

    # Start by retracting to safe Z in work coordinates.
    # This reduces the chance of dragging the probe during the initial move.
    r = grbl.run_step(
        f"G0 Z{retract_z:.3f}",
        "Initial retract to safe Z",
        timeout_s=30.0,
        wait_idle=True,
    )
    if not r.ok:
        return Result.fail(r.error or "Initial retract failed")

    # Move to first sample point (start_x/start_y) rather than hardcoding X0 Y0.
    r = grbl.run_step(
        f"G1 X{xs[0]:.3f} Y{ys[0]:.3f} F{travel_feed:.3f}",
        "Move to first XY",
        timeout_s=60.0,
        wait_idle=True,
    )
    if not r.ok:
        return Result.fail(r.error or "Initial XY move failed")

    if settle_s > 0:
        time.sleep(settle_s)

    for iy, y in enumerate(ys):
        # Serpentine: alternate direction each row
        row = list(enumerate(xs))
        if iy % 2 == 1:
            row = list(reversed(row))

        for ix, x in row:
            print(f"\n--- Sample ix={ix} iy={iy} at X={x:.3f} Y={y:.3f} ---")

            # Retract before any XY move (in work coordinates).
            r = grbl.run_step(
                f"G0 Z{retract_z:.3f}",
                "Retract to safe Z",
                timeout_s=30.0,
                wait_idle=True,
            )
            if not r.ok:
                return Result.fail(r.error or "Retract failed")

            # XY move at travel feed (use G1 so F applies consistently)
            r = grbl.run_step(
                f"G1 X{x:.3f} Y{y:.3f} F{travel_feed:.3f}",
                "Move to XY",
                timeout_s=60.0,
                wait_idle=True,
            )
            if not r.ok:
                return Result.fail(r.error or "XY move failed")

            if settle_s > 0:
                time.sleep(settle_s)

            # Probe down relative: switch to G91, then clamp the requested travel to soft limits.
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
                # If probe fails, GRBL likely ALARMs; caller wanted "fail after max travel"
                return Result.fail(
                    (probe.error or "Probe failed")
                    + f" at ix={ix} iy={iy} (X={x:.3f}, Y={y:.3f})."
                )

            prb = grbl.get_probe_result(probe.value or [])
            if prb is None or not prb.success:
                return Result.fail(
                    f"Probe did not report success (no PRB:...:1) at ix={ix} iy={iy}."
                )

            # Immediately read current Z in work coordinates after probe.
            # This is usually the most useful Z height for a heightmap in the active WCS (e.g. G54).
            try:
                _, _, wz = grbl.best_effort_wpos(timeout_s=2.0)
                measured_z = wz
            except Exception:
                # Fallback: use PRB Z (coordinate basis can differ by sender/config).
                measured_z = prb.prb_z

            # Restore absolute mode and retract to safe height.
            r = grbl.run_step("G90", "Restore absolute mode (G90)", timeout_s=10.0)
            if not r.ok:
                return Result.fail(r.error or "Failed to set G90")

            r = grbl.run_step(
                f"G0 Z{retract_z:.3f}",
                "Retract after probe",
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
        description="GRBL Z height-map probe across a rectangular area (work coordinates)."
    )
    ap.add_argument(
        "--port", required=True, help="Serial port (e.g. COM3 or /dev/ttyUSB0)"
    )
    ap.add_argument(
        "--baud", type=int, default=115200, help="Baud rate (default: 115200)"
    )

    ap.add_argument(
        "--start-x", type=float, required=True, help="Start X (mm) in work coordinates"
    )
    ap.add_argument(
        "--start-y", type=float, required=True, help="Start Y (mm) in work coordinates"
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
        "--steps-x", type=int, required=True, help="Number of samples along X (>=1)"
    )
    ap.add_argument(
        "--steps-y", type=int, required=True, help="Number of samples along Y (>=1)"
    )

    ap.add_argument(
        "--retract-z",
        type=float,
        required=True,
        help="Safe retract Z (mm) in work coordinates",
    )
    ap.add_argument(
        "--max-probe-travel",
        type=float,
        required=True,
        help="Maximum Z travel (mm) without probe trigger before failing (G38.2 distance). "
        "This will be clamped to soft limits automatically.",
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

    ap.add_argument(
        "--unlock",
        action="store_true",
        help="Send $X at start (recommended if you might be alarm-locked)",
    )
    ap.add_argument(
        "--home",
        action="store_true",
        help="Send $H at start (only if your machine is set up for homing)",
    )

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
            steps_x=args.steps_x,
            steps_y=args.steps_y,
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

        if args.out_json:
            payload = {
                "start_x": args.start_x,
                "start_y": args.start_y,
                "width": args.width,
                "height": args.height,
                "steps_x": args.steps_x,
                "steps_y": args.steps_y,
                "retract_z": args.retract_z,
                "max_probe_travel": args.max_probe_travel,
                "probe_feed": args.probe_feed,
                "travel_feed": args.travel_feed,
                "points": [p.__dict__ for p in points],
            }
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote JSON: {args.out_json}")

        # Minimal summary
        zs = [p.z for p in points]
        if zs:
            print(
                f"Samples: {len(points)}  Z(min/avg/max): "
                f"{min(zs):.4f} / {sum(zs) / len(zs):.4f} / {max(zs):.4f} (mm)"
            )

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
                # Best-effort: raise Z to the top soft-limit boundary before closing.
                # - Uses machine coordinates (G53) so it is independent of work offsets.
                # - Assumes homed Z=0 at the top (typical GRBL convention).
                # - If the controller is in ALARM (or not homed), GRBL may reject motion.
                grbl.run_step("$X", "Unlock before final retract ($X)", timeout_s=10.0)

                # Absolute machine move to Z0 (top). Use G0 for rapid.
                grbl.run_step(
                    "G53 G0 Z0",
                    "Final retract to machine Z0 (G53)",
                    timeout_s=30.0,
                    wait_idle=True,
                    idle_timeout_s=60.0,
                )
            except Exception as ex:
                # Do not mask the original exception; just report and proceed to close.
                print(
                    f"Warning: failed to retract Z to top before close: {ex}",
                    file=sys.stderr,
                )
            finally:
                grbl.close()


if __name__ == "__main__":
    raise SystemExit(main())
