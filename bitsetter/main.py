#!/usr/bin/env python3
"""
BitSetter-like routine for GRBL (e.g., Shapeoko/Carbide Motion machines)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar, Tuple

import serial

# Matches GRBL probe report lines, for example:
#   PRB:123.456,78.900,-12.345:1
#
# Captures:
#   group(1) -> X position at probe trigger
#   group(2) -> Y position at probe trigger
#   group(3) -> Z position at probe trigger
#   group(4) -> Probe success flag:
#               '1' = probe input triggered
#               '0' = probe failed / did not trigger
#
# Emitted by GRBL after G38.x probing commands.
PRB_RE = re.compile(r"PRB:([-\d.]+),([-\d.]+),([-\d.]+):([01])")


# Matches the machine-position (MPos) field in a GRBL real-time status report, for example:
#   <Idle|MPos:0.000,0.000,0.000|FS:0,0>
#   <Run|MPos:-10.500,200.000,-45.000|WPos:0.000,0.000,0.000>
#
# Captures:
#   group(1) -> Machine X position (mm)
#   group(2) -> Machine Y position (mm)
#   group(3) -> Machine Z position (mm)
#
# The non-capturing groups (?:...) allow matching whether MPos appears:
#   - at the start of the status payload
#   - or between '|' separators
#   - and regardless of what fields follow it
#
# This regex is intentionally tolerant of additional GRBL status fields.
STATUS_MPOS_RE = re.compile(r"(?:^|[|])MPos:([-\d.]+),([-\d.]+),([-\d.]+)(?:[|>]|$)")

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


@dataclass
class ProbeResult:
    x: float
    y: float
    z: float
    success: bool


def describe_command(cmd: str) -> List[str]:
    """
    Returns human-friendly descriptions for the tokens present in the command line.
    """
    # Tokenize roughly by spaces; keep things like "G38.2" intact.
    tokens = cmd.strip().split()

    # Also detect codes embedded with parameters (e.g. "G0", "G53", "G38.2", "M5")
    # Some lines are like: "G53 G0 Z-45.000"
    codes = []
    for t in tokens:
        if t.startswith(("G", "M", "$")):
            # Strip any trailing punctuation (unlikely) but keep dot variants
            codes.append(t)

    # De-dupe while preserving order
    seen = set()
    ordered = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    desc = {
        "$H": "($H) GRBL homing cycle: establishes machine zero using limit switches.",
        "$X": "($X) GRBL unlock: clears an alarm lock so motion/G-code can run (does not home).",
        "G90": "(G90) Absolute positioning: coordinates are interpreted as absolute in the active coordinate system.",
        "G91": "(G91) Relative positioning: coordinates are interpreted as incremental moves from the current position.",
        "G21": "(G21) Units to millimeters.",
        "G94": "(G94) Feed rate mode: feed per minute.",
        "G49": "(G49) Cancel tool length offset (TLO).",
        "M5": "(M5) Spindle stop.",
        "G0": "(G0) Rapid move: fastest non-cutting move (no feed rate).",
        "G53": (
            "(G53) Use machine coordinates for this line only (one-shot), "
            "ignoring work offsets (G54–G59). Useful for fixed locations like tool change / BitSetter."
        ),
        "G38.2": (
            "(G38.2) Probe toward the target: moves until the probe input triggers. "
            "If it does not trigger within the commanded distance, GRBL raises an alarm."
        ),
    }

    out: List[str] = []
    for c in ordered:
        # Normalize common “G38.2” token
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
            timeout=timeout,  # per-read timeout
            write_timeout=timeout,
        )

    def close(self) -> None:
        try:
            if self.ser.is_open:
                self.ser.close()
        except Exception:
            # Closing should never crash the process.
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
        Send a command and read until 'ok' or 'error:'/'alarm:'.

        timeout_s is an overall deadline for the command (not the serial read timeout).
        If timeout_s is None, a default of 5 seconds is used.
        """
        cmd = cmd.strip()
        if not cmd:
            return []

        overall_timeout = 5.0 if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + overall_timeout

        self.ser.write((cmd + "\n").encode("ascii"))
        self.ser.flush()

        lines: List[str] = []
        while True:
            # If we hit the overall deadline, fail.
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out after {overall_timeout:.1f}s waiting for response to: {cmd!r}"
                )

            line = self._readline()

            # IMPORTANT: empty line just means "no data yet" due to serial timeout.
            # Keep waiting until deadline.
            if not line:
                continue

            low = line.lower()
            if low == "ok":
                return lines
            if low.startswith("error:") or low.startswith("alarm:"):
                raise RuntimeError(f"{line} (while running {cmd!r})")

            lines.append(line)

    def query_status(self, timeout_s: float = 2.0) -> str:
        """
        Sends '?' and returns the first status line like:
        <Idle|MPos:0.000,0.000,0.000|FS:0,0|...>
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

    def get_mpos(self, timeout_s: float = 2.0) -> Tuple[float, float, float]:
        status = self.query_status(timeout_s=timeout_s)
        m = STATUS_MPOS_RE.search(status)
        if not m:
            raise RuntimeError(f"Status line did not contain MPos: {status}")
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    def get_probe_result(self, response_lines: List[str]) -> Optional[ProbeResult]:
        for line in response_lines:
            m = PRB_RE.search(line)
            if m:
                x, y, z, s = m.groups()
                return ProbeResult(float(x), float(y), float(z), s == "1")
        return None

    def run_step(
        self,
        cmd: str,
        context: str,
        timeout_s: Optional[float] = None,
        status_timeout_s: float = 2.0,
    ) -> Result[List[str]]:
        """
        Callable method:
          - prints current MPos (best effort, but treated as required; fails if status can't be read)
          - prints descriptions of any G/M/$ codes on this line
          - executes the command
        """
        try:
            x, y, z = self.get_mpos(timeout_s=status_timeout_s)
            print(f"Current MPos: X={x:.3f} Y={y:.3f} Z={z:.3f}")

            for d in describe_command(cmd):
                print(d)

            print(f"Executing: {cmd}")
            lines = self.send(cmd, timeout_s=timeout_s)
            return Result.success(lines)
        except Exception as ex:
            return Result.fail(f"{context}: {ex}")


def run_bitsetter_like(
    grbl: Grbl,
    bitsetter_x: float,
    bitsetter_y: float,
    safe_z: float,
    probe_distance: float,
    probe_fast_feed: float,
    probe_slow_feed: float,
    retract: float,
    settle_s: float,
) -> Result[ProbeResult]:
    print("Unlocking ($X)...")
    r = grbl.run_step("$X", "Unlock ($X)", timeout_s=10.0)
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    for cmd, ctx, tmo in [
        ("G90", "Set absolute mode (G90)", 10.0),
        ("G21", "Set mm units (G21)", 10.0),
        ("G94", "Set feed/min (G94)", 10.0),
        ("G49", "Cancel TLO (G49)", 10.0),
        ("M5", "Spindle off (M5)", 10.0),
    ]:
        r = grbl.run_step(cmd, ctx, timeout_s=tmo)
        if not r.ok:
            print(r.error, file=sys.stderr)
            return Result.fail(r.error)

    print("Homing ($H)...")
    r = grbl.run_step("$H", "Homing ($H)", timeout_s=180.0)
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    print(f"Moving to safe Z (G53 Z{safe_z})...")
    r = grbl.run_step(f"G53 G0 Z{safe_z:.3f}", "Move to safe Z (G53)", timeout_s=30.0)
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    print(f"Moving to BitSetter XY (G53 X{bitsetter_x}, Y{bitsetter_y})...")
    r = grbl.run_step(
        f"G53 G0 X{bitsetter_x:.3f} Y{bitsetter_y:.3f}",
        "Move to BitSetter XY (G53)",
        timeout_s=60.0,
    )
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    if settle_s > 0:
        time.sleep(settle_s)

    print("Pre-positioning Z to -45.0 mm (G53)...")
    r = grbl.run_step(
        "G53 G0 Z-45.000", "Pre-position Z to -45 mm (G53)", timeout_s=30.0
    )
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    if settle_s > 0:
        time.sleep(settle_s)

    r = grbl.run_step("G91", "Set relative mode (G91)", timeout_s=10.0)
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    print(f"Probing fast: G38.2 Z-{probe_distance} F{probe_fast_feed} ...")
    fast = grbl.run_step(
        f"G38.2 Z-{probe_distance:.3f} F{probe_fast_feed:.3f}",
        "Fast probe (G38.2)",
        timeout_s=180.0,
    )
    if not fast.ok:
        print(fast.error, file=sys.stderr)
        return Result.fail(fast.error)

    fast_prb = grbl.get_probe_result(fast.value or [])
    if not fast_prb or not fast_prb.success:
        msg = "Fast probe did not report success (no PRB:...:1 found)."
        print(msg, file=sys.stderr)
        return Result.fail(msg)

    r = grbl.run_step(
        f"G0 Z{retract:.3f}", "Retract after fast probe (G0)", timeout_s=30.0
    )
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    if settle_s > 0:
        time.sleep(settle_s)

    slow_dist = retract + 5.0
    print(f"Probing slow: G38.2 Z-{slow_dist} F{probe_slow_feed} ...")
    slow = grbl.run_step(
        f"G38.2 Z-{slow_dist:.3f} F{probe_slow_feed:.3f}",
        "Slow probe (G38.2)",
        timeout_s=180.0,
    )
    if not slow.ok:
        print(slow.error, file=sys.stderr)
        return Result.fail(slow.error)

    slow_prb = grbl.get_probe_result(slow.value or [])
    if not slow_prb or not slow_prb.success:
        msg = "Slow probe did not report success (no PRB:...:1 found)."
        print(msg, file=sys.stderr)
        return Result.fail(msg)

    r = grbl.run_step("G90", "Restore absolute mode (G90)", timeout_s=10.0)
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    print(f"Returning to safe Z (G53 Z{safe_z})...")
    r = grbl.run_step(f"G53 G0 Z{safe_z:.3f}", "Return to safe Z (G53)", timeout_s=30.0)
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    return Result.success(slow_prb)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="BitSetter-like homing + move + probe routine (GRBL)."
    )
    ap.add_argument(
        "--port", required=True, help="Serial port (e.g. COM3 or /dev/ttyUSB0)"
    )
    ap.add_argument(
        "--baud", type=int, default=115200, help="Baud rate (default: 115200)"
    )
    ap.add_argument(
        "--bitsetter-x",
        type=float,
        default=-4.0,
        help="BitSetter X in machine coords (mm)",
    )
    ap.add_argument(
        "--bitsetter-y",
        type=float,
        default=-858.5,
        help="BitSetter Y in machine coords (mm)",
    )
    ap.add_argument(
        "--safe-z", type=float, default=0.0, help="Machine-coordinate Z clearance (mm)."
    )
    ap.add_argument(
        "--probe-distance", type=float, default=30.0, help="Probe travel distance (mm)."
    )
    ap.add_argument(
        "--probe-fast-feed", type=float, default=150.0, help="Fast probe feed (mm/min)"
    )
    ap.add_argument(
        "--probe-slow-feed", type=float, default=25.0, help="Slow probe feed (mm/min)"
    )
    ap.add_argument(
        "--retract",
        type=float,
        default=2.0,
        help="Retract between fast/slow probe (mm)",
    )
    ap.add_argument(
        "--settle",
        type=float,
        default=0.25,
        help="Settle time between actions (seconds)",
    )
    args = ap.parse_args()

    grbl: Optional[Grbl] = None
    try:
        grbl = Grbl(args.port, args.baud)
        grbl.wake()

        res = run_bitsetter_like(
            grbl=grbl,
            bitsetter_x=args.bitsetter_x,
            bitsetter_y=args.bitsetter_y,
            safe_z=args.safe_z,
            probe_distance=args.probe_distance,
            probe_fast_feed=args.probe_fast_feed,
            probe_slow_feed=args.probe_slow_feed,
            retract=args.retract,
            settle_s=args.settle,
        )

        if not res.ok:
            return 1

        result = res.value
        if result is None:
            print("Unexpected: missing probe result.", file=sys.stderr)
            return 1

        print("\nProbe result (from PRB report):")
        print(f"  success: {result.success}")
        print(f"  X: {result.x:.3f} mm")
        print(f"  Y: {result.y:.3f} mm")
        print(f"  Z: {result.z:.3f} mm")

        # "Tool length" (useful value) = Z machine position when the probe triggered on the final (slow) probe.
        # This is the BitSetter probe trigger reading for this tool in machine coordinates.
        print("\nTool length reading (probe trigger):")
        print(f"  Trigger MPos Z: {result.z:.3f} mm")

        return 0

    except serial.SerialException as ex:
        print(f"Serial error: {ex}", file=sys.stderr)
        return 1
    except Exception as ex:
        print(f"Unhandled error: {ex}", file=sys.stderr)
        return 1
    finally:
        if grbl is not None:
            grbl.close()


if __name__ == "__main__":
    raise SystemExit(main())
