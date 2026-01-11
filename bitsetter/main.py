#!/usr/bin/env python3
"""
BitSetter-like routine for GRBL (e.g., Shapeoko/Carbide Motion machines)

Changes vs your version:
- Grbl.send() now returns List[str] (never Optional) and does NOT swallow exceptions.
- Added a small Result[T] type. Callers check ok/err and bail early.
- All failures are printed to console (stderr) at the call site (and in main for unexpected exceptions).
- Serial port is always closed before exiting (finally).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

import serial


PRB_RE = re.compile(r"PRB:([-\d.]+),([-\d.]+),([-\d.]+):([01])")

T = TypeVar("T")


@dataclass(frozen=True)
class Result(Generic[T]):
    ok: bool
    value: Optional[T] = None
    error: Optional[str] = None

    @staticmethod
    def success(value: T) -> "Result[T]":
        return Result(ok=True, value=value, error=None)

    @staticmethod
    def fail(error: str) -> "Result[T]":
        return Result(ok=False, value=None, error=error)


@dataclass
class ProbeResult:
    x: float
    y: float
    z: float
    success: bool


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

    def soft_reset(self) -> None:
        self.ser.write(b"\x18")  # Ctrl-X
        self.ser.flush()
        time.sleep(0.5)
        self.ser.reset_input_buffer()

    def get_probe_result(self, response_lines: List[str]) -> Optional[ProbeResult]:
        for line in response_lines:
            m = PRB_RE.search(line)
            if m:
                x, y, z, s = m.groups()
                return ProbeResult(float(x), float(y), float(z), s == "1")
        return None


def _try_send(
    grbl: Grbl, cmd: str, context: str, timeout_s: Optional[float] = None
) -> Result[List[str]]:
    try:
        return Result.success(grbl.send(cmd, timeout_s=timeout_s))
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
    # 1) Home
    print("Homing ($X)...")
    r = _try_send(grbl, "$X", "Homing ($X)")
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    # Ensure known state.
    for cmd, ctx in [
        ("G90", "Set absolute mode (G90)"),
        ("G21", "Set mm units (G21)"),
        ("G94", "Set feed per minute (G94)"),
        ("G49", "Cancel tool length offsets (G49)"),
        ("M5", "Spindle off (M5)"),
    ]:
        r = _try_send(grbl, cmd, ctx)
        if not r.ok:
            print(r.error, file=sys.stderr)
            return Result.fail(r.error)

    # 1) Home
    print("Homing ($H)...")
    r = _try_send(grbl, "$H", "Homing ($H)", timeout_s=180.0)
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    # 2) Move to safe Z then BitSetter XY (machine coords via G53)
    print(f"Moving to safe Z (G53 Z{safe_z})...")
    r = _try_send(grbl, f"G53 G0 Z{safe_z:.3f}", "Move to safe Z (G53)")
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    print(f"Moving to BitSetter XY (G53 X{bitsetter_x}, Y{bitsetter_y})...")
    r = _try_send(
        grbl,
        f"G53 G0 X{bitsetter_x:.3f} Y{bitsetter_y:.3f}",
        "Move to BitSetter XY (G53)",
    )
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    if settle_s > 0:
        time.sleep(settle_s)

    print("Pre-positioning Z to -45.0 mm (G53)...")
    r = _try_send(grbl, "G53 G0 Z-45.000", "Pre-position Z to -45 mm (G53)")
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    if settle_s > 0:
        time.sleep(settle_s)

    # 3) Two-stage probe
    print(f"Probing fast: G91 G38.2 Z-{probe_distance} F{probe_fast_feed} ...")
    r = _try_send(grbl, "G91", "Set relative mode (G91)")
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    fast = _try_send(
        grbl,
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

    print(f"Retracting: G0 Z{retract} (relative)...")
    r = _try_send(grbl, f"G0 Z{retract:.3f}", "Retract after fast probe (G0)")
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    if settle_s > 0:
        time.sleep(settle_s)

    slow_dist = retract + 5.0
    print(f"Probing slow: G38.2 Z-{slow_dist} F{probe_slow_feed} ...")
    slow = _try_send(
        grbl,
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

    r = _try_send(grbl, "G90", "Restore absolute mode (G90)")
    if not r.ok:
        print(r.error, file=sys.stderr)
        return Result.fail(r.error)

    print(f"Returning to safe Z (G53 Z{safe_z})...")
    r = _try_send(grbl, f"G53 G0 Z{safe_z:.3f}", "Return to safe Z (G53)")
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
        default=-1.0,
        help="BitSetter X in machine coords (mm)",
    )
    ap.add_argument(
        "--bitsetter-y",
        type=float,
        default=-853.5,
        help="BitSetter Y in machine coords (mm)",
    )
    ap.add_argument(
        "--safe-z",
        type=float,
        default=0.0,
        help="Machine-coordinate Z clearance to move at (mm). Set this to a known-safe machine Z.",
    )
    ap.add_argument(
        "--probe-distance",
        type=float,
        default=30.0,
        help="Max Z distance to probe downward (mm) (relative). Must be safe for your setup.",
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

    grbl = None
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
            # run_bitsetter_like already printed a specific error, but main returns failure.
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
