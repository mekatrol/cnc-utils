#!/usr/bin/env python3
# Qt6 GUI version (PySide6) of the GRBL height-map probe tool.
# - No console output: all output goes to a QTextEdit log widget.
# - Displays MPos and WCO in dedicated widgets (and WPos derived best-effort).
#
# Dependencies:
#   pip install PySide6 pyserial
#
# Windows "no console window" options:
#   - Run with pythonw.exe:  pythonw grbl_probe_gui.py
#   - Or PyInstaller:  pyinstaller --noconsole --onefile grbl_probe_gui.py

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import serial
from serial.tools import list_ports

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, QSettings
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

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
        key = "G38.2" if c.startswith("G38.2") else c
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
        sp = self.get_status_pos(timeout_s=status_timeout_s)
        if sp.mpos is None:
            raise RuntimeError("Cannot set world Z=0: status did not include MPos.")
        mx, my, mz = sp.mpos

        ox = self._wco_override[0] if self._wco_override is not None else mx
        oy = self._wco_override[1] if self._wco_override is not None else my
        self._wco_override = (ox, oy, mz)

    def best_effort_wpos(self, timeout_s: float = 2.0) -> Tuple[float, float, float]:
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
        raise RuntimeError("No usable WPos source (missing WPos, and no WCO/override).")

    def best_effort_wco(
        self, timeout_s: float = 2.0
    ) -> Optional[Tuple[float, float, float]]:
        sp = self.get_status_pos(timeout_s=timeout_s)
        if sp.wco is not None:
            return sp.wco
        if self._wco_override is not None:
            return self._wco_override
        return None

    def read_setting(self, setting_num: int, timeout_s: float = 5.0) -> float:
        lines = self.send("$$", timeout_s=timeout_s)
        for line in lines:
            m = SETTING_RE.match(line.strip())
            if m and int(m.group(1)) == setting_num:
                return float(m.group(2))
        raise RuntimeError(
            f"GRBL did not return ${setting_num} in response to '$$': {lines!r}"
        )

    def get_probe_result(self, response_lines: List[str]) -> Optional[ProbeResult]:
        for line in response_lines:
            m = PRB_RE.search(line)
            if m:
                x, y, z, s = m.groups()
                return ProbeResult(float(x), float(y), float(z), s == "1")
        return None


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
            f"No safe Z- probing travel available (MPosZ={mpos_z:.3f}, z_min={z_min:.3f})."
        )
    return safe_travel


@dataclass(frozen=True)
class ProbeParams:
    start_x: float
    start_y: float
    width: float
    height: float
    step_distance_x: float
    step_distance_y: float
    retract_z: float
    max_probe_travel: float
    probe_feed: float
    travel_feed: float
    settle_s: float
    unlock: bool
    home: bool
    out_csv: str
    out_json: str


class ProbeWorker(QObject):
    log = Signal(str)
    finished_ok = Signal(str)  # message
    finished_error = Signal(str)  # error text

    # live telemetry snapshots (optional)
    telemetry = Signal(object)  # dict with mpos/wco/wpos/state if available

    def __init__(self, port: str, baud: int, params: ProbeParams) -> None:
        super().__init__()
        self._port = port
        self._baud = baud
        self._params = params
        self._stop_requested = False

    @Slot()
    def request_stop(self) -> None:
        self._stop_requested = True

    def _log(self, s: str) -> None:
        self.log.emit(s)

    def _maybe_stop(self) -> None:
        if self._stop_requested:
            raise RuntimeError("Stop requested by user.")

    def _run_step(
        self,
        grbl: Grbl,
        cmd: str,
        context: str,
        *,
        timeout_s: Optional[float] = None,
        wait_idle: bool = False,
        idle_timeout_s: float = 30.0,
        status_timeout_s: float = 2.0,
    ) -> List[str]:
        self._maybe_stop()

        try:
            # Log current WPos if available; else MPos.
            try:
                wx, wy, wz = grbl.best_effort_wpos(timeout_s=status_timeout_s)
                self._log(f"Current WPos: X={wx:.3f} Y={wy:.3f} Z={wz:.3f}")
            except Exception:
                sp = grbl.get_status_pos(timeout_s=status_timeout_s)
                if sp.mpos is not None:
                    mx, my, mz = sp.mpos
                    self._log(f"Current MPos: X={mx:.3f} Y={my:.3f} Z={mz:.3f}")

            for d in describe_command(cmd):
                self._log(d)

            self._log(f"Executing: {cmd}")

            lines = grbl.send(cmd, timeout_s=timeout_s)
            if wait_idle:
                deadline = time.monotonic() + float(idle_timeout_s)
                while True:
                    self._maybe_stop()
                    if time.monotonic() > deadline:
                        raise TimeoutError("Timed out waiting for Idle")
                    status = grbl.query_status_line(timeout_s=status_timeout_s)
                    m_state = STATUS_STATE_RE.search(status)
                    state = m_state.group(1) if m_state else "?"
                    # emit telemetry snapshot for UI
                    self._emit_telemetry(grbl, status)
                    if state == "Idle":
                        break
                    if state == "Alarm":
                        raise RuntimeError(
                            "Controller entered ALARM while waiting for Idle"
                        )
                    time.sleep(0.05)

            return lines
        except Exception as ex:
            raise RuntimeError(f"{context}: {ex}")

    def _emit_telemetry(self, grbl: Grbl, status_line: Optional[str] = None) -> None:
        try:
            sp = (
                grbl.get_status_pos(timeout_s=1.0)
                if status_line is None
                else grbl.get_status_pos(timeout_s=1.0)
            )
            # State from provided status_line if any; else query.
            st = "?"
            if status_line:
                m = STATUS_STATE_RE.search(status_line)
                st = m.group(1) if m else "?"
            else:
                try:
                    sl = grbl.query_status_line(timeout_s=1.0)
                    m = STATUS_STATE_RE.search(sl)
                    st = m.group(1) if m else "?"
                except Exception:
                    st = "?"
            wco_eff = grbl.best_effort_wco(timeout_s=1.0)
            wpos_eff = None
            try:
                wpos_eff = grbl.best_effort_wpos(timeout_s=1.0)
            except Exception:
                wpos_eff = None

            self.telemetry.emit(
                {
                    "state": st,
                    "mpos": sp.mpos,
                    "wco": wco_eff,
                    "wpos": wpos_eff,
                }
            )
        except Exception:
            # Keep UI resilient.
            pass

    @Slot()
    def run(self) -> None:
        grbl: Optional[Grbl] = None
        try:
            grbl = Grbl(self._port, self._baud)
            grbl.wake()
            self._log(f"Connected: {self._port} @ {self._baud}")

            p = self._params

            # Optional unlock/home
            if p.unlock:
                self._run_step(grbl, "$X", "Unlock ($X)", timeout_s=10.0)
            if p.home:
                self._run_step(grbl, "$H", "Homing ($H)", timeout_s=180.0)

            # Modal setup
            for cmd, ctx, tmo in [
                ("G90", "Set absolute mode (G90)", 10.0),
                ("G21", "Set mm units (G21)", 10.0),
                ("G94", "Set feed/min (G94)", 10.0),
                ("M5", "Spindle off (M5)", 10.0),
            ]:
                self._run_step(grbl, cmd, ctx, timeout_s=tmo)

            # Move to world (0,0)
            self._run_step(
                grbl,
                f"G1 X0 Y0 F{p.travel_feed:.3f}",
                "Move to world (0, 0) to establish XY origin",
                timeout_s=180.0,
                wait_idle=True,
            )

            # Capture XY origin for display override
            grbl.capture_world_xy_zero_from_current_mpos(status_timeout_s=2.0)
            if grbl._wco_override:
                ox, oy, oz = grbl._wco_override
                self._log(
                    f"World XY origin captured (display override): WCO_override={ox:.3f},{oy:.3f},{oz:.3f}"
                )

            # Pre-probe retract (in current coordinate system)
            self._run_step(
                grbl,
                f"G0 Z{p.retract_z:.3f}",
                "Initial retract before Z-zero probe",
                timeout_s=30.0,
                wait_idle=True,
            )
            if p.settle_s > 0:
                time.sleep(p.settle_s)

            self._log("--- Establishing world Z=0 by probing at X=0 Y=0 ---")
            self._run_step(
                grbl, "G91", "Set relative mode (G91) for Z-zero probe", timeout_s=10.0
            )

            safe_travel = clamp_probe_travel_to_soft_limits(
                grbl, p.max_probe_travel, margin_mm=0.5, status_timeout_s=2.0
            )

            lines = self._run_step(
                grbl,
                f"G38.2 Z-{safe_travel:.3f} F{p.probe_feed:.3f}",
                "Probe down to establish Z=0 (G38.2)",
                timeout_s=180.0,
                wait_idle=True,
            )

            prb = grbl.get_probe_result(lines)
            if prb is None or not prb.success:
                raise RuntimeError(
                    "Z-zero probe did not report success (no PRB:...:1) at X=0 Y=0."
                )

            self._run_step(
                grbl,
                "G90",
                "Restore absolute mode (G90) after Z-zero probe",
                timeout_s=10.0,
            )

            # Capture Z=0 for display override, then apply G92 Z0 for motion
            grbl.set_world_z_zero_from_current_mpos(status_timeout_s=2.0)
            if grbl._wco_override:
                ox, oy, oz = grbl._wco_override
                self._log(
                    f"World Z=0 captured (display override): WCO_override={ox:.3f},{oy:.3f},{oz:.3f}"
                )

            self._run_step(
                grbl, "G92 Z0", "Set temporary world Z0 (G92 Z0)", timeout_s=10.0
            )

            # Retract in world coordinates
            self._run_step(
                grbl,
                f"G0 Z{p.retract_z:.3f}",
                "Retract after establishing world Z0",
                timeout_s=30.0,
                wait_idle=True,
            )

            # Build grid
            xs = build_axis_by_step_distance(p.start_x, p.width, p.step_distance_x)
            ys = build_axis_by_step_distance(p.start_y, p.height, p.step_distance_y)
            self._log(
                f"Grid: X samples={len(xs)} (step {p.step_distance_x}mm)  Y samples={len(ys)} (step {p.step_distance_y}mm)"
            )
            self._log(
                f"X range: {xs[0]:.3f} .. {xs[-1]:.3f}   Y range: {ys[0]:.3f} .. {ys[-1]:.3f}"
            )

            samples: List[SamplePoint] = []

            # Move to first sample point
            self._run_step(
                grbl,
                f"G1 X{xs[0]:.3f} Y{ys[0]:.3f} F{p.travel_feed:.3f}",
                "Move to first XY (world)",
                timeout_s=60.0,
                wait_idle=True,
            )
            if p.settle_s > 0:
                time.sleep(p.settle_s)

            for iy, y in enumerate(ys):
                row = list(enumerate(xs))
                if iy % 2 == 1:
                    row = list(reversed(row))

                for ix, x in row:
                    self._maybe_stop()
                    self._log(f"--- Sample ix={ix} iy={iy} at X={x:.3f} Y={y:.3f} ---")

                    self._run_step(
                        grbl,
                        f"G0 Z{p.retract_z:.3f}",
                        "Retract to safe Z (world)",
                        timeout_s=30.0,
                        wait_idle=True,
                    )

                    self._run_step(
                        grbl,
                        f"G1 X{x:.3f} Y{y:.3f} F{p.travel_feed:.3f}",
                        "Move to XY (world)",
                        timeout_s=60.0,
                        wait_idle=True,
                    )

                    if p.settle_s > 0:
                        time.sleep(p.settle_s)

                    self._run_step(
                        grbl, "G91", "Set relative mode (G91)", timeout_s=10.0
                    )

                    safe_travel = clamp_probe_travel_to_soft_limits(
                        grbl, p.max_probe_travel, margin_mm=0.5, status_timeout_s=2.0
                    )

                    lines = self._run_step(
                        grbl,
                        f"G38.2 Z-{safe_travel:.3f} F{p.probe_feed:.3f}",
                        "Probe down (G38.2)",
                        timeout_s=180.0,
                        wait_idle=True,
                    )

                    prb = grbl.get_probe_result(lines)
                    if prb is None or not prb.success:
                        raise RuntimeError(
                            f"Probe did not report success at ix={ix} iy={iy}."
                        )

                    self._run_step(
                        grbl, "G90", "Restore absolute mode (G90)", timeout_s=10.0
                    )

                    try:
                        _, _, wz = grbl.best_effort_wpos(timeout_s=2.0)
                        measured_z = wz
                    except Exception:
                        measured_z = prb.prb_z

                    self._run_step(
                        grbl,
                        f"G0 Z{p.retract_z:.3f}",
                        "Retract after probe (world)",
                        timeout_s=30.0,
                        wait_idle=True,
                    )

                    samples.append(SamplePoint(ix=ix, iy=iy, x=x, y=y, z=measured_z))

            # Write outputs
            self._write_csv(p.out_csv, samples)
            msg = f"Wrote CSV: {p.out_csv}"
            self._log(msg)

            if p.out_json:
                self._write_json(p.out_json, p, samples)
                self._log(f"Wrote JSON: {p.out_json}")

            zs = [pt.z for pt in samples]
            if zs:
                self._log(
                    f"Samples: {len(samples)}  Z(min/avg/max): {min(zs):.4f} / {sum(zs) / len(zs):.4f} / {max(zs):.4f} (mm)"
                )

            self.finished_ok.emit(msg)

        except Exception as ex:
            self.finished_error.emit(str(ex))
        finally:
            if grbl is not None:
                try:
                    # Best-effort cleanup (same spirit as original).
                    try:
                        grbl.send("$X", timeout_s=10.0)
                    except Exception:
                        pass
                    try:
                        grbl.send("G53 G0 Z0", timeout_s=30.0)
                    except Exception:
                        pass
                    try:
                        grbl.send("$H", timeout_s=180.0)
                    except Exception:
                        pass
                finally:
                    grbl.close()

    @staticmethod
    def _write_csv(path: str, points: List[SamplePoint]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("ix,iy,x_mm,y_mm,z_mm\n")
            for p in points:
                f.write(f"{p.ix},{p.iy},{p.x:.6f},{p.y:.6f},{p.z:.6f}\n")

    @staticmethod
    def _write_json(path: str, params: ProbeParams, points: List[SamplePoint]) -> None:
        steps_x = max((p.ix for p in points), default=-1) + 1
        steps_y = max((p.iy for p in points), default=-1) + 1
        payload = {
            "start_x": params.start_x,
            "start_y": params.start_y,
            "width": params.width,
            "height": params.height,
            "step_distance_x": params.step_distance_x,
            "step_distance_y": params.step_distance_y,
            "steps_x": steps_x,
            "steps_y": steps_y,
            "retract_z": params.retract_z,
            "max_probe_travel": params.max_probe_travel,
            "probe_feed": params.probe_feed,
            "travel_feed": params.travel_feed,
            "points": [p.__dict__ for p in points],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


class MainWindow(QMainWindow):
    # QSettings keys (kept simple and stable; values are plain strings/bools/ints).
    _SET_LAST_PORT = "connection/last_port"
    _SET_LAST_BAUD = "connection/last_baud"
    _SET_UNLOCK = "probe/send_unlock"
    _SET_HOME = "probe/send_home"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GRBL Heightmap Probe (Qt6)")

        # Settings are stored by the OS in the standard per-user location.
        # - Windows: Registry
        # - macOS:   plist under ~/Library/Preferences
        # - Linux:   ~/.config
        #
        # Using QSettings avoids managing our own config file and handles permissions/paths.
        self._settings = QSettings("MekatrolTools", "GrblHeightmapProbe")

        self._grbl_live: Optional[Grbl] = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._poll_status)

        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[ProbeWorker] = None

        # --- Connection UI
        self.port_combo = QComboBox()
        self.port_combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        self.btn_refresh_ports = QPushButton("Refresh")
        self.btn_refresh_ports.clicked.connect(self._refresh_ports)

        self.baud_edit = QLineEdit("115200")

        self.btn_connect = QPushButton("Connect")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)

        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)

        conn_box = QGroupBox("Connection")
        conn_form = QFormLayout(conn_box)
        port_row = QHBoxLayout()
        port_row.addWidget(self.port_combo, 1)
        port_row.addWidget(self.btn_refresh_ports, 0)
        conn_form.addRow("Port:", port_row)
        conn_form.addRow("Baud:", self.baud_edit)

        conn_btns = QHBoxLayout()
        conn_btns.addWidget(self.btn_connect)
        conn_btns.addWidget(self.btn_disconnect)
        conn_form.addRow(conn_btns)

        # --- Position display widgets (MPos / WCO)
        self.lbl_state = QLabel("State: ?")
        self.lbl_mpos = QLabel("MPos: X=— Y=— Z=—")
        self.lbl_wco = QLabel("WCO: X=— Y=— Z=—")
        self.lbl_wpos = QLabel("WPos: X=— Y=— Z=—")

        pos_box = QGroupBox("Status")
        pos_layout = QVBoxLayout(pos_box)
        pos_layout.addWidget(self.lbl_state)
        pos_layout.addWidget(self.lbl_mpos)
        pos_layout.addWidget(self.lbl_wco)
        pos_layout.addWidget(self.lbl_wpos)

        # --- Parameters
        params_box = QGroupBox("Probe Parameters")
        grid = QGridLayout(params_box)

        def spin(
            value: float,
            step: float,
            decimals: int = 3,
            minimum: float = -1e9,
            maximum: float = 1e9,
        ) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setDecimals(decimals)
            s.setRange(minimum, maximum)
            s.setSingleStep(step)
            s.setValue(value)
            return s

        self.start_x = spin(0.0, 1.0)
        self.start_y = spin(0.0, 1.0)
        self.width = spin(10.0, 1.0, minimum=0.0)
        self.height = spin(10.0, 1.0, minimum=0.0)
        self.step_x = spin(1.0, 0.1, minimum=0.001)
        self.step_y = spin(1.0, 0.1, minimum=0.001)
        self.retract_z = spin(5.0, 0.1)
        self.max_probe = spin(10.0, 0.1, minimum=0.001)
        self.probe_feed = spin(60.0, 10.0, minimum=0.001)
        self.travel_feed = spin(600.0, 10.0, minimum=0.001)
        self.settle = spin(0.1, 0.05, decimals=2, minimum=0.0)

        self.chk_unlock = QCheckBox("Send $X (unlock)")
        self.chk_home = QCheckBox("Send $H (home)")

        self.out_csv = QLineEdit("heightmap.csv")
        self.btn_browse_csv = QPushButton("Browse…")
        self.btn_browse_csv.clicked.connect(self._browse_csv)

        self.out_json = QLineEdit("")
        self.btn_browse_json = QPushButton("Browse…")
        self.btn_browse_json.clicked.connect(self._browse_json)

        row = 0
        grid.addWidget(QLabel("Start X (world, mm)"), row, 0)
        grid.addWidget(self.start_x, row, 1)
        row += 1
        grid.addWidget(QLabel("Start Y (world, mm)"), row, 0)
        grid.addWidget(self.start_y, row, 1)
        row += 1
        grid.addWidget(QLabel("Width (+X, mm)"), row, 0)
        grid.addWidget(self.width, row, 1)
        row += 1
        grid.addWidget(QLabel("Height (+Y, mm)"), row, 0)
        grid.addWidget(self.height, row, 1)
        row += 1
        grid.addWidget(QLabel("Step X (mm)"), row, 0)
        grid.addWidget(self.step_x, row, 1)
        row += 1
        grid.addWidget(QLabel("Step Y (mm)"), row, 0)
        grid.addWidget(self.step_y, row, 1)
        row += 1
        grid.addWidget(QLabel("Retract Z (world, mm)"), row, 0)
        grid.addWidget(self.retract_z, row, 1)
        row += 1
        grid.addWidget(QLabel("Max probe travel (mm)"), row, 0)
        grid.addWidget(self.max_probe, row, 1)
        row += 1
        grid.addWidget(QLabel("Probe feed (mm/min)"), row, 0)
        grid.addWidget(self.probe_feed, row, 1)
        row += 1
        grid.addWidget(QLabel("Travel feed (mm/min)"), row, 0)
        grid.addWidget(self.travel_feed, row, 1)
        row += 1
        grid.addWidget(QLabel("Settle (s)"), row, 0)
        grid.addWidget(self.settle, row, 1)
        row += 1
        grid.addWidget(self.chk_unlock, row, 0, 1, 2)
        row += 1
        grid.addWidget(self.chk_home, row, 0, 1, 2)
        row += 1

        out_row = QHBoxLayout()
        out_row.addWidget(self.out_csv)
        out_row.addWidget(self.btn_browse_csv)
        grid.addWidget(QLabel("Output CSV"), row, 0)
        grid.addLayout(out_row, row, 1)
        row += 1

        outj_row = QHBoxLayout()
        outj_row.addWidget(self.out_json)
        outj_row.addWidget(self.btn_browse_json)
        grid.addWidget(QLabel("Output JSON (optional)"), row, 0)
        grid.addLayout(outj_row, row, 1)

        # --- Run controls + log
        self.btn_start = QPushButton("Start Probe")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self._start_probe)
        self.btn_stop.clicked.connect(self._stop_probe)

        run_row = QHBoxLayout()
        run_row.addWidget(self.btn_start)
        run_row.addWidget(self.btn_stop)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # --- Layout
        root = QWidget()
        main = QVBoxLayout(root)

        top = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(conn_box)
        left.addWidget(pos_box)
        left.addStretch(1)

        right = QVBoxLayout()
        right.addWidget(params_box)
        right.addLayout(run_row)
        right.addWidget(QLabel("Log"))
        right.addWidget(self.log)

        top.addLayout(left, 0)
        top.addLayout(right, 1)

        main.addLayout(top)
        self.setCentralWidget(root)

        # --- Persistent UI state (port, baud, $X/$H checkboxes)
        # Load saved values before first refresh so the refresh can select the saved port.
        self._restore_settings()

        # Persist changes as the user edits (so a crash/power loss still keeps last values).
        self.port_combo.currentIndexChanged.connect(self._save_settings)
        self.baud_edit.editingFinished.connect(self._save_settings)
        self.chk_unlock.toggled.connect(self._save_settings)
        self.chk_home.toggled.connect(self._save_settings)

        self._refresh_ports()

    def _restore_settings(self) -> None:
        """
        Restore the last-used connection choices and probe toggles.

        Notes:
        - The actual port selection is applied during _refresh_ports() after ports are enumerated.
        - If the saved baud is invalid (non-int), we fall back to the default displayed value.
        """
        # Restore baud (text field).
        baud = self._settings.value(self._SET_LAST_BAUD, "", type=str)
        if baud:
            # Keep it as text; we'll validate/parse when connecting.
            self.baud_edit.setText(baud)

        # Restore $X/$H checkboxes.
        self.chk_unlock.setChecked(bool(self._settings.value(self._SET_UNLOCK, False, type=bool)))
        self.chk_home.setChecked(bool(self._settings.value(self._SET_HOME, False, type=bool)))

    def _save_settings(self) -> None:
        """
        Save the currently selected connection choices and probe toggles.

        This is called:
        - when the user changes port selection
        - when baud editing finishes
        - when $X/$H checkboxes toggle
        - on window close (closeEvent) as a final best-effort write
        """
        port = (self.port_combo.currentData() or "").strip()
        # Save baud exactly as typed (lets the user keep partial edits; we validate at connect).
        baud = self.baud_edit.text().strip()

        self._settings.setValue(self._SET_LAST_PORT, port)
        self._settings.setValue(self._SET_LAST_BAUD, baud)
        self._settings.setValue(self._SET_UNLOCK, self.chk_unlock.isChecked())
        self._settings.setValue(self._SET_HOME, self.chk_home.isChecked())
        self._settings.sync()

    def _list_serial_ports(self) -> List[Tuple[str, str]]:
        """
        Returns list of (device, display_name).
        Works on Windows/macOS/Linux via pyserial's list_ports.
        """
        out: List[Tuple[str, str]] = []
        for p in list_ports.comports():
            device = p.device  # e.g. COM3, /dev/ttyUSB0, /dev/cu.usbserial-...
            desc = p.description or ""
            hwid = p.hwid or ""
            display = f"{device} — {desc}".strip()
            if hwid:
                display = f"{display} ({hwid})"
            out.append((device, display))
        out.sort(key=lambda t: t[0].lower())
        return out

    @Slot()
    def _refresh_ports(self) -> None:
        current = self.port_combo.currentData()
        ports = self._list_serial_ports()

        self.port_combo.blockSignals(True)
        self.port_combo.clear()

        for device, display in ports:
            self.port_combo.addItem(display, device)

        # Fallback entry if nothing found (lets user still type nothing?).
        # If you want manual entry, use QComboBox.setEditable(True).
        if not ports:
            self.port_combo.addItem("(No ports found)", "")

        # Restore prior selection if still present (1) current UI selection, else (2) saved selection.
        restore_port = (current or "").strip()
        if not restore_port:
            restore_port = (self._settings.value(self._SET_LAST_PORT, "", type=str) or "").strip()

        if restore_port:
            idx = self.port_combo.findData(restore_port)
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)

        self.port_combo.blockSignals(False)

        # Now that refresh has happened, persist the selected port (if any).
        self._save_settings()

    def _append_log(self, s: str) -> None:
        self.log.append(s)

    def _set_buttons_connected(self, connected: bool) -> None:
        self.btn_connect.setEnabled(not connected and self._worker_thread is None)
        self.btn_disconnect.setEnabled(connected and self._worker_thread is None)
        self.btn_start.setEnabled(connected and self._worker_thread is None)
        self.btn_stop.setEnabled(self._worker_thread is not None)

    @Slot()
    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Select CSV Output", self.out_csv.text(), "CSV (*.csv);;All Files (*)"
        )
        if path:
            self.out_csv.setText(path)

    @Slot()
    def _browse_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select JSON Output",
            self.out_json.text(),
            "JSON (*.json);;All Files (*)",
        )
        if path:
            self.out_json.setText(path)

    @Slot()
    def _connect(self) -> None:
        # UI-side exceptions should never close the app.
        try:
            if self._grbl_live is not None:
                return
            port = (self.port_combo.currentData() or self.port_combo.currentText()).strip()
            if not port:
                QMessageBox.warning(
                    self, "Missing port", "Enter a serial port (e.g. COM3 or /dev/ttyUSB0)."
                )
                return
            try:
                baud = int(self.baud_edit.text().strip())
            except Exception:
                QMessageBox.warning(
                    self, "Invalid baud", "Baud must be an integer (e.g. 115200)."
                )
                return

            try:
                self._grbl_live = Grbl(port, baud)
                self._grbl_live.wake()
                self._append_log(f"Connected (status polling): {port} @ {baud}")
                self._poll_timer.start()
                self._set_buttons_connected(True)

                # Persist last known-good selections.
                self._save_settings()
            except Exception as ex:
                self._grbl_live = None
                QMessageBox.critical(self, "Connect failed", str(ex))
        except Exception as ex:
            # Catch-all: do not allow unexpected slot exceptions to kill the app.
            QMessageBox.critical(self, "Unexpected error", str(ex))

    @Slot()
    def _disconnect(self) -> None:
        # UI-side exceptions should never close the app.
        try:
            self._poll_timer.stop()
            if self._grbl_live is not None:
                try:
                    self._grbl_live.close()
                finally:
                    self._grbl_live = None
            self._append_log("Disconnected.")
            self.lbl_state.setText("State: ?")
            self.lbl_mpos.setText("MPos: X=— Y=— Z=—")
            self.lbl_wco.setText("WCO: X=— Y=— Z=—")
            self.lbl_wpos.setText("WPos: X=— Y=— Z=—")
            self._set_buttons_connected(False)
        except Exception as ex:
            QMessageBox.critical(self, "Unexpected error", str(ex))

    @Slot()
    def _poll_status(self) -> None:
        if self._grbl_live is None:
            return
        try:
            status = self._grbl_live.query_status_line(timeout_s=0.5)
            st_m = STATUS_STATE_RE.search(status)
            state = st_m.group(1) if st_m else "?"
            sp = self._grbl_live.get_status_pos(timeout_s=0.5)

            self.lbl_state.setText(f"State: {state}")

            if sp.mpos is not None:
                mx, my, mz = sp.mpos
                self.lbl_mpos.setText(f"MPos: X={mx:.3f} Y={my:.3f} Z={mz:.3f}")
            else:
                self.lbl_mpos.setText("MPos: X=— Y=— Z=—")

            wco_eff = self._grbl_live.best_effort_wco(timeout_s=0.5)
            if wco_eff is not None:
                ox, oy, oz = wco_eff
                self.lbl_wco.setText(f"WCO: X={ox:.3f} Y={oy:.3f} Z={oz:.3f}")
            else:
                self.lbl_wco.setText("WCO: X=— Y=— Z=—")

            try:
                wx, wy, wz = self._grbl_live.best_effort_wpos(timeout_s=0.5)
                self.lbl_wpos.setText(f"WPos: X={wx:.3f} Y={wy:.3f} Z={wz:.3f}")
            except Exception:
                self.lbl_wpos.setText("WPos: X=— Y=— Z=—")

        except Exception:
            # Keep UI stable; transient serial hiccups are common.
            pass

    def _build_params(self) -> ProbeParams:
        return ProbeParams(
            start_x=float(self.start_x.value()),
            start_y=float(self.start_y.value()),
            width=float(self.width.value()),
            height=float(self.height.value()),
            step_distance_x=float(self.step_x.value()),
            step_distance_y=float(self.step_y.value()),
            retract_z=float(self.retract_z.value()),
            max_probe_travel=float(self.max_probe.value()),
            probe_feed=float(self.probe_feed.value()),
            travel_feed=float(self.travel_feed.value()),
            settle_s=float(self.settle.value()),
            unlock=self.chk_unlock.isChecked(),
            home=self.chk_home.isChecked(),
            out_csv=self.out_csv.text().strip() or "heightmap.csv",
            out_json=self.out_json.text().strip(),
        )

    @Slot()
    def _start_probe(self) -> None:
        # UI-side exceptions should never close the app.
        try:
            if self._worker_thread is not None:
                return
            if self._grbl_live is None:
                QMessageBox.warning(self, "Not connected", "Connect to GRBL first.")
                return

            # Stop polling while the worker owns the port.
            self._poll_timer.stop()

            port = (self.port_combo.currentData() or self.port_combo.currentText()).strip()
            baud = int(self.baud_edit.text().strip())
            params = self._build_params()

            # Persist the choices used for this run (port/baud + $X/$H).
            self._save_settings()

            # Close live polling connection; worker will open its own serial instance.
            try:
                self._grbl_live.close()
            except Exception:
                pass
            self._grbl_live = None

            self._append_log("Starting probe job...")

            thread = QThread(self)
            worker = ProbeWorker(port, baud, params)
            worker.moveToThread(thread)

            worker.log.connect(self._append_log)
            worker.telemetry.connect(self._on_worker_telemetry)
            worker.finished_ok.connect(self._on_worker_ok)
            worker.finished_error.connect(self._on_worker_error)

            thread.started.connect(worker.run)
            thread.finished.connect(thread.deleteLater)

            self._worker_thread = thread
            self._worker = worker

            self._set_buttons_connected(False)
            self.btn_stop.setEnabled(True)

            thread.start()
        except Exception as ex:
            QMessageBox.critical(self, "Unexpected error", str(ex))

    @Slot()
    def _stop_probe(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._append_log("Stop requested...")

    @Slot(object)
    def _on_worker_telemetry(self, data: object) -> None:
        if not isinstance(data, dict):
            return
        state = data.get("state", "?")
        self.lbl_state.setText(f"State: {state}")

        mpos = data.get("mpos", None)
        if isinstance(mpos, tuple) and len(mpos) == 3:
            mx, my, mz = mpos
            self.lbl_mpos.setText(f"MPos: X={mx:.3f} Y={my:.3f} Z={mz:.3f}")

        wco = data.get("wco", None)
        if isinstance(wco, tuple) and len(wco) == 3:
            ox, oy, oz = wco
            self.lbl_wco.setText(f"WCO: X={ox:.3f} Y={oy:.3f} Z={oz:.3f}")

        wpos = data.get("wpos", None)
        if isinstance(wpos, tuple) and len(wpos) == 3:
            wx, wy, wz = wpos
            self.lbl_wpos.setText(f"WPos: X={wx:.3f} Y={wy:.3f} Z={wz:.3f}")

    @Slot(str)
    def _on_worker_ok(self, msg: str) -> None:
        # Worker completion should never close the app, even if message boxes fail.
        try:
            self._append_log(msg)
            self._cleanup_worker()
            QMessageBox.information(self, "Done", msg)
        except Exception as ex:
            self._append_log(f"ERROR (UI): {ex}")
            self._cleanup_worker()

    @Slot(str)
    def _on_worker_error(self, err: str) -> None:
        # Requirement: do not close the app if an error occurs.
        # Probe errors are handled here, logged, and shown to the user, but the app stays open.
        try:
            self._append_log(f"ERROR: {err}")
            self._cleanup_worker()
            QMessageBox.critical(self, "Probe failed", err)
        except Exception as ex:
            self._append_log(f"ERROR (UI): {ex}")
            self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        self._worker_thread = None
        self._worker = None

        # After job, user must reconnect for polling (keeps ownership clear).
        self._set_buttons_connected(False)
        self.btn_stop.setEnabled(False)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # Save settings on exit as a final best-effort (we also save on change).
        try:
            self._save_settings()
        except Exception:
            pass

        try:
            self._poll_timer.stop()
        except Exception:
            pass
        try:
            if self._worker is not None:
                self._worker.request_stop()
        except Exception:
            pass
        try:
            if self._worker_thread is not None:
                self._worker_thread.quit()
                self._worker_thread.wait(1000)
        except Exception:
            pass
        try:
            if self._grbl_live is not None:
                self._grbl_live.close()
        except Exception:
            pass
        super().closeEvent(event)


def _install_qt_exception_hook(app: QApplication) -> None:
    """
    Install a global exception hook to prevent unexpected UI-thread exceptions
    from silently killing the process.

    This does not "fix" logic errors, but it ensures the user sees a dialog and
    the app remains open whenever possible.
    """
    old_hook = sys.excepthook

    def _hook(exctype, value, tb) -> None:
        try:
            msg = f"{exctype.__name__}: {value}"
            QMessageBox.critical(None, "Unhandled exception", msg)
        except Exception:
            # If Qt is not in a good state, fall back to the default hook.
            pass
        old_hook(exctype, value, tb)

    sys.excepthook = _hook


def main() -> int:
    app = QApplication(sys.argv)

    # Keep the application resilient: show a dialog on unhandled exceptions rather than just exiting.
    _install_qt_exception_hook(app)

    w = MainWindow()
    w.resize(1100, 750)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
