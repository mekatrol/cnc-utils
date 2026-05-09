from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import serial
from PySide6.QtCore import QObject, Signal, Slot

PRB_RE = re.compile(r"PRB:([-\d.]+),([-\d.]+),([-\d.]+):([01])")
SETTING_RE = re.compile(r"^\$(\d+)=([-\d.]+)\s*$")
STATUS_MPOS_RE = re.compile(r"(?:^|[|])MPos:([-\d.]+),([-\d.]+),([-\d.]+)(?:[|>]|$)")
STATUS_STATE_RE = re.compile(r"^<([^|>]+)")
STATUS_WCO_RE = re.compile(r"(?:^|[|])WCO:([-\d.]+),([-\d.]+),([-\d.]+)(?:[|>]|$)")
STATUS_WPOS_RE = re.compile(r"(?:^|[|])WPos:([-\d.]+),([-\d.]+),([-\d.]+)(?:[|>]|$)")


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
    out_json: str = ""


@dataclass(frozen=True)
class ProbeResult:
    prb_x: float
    prb_y: float
    prb_z: float
    success: bool


@dataclass(frozen=True)
class SamplePoint:
    ix: int
    iy: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class StatusPos:
    mpos: tuple[float, float, float] | None
    wpos: tuple[float, float, float] | None
    wco: tuple[float, float, float] | None


class GrblProbeConnection:
    def __init__(self, port: str, baud: int, timeout: float = 0.25) -> None:
        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=timeout,
            write_timeout=timeout,
        )
        self._wco_override: tuple[float, float, float] | None = None
        self._wco: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def close(self) -> None:
        try:
            if self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

    def wake(self, should_stop) -> None:
        self.ser.reset_input_buffer()
        self.ser.write(b"\r\n\r\n")
        self.ser.flush()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            should_stop()
            line = self._readline()
            if not line:
                continue

    def _readline(self) -> str:
        return self.ser.readline().decode("ascii", errors="replace").strip()

    def feed_hold(self) -> None:
        try:
            self.ser.write(b"!")
            self.ser.flush()
        except Exception:
            pass

    def send(self, cmd: str, *, timeout_s: float, should_stop) -> list[str]:
        cmd = cmd.strip()
        if not cmd:
            return []
        deadline = time.monotonic() + timeout_s
        self.ser.write((cmd + "\n").encode("ascii"))
        self.ser.flush()
        lines: list[str] = []
        while True:
            should_stop()
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out after {timeout_s:.1f}s: {cmd!r}")
            line = self._readline()
            if not line:
                continue
            low = line.lower()
            if low == "ok":
                return lines
            if low.startswith("error:") or low.startswith("alarm:"):
                raise RuntimeError(f"{line} (while running {cmd!r})")
            lines.append(line)

    def query_status_line(self, *, timeout_s: float, should_stop) -> str:
        deadline = time.monotonic() + timeout_s
        self.ser.write(b"?")
        self.ser.flush()
        while True:
            should_stop()
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for status.")
            line = self._readline()
            if line.startswith("<") and line.endswith(">"):
                return line

    def get_status_pos(self, *, timeout_s: float, should_stop) -> StatusPos:
        status = self.query_status_line(timeout_s=timeout_s, should_stop=should_stop)
        mpos = None
        wpos = None
        m = STATUS_MPOS_RE.search(status)
        if m:
            mpos = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        w = STATUS_WPOS_RE.search(status)
        if w:
            wpos = (float(w.group(1)), float(w.group(2)), float(w.group(3)))
        c = STATUS_WCO_RE.search(status)
        if c:
            self._wco = (float(c.group(1)), float(c.group(2)), float(c.group(3)))
        return StatusPos(mpos=mpos, wpos=wpos, wco=self._wco)

    def capture_world_xy_zero_from_current_mpos(self, should_stop) -> None:
        sp = self.get_status_pos(timeout_s=2.0, should_stop=should_stop)
        if sp.mpos is None:
            raise RuntimeError("Cannot capture world XY=0: status did not include MPos.")
        mx, my, _ = sp.mpos
        oz = self._wco_override[2] if self._wco_override is not None else 0.0
        self._wco_override = (mx, my, oz)

    def set_world_z_zero_from_current_mpos(self, should_stop) -> None:
        sp = self.get_status_pos(timeout_s=2.0, should_stop=should_stop)
        if sp.mpos is None:
            raise RuntimeError("Cannot set world Z=0: status did not include MPos.")
        mx, my, mz = sp.mpos
        ox = self._wco_override[0] if self._wco_override is not None else mx
        oy = self._wco_override[1] if self._wco_override is not None else my
        self._wco_override = (ox, oy, mz)

    def best_effort_wpos(self, should_stop) -> tuple[float, float, float]:
        sp = self.get_status_pos(timeout_s=2.0, should_stop=should_stop)
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
        raise RuntimeError("No usable WPos source.")

    def best_effort_wco(self, should_stop) -> tuple[float, float, float] | None:
        sp = self.get_status_pos(timeout_s=1.0, should_stop=should_stop)
        if sp.wco is not None:
            return sp.wco
        return self._wco_override

    def read_setting(self, setting_num: int, should_stop) -> float:
        lines = self.send("$$", timeout_s=5.0, should_stop=should_stop)
        for line in lines:
            match = SETTING_RE.match(line.strip())
            if match and int(match.group(1)) == setting_num:
                return float(match.group(2))
        raise RuntimeError(f"GRBL did not return ${setting_num}.")

    def get_probe_result(self, response_lines: list[str]) -> ProbeResult | None:
        for line in response_lines:
            match = PRB_RE.search(line)
            if match:
                x, y, z, success = match.groups()
                return ProbeResult(float(x), float(y), float(z), success == "1")
        return None


class HeightMapProbeWorker(QObject):
    log = Signal(str)
    telemetry = Signal(object)
    finished_ok = Signal(str)
    finished_error = Signal(str)
    canceled = Signal(str)

    def __init__(self, port: str, baud: int, params: ProbeParams) -> None:
        super().__init__()
        self._port = port
        self._baud = baud
        self._params = params
        self._stop_requested = False
        self._grbl: GrblProbeConnection | None = None

    @Slot()
    def request_stop(self) -> None:
        self._stop_requested = True
        if self._grbl is not None:
            self._grbl.feed_hold()

    @Slot()
    def run(self) -> None:
        grbl: GrblProbeConnection | None = None
        try:
            grbl = GrblProbeConnection(self._port, self._baud)
            self._grbl = grbl
            grbl.wake(self._check_stop)
            self.log.emit(f"Connected: {self._port} @ {self._baud}")
            samples = self._probe(grbl)
            self._write_csv(Path(self._params.out_csv), samples)
            if self._params.out_json:
                self._write_json(Path(self._params.out_json), samples)
            self.finished_ok.emit(f"Wrote CSV: {self._params.out_csv}")
        except _ProbeCanceled as exc:
            self.canceled.emit(str(exc))
        except Exception as exc:
            self.finished_error.emit(str(exc))
        finally:
            if grbl is not None:
                self._cleanup(grbl)
                grbl.close()
            self._grbl = None

    def _check_stop(self) -> None:
        if self._stop_requested:
            raise _ProbeCanceled("Height-map probe canceled.")

    def _probe(self, grbl: GrblProbeConnection) -> list[SamplePoint]:
        p = self._params
        if p.unlock:
            self._run_step(grbl, "$X", "Unlock ($X)", timeout_s=10.0)
        if p.home:
            self._run_step(grbl, "$H", "Homing ($H)", timeout_s=180.0)
        for cmd, context, timeout in (
            ("G90", "Set absolute mode (G90)", 10.0),
            ("G21", "Set mm units (G21)", 10.0),
            ("G94", "Set feed/min (G94)", 10.0),
            ("M5", "Spindle off (M5)", 10.0),
        ):
            self._run_step(grbl, cmd, context, timeout_s=timeout)
        self._run_step(
            grbl,
            f"G1 X0 Y0 F{p.travel_feed:.3f}",
            "Move to world (0, 0) to establish XY origin",
            timeout_s=180.0,
            wait_idle=True,
        )
        grbl.capture_world_xy_zero_from_current_mpos(self._check_stop)
        self._run_step(
            grbl,
            f"G0 Z{p.retract_z:.3f}",
            "Initial retract before Z-zero probe",
            timeout_s=30.0,
            wait_idle=True,
        )
        self._sleep(p.settle_s)
        self._run_step(grbl, "G91", "Set relative mode (G91)", timeout_s=10.0)
        safe_travel = self._safe_probe_travel(grbl)
        lines = self._run_step(
            grbl,
            f"G38.2 Z-{safe_travel:.3f} F{p.probe_feed:.3f}",
            "Probe down to establish Z=0 (G38.2)",
            timeout_s=180.0,
            wait_idle=True,
        )
        probe_result = grbl.get_probe_result(lines)
        if probe_result is None or not probe_result.success:
            raise RuntimeError("Z-zero probe did not report success at X=0 Y=0.")
        self._run_step(grbl, "G90", "Restore absolute mode (G90)", timeout_s=10.0)
        grbl.set_world_z_zero_from_current_mpos(self._check_stop)
        self._run_step(grbl, "G92 Z0", "Set temporary world Z0", timeout_s=10.0)
        self._run_step(
            grbl,
            f"G0 Z{p.retract_z:.3f}",
            "Retract after establishing world Z0",
            timeout_s=30.0,
            wait_idle=True,
        )
        xs = _build_axis(p.start_x, p.width, p.step_distance_x)
        ys = _build_axis(p.start_y, p.height, p.step_distance_y)
        self.log.emit(f"Grid: X samples={len(xs)} Y samples={len(ys)}")
        samples: list[SamplePoint] = []
        self._run_step(
            grbl,
            f"G1 X{xs[0]:.3f} Y{ys[0]:.3f} F{p.travel_feed:.3f}",
            "Move to first XY",
            timeout_s=60.0,
            wait_idle=True,
        )
        self._sleep(p.settle_s)
        for iy, y in enumerate(ys):
            row = list(enumerate(xs))
            if iy % 2 == 1:
                row.reverse()
            for ix, x in row:
                self._check_stop()
                self.log.emit(f"Sample ix={ix} iy={iy} X={x:.3f} Y={y:.3f}")
                self._run_step(
                    grbl,
                    f"G0 Z{p.retract_z:.3f}",
                    "Retract to safe Z",
                    timeout_s=30.0,
                    wait_idle=True,
                )
                self._run_step(
                    grbl,
                    f"G1 X{x:.3f} Y{y:.3f} F{p.travel_feed:.3f}",
                    "Move to XY",
                    timeout_s=60.0,
                    wait_idle=True,
                )
                self._sleep(p.settle_s)
                self._run_step(grbl, "G91", "Set relative mode", timeout_s=10.0)
                safe_travel = self._safe_probe_travel(grbl)
                lines = self._run_step(
                    grbl,
                    f"G38.2 Z-{safe_travel:.3f} F{p.probe_feed:.3f}",
                    "Probe down",
                    timeout_s=180.0,
                    wait_idle=True,
                )
                probe_result = grbl.get_probe_result(lines)
                if probe_result is None or not probe_result.success:
                    raise RuntimeError(f"Probe failed at ix={ix} iy={iy}.")
                self._run_step(grbl, "G90", "Restore absolute mode", timeout_s=10.0)
                try:
                    _, _, measured_z = grbl.best_effort_wpos(self._check_stop)
                except Exception:
                    measured_z = probe_result.prb_z
                self._run_step(
                    grbl,
                    f"G0 Z{p.retract_z:.3f}",
                    "Retract after probe",
                    timeout_s=30.0,
                    wait_idle=True,
                )
                samples.append(SamplePoint(ix=ix, iy=iy, x=x, y=y, z=measured_z))
        return samples

    def _run_step(
        self,
        grbl: GrblProbeConnection,
        cmd: str,
        context: str,
        *,
        timeout_s: float,
        wait_idle: bool = False,
        idle_timeout_s: float = 30.0,
    ) -> list[str]:
        self._check_stop()
        self.log.emit(f"Executing: {cmd}")
        try:
            lines = grbl.send(cmd, timeout_s=timeout_s, should_stop=self._check_stop)
            if wait_idle:
                self._wait_idle(grbl, idle_timeout_s)
            return lines
        except _ProbeCanceled:
            raise
        except Exception as exc:
            raise RuntimeError(f"{context}: {exc}") from exc

    def _wait_idle(self, grbl: GrblProbeConnection, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            self._check_stop()
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for Idle.")
            status = grbl.query_status_line(timeout_s=2.0, should_stop=self._check_stop)
            match = STATUS_STATE_RE.search(status)
            state = match.group(1) if match else "?"
            self._emit_telemetry(grbl, state)
            if state == "Idle":
                return
            if state == "Alarm":
                raise RuntimeError("Controller entered ALARM while waiting for Idle.")
            self._sleep(0.05)

    def _emit_telemetry(self, grbl: GrblProbeConnection, state: str) -> None:
        try:
            sp = grbl.get_status_pos(timeout_s=1.0, should_stop=self._check_stop)
            try:
                wpos = grbl.best_effort_wpos(self._check_stop)
            except Exception:
                wpos = None
            self.telemetry.emit(
                {"state": state, "mpos": sp.mpos, "wco": sp.wco, "wpos": wpos}
            )
        except Exception:
            pass

    def _safe_probe_travel(self, grbl: GrblProbeConnection) -> float:
        p = self._params
        sp = grbl.get_status_pos(timeout_s=2.0, should_stop=self._check_stop)
        if sp.mpos is None:
            raise RuntimeError("Status did not include MPos; cannot clamp soft limits.")
        _, _, mpos_z = sp.mpos
        z_max_travel = grbl.read_setting(132, self._check_stop)
        z_min = -z_max_travel
        max_down = mpos_z - z_min
        safe_travel = min(p.max_probe_travel, max(0.0, max_down - 0.5))
        if safe_travel <= 0.0:
            raise RuntimeError(f"No safe Z probe travel available at MPosZ={mpos_z:.3f}.")
        return safe_travel

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            self._check_stop()
            time.sleep(min(0.05, deadline - time.monotonic()))

    def _cleanup(self, grbl: GrblProbeConnection) -> None:
        for cmd, timeout_s in (("$X", 10.0), ("G53 G0 Z0", 30.0), ("$H", 180.0)):
            try:
                grbl.send(cmd, timeout_s=timeout_s, should_stop=lambda: None)
            except Exception:
                pass

    def _write_csv(self, path: Path, points: list[SamplePoint]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as csv_file:
            csv_file.write("ix,iy,x_mm,y_mm,z_mm\n")
            for point in points:
                csv_file.write(
                    f"{point.ix},{point.iy},{point.x:.6f},"
                    f"{point.y:.6f},{point.z:.6f}\n"
                )

    def _write_json(self, path: Path, points: list[SamplePoint]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "points": [point.__dict__ for point in points],
            "width": self._params.width,
            "height": self._params.height,
            "step_distance_x": self._params.step_distance_x,
            "step_distance_y": self._params.step_distance_y,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class _ProbeCanceled(Exception):
    pass


def _build_axis(start: float, length: float, step_distance: float) -> list[float]:
    if step_distance <= 0.0:
        raise ValueError("step distance must be greater than 0.")
    if length < 0.0:
        raise ValueError("length must be greater than or equal to 0.")
    end = start + length
    if length == 0.0:
        return [start]
    points = [start]
    index = 1
    while True:
        value = start + (index * step_distance)
        if value >= end:
            break
        points.append(value)
        index += 1
    points.append(end)
    return points
