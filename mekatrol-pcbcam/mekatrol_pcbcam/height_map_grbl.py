from __future__ import annotations

import re
import time

import serial

STATUS_MPOS_RE = re.compile(r"MPos:([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)")
STATUS_STATE_RE = re.compile(r"^<([^|>]+)")
STATUS_WCO_RE = re.compile(r"WCO:([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)")
STATUS_WPOS_RE = re.compile(r"WPos:([-+]?\d*\.?\d+),([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)")


class HeightMapGrbl:
    def __init__(self, port: str, baud: int, timeout: float = 2.0) -> None:
        self._serial = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=timeout,
            write_timeout=timeout,
        )

    def wake(self) -> None:
        self._serial.write(b"\r\n\r\n")
        self._serial.flush()
        time.sleep(1.5)
        self._serial.reset_input_buffer()

    def close(self) -> None:
        self._serial.close()

    def status(self) -> tuple[str, str, str, str]:
        status_line = self._query_status_line()
        state_match = STATUS_STATE_RE.search(status_line)
        state = state_match.group(1) if state_match else "?"
        mpos = self._format_position("MPos", STATUS_MPOS_RE.search(status_line))
        wco = self._format_position("WCO", STATUS_WCO_RE.search(status_line))
        wpos_match = STATUS_WPOS_RE.search(status_line)
        if wpos_match is not None:
            wpos = self._format_position("WPos", wpos_match)
        else:
            wpos = self._best_effort_wpos(status_line)
        return f"State: {state}", mpos, wco, wpos

    def _query_status_line(self) -> str:
        self._serial.write(b"?")
        self._serial.flush()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            raw = self._serial.readline().decode(errors="replace").strip()
            if raw.startswith("<") and raw.endswith(">"):
                return raw
        raise TimeoutError("Timed out waiting for status.")

    def _best_effort_wpos(self, status_line: str) -> str:
        mpos_match = STATUS_MPOS_RE.search(status_line)
        wco_match = STATUS_WCO_RE.search(status_line)
        if mpos_match is None or wco_match is None:
            return "WPos: X=- Y=- Z=-"
        mx, my, mz = self._match_values(mpos_match)
        ox, oy, oz = self._match_values(wco_match)
        return f"WPos: X={mx - ox:.3f} Y={my - oy:.3f} Z={mz - oz:.3f}"

    def _format_position(self, label: str, match: re.Match[str] | None) -> str:
        if match is None:
            return f"{label}: X=- Y=- Z=-"
        x, y, z = self._match_values(match)
        return f"{label}: X={x:.3f} Y={y:.3f} Z={z:.3f}"

    def _match_values(self, match: re.Match[str]) -> tuple[float, float, float]:
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
