#!/usr/bin/env python3
"""PRIVES hardware telemetry daemon for Raspberry Pi 5 + Hailo-8 (AI HAT+).

Samples CPU, memory, SoC temperature, throttling state, PMIC power and NPU
statistics at a fixed period (default 500 ms) and appends them to a CSV file.

Relative output paths are resolved against ``benchmarks/results/``.
The daemon stops cleanly (flushing and closing the CSV) on SIGINT/SIGTERM.

The first 16 columns follow the suite specification; system-wide RAM and NPU
utilisation are appended at the end so the required prefix stays intact.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - environment guard
    sys.exit("hw_monitor: the 'psutil' package is required (pip install psutil).")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CSV_COLUMNS = [
    "timestamp", "epoch_ms",
    "cpu_total_pct", "cpu_prives_pct",
    "cpu_core0", "cpu_core1", "cpu_core2", "cpu_core3",
    "ram_rss_mb", "ram_vms_mb",
    "temp_soc_c", "power_w", "voltage_v", "current_a",
    "throttled_hex", "npu_temp_c",
    # Extensions
    "ram_sys_used_mb", "ram_sys_pct", "npu_util_pct",
]

NUM_CORE_COLUMNS = 4

THROTTLE_BITS = {
    0: "under-voltage",
    1: "arm-frequency-capped",
    2: "throttled",
    3: "soft-temperature-limit",
    16: "under-voltage-occurred",
    17: "arm-frequency-capping-occurred",
    18: "throttling-occurred",
    19: "soft-temperature-limit-occurred",
}

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def log(message: str) -> None:
    print(f"[hw_monitor] {message}", file=sys.stderr, flush=True)


def run_command(args: list[str], timeout: float = 2.0) -> Optional[str]:
    """Runs a command and returns stdout, or None on any failure."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def decode_throttled(value: int) -> list[str]:
    return [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]


# -----------------------------------------------------------------------------
# PMIC power parsing
# -----------------------------------------------------------------------------

_PMIC_LINE = re.compile(
    r"^\s*(?P<label>[A-Za-z0-9_]+)\s+"
    r"(?:[A-Za-z]+\s*\(\s*\d+\s*\))?\s*[=:]\s*"
    r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(?P<prefix>[mu]?)(?P<unit>[AVW])?\b"
)

_PREFIX_SCALE = {"": 1.0, "m": 1e-3, "u": 1e-6}


def parse_pmic_adc(text: str) -> dict[str, float]:
    """Parses ``vcgencmd pmic_read_adc`` output into {label: SI value}.

    Accepts lines such as ``EXT5V_V volt(24)=5.13600000V`` or
    ``VDD_CORE_A current(7)=2.1A``; unknown or malformed lines are ignored.
    """
    readings: dict[str, float] = {}
    for line in text.splitlines():
        match = _PMIC_LINE.match(line)
        if not match:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        readings[match.group("label").upper()] = value * _PREFIX_SCALE[match.group("prefix")]
    return readings


@dataclass
class PowerSample:
    power_w: Optional[float] = None
    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    source: str = "unavailable"


def compute_power(readings: dict[str, float]) -> PowerSample:
    """Derives input power from PMIC readings.

    Preferred: P = EXT5V_V x EXT5V_A. When the input current channel is not
    exposed (stock RPi 5 firmware), falls back to the sum of V x I over every
    PMIC rail that reports both quantities. The fallback excludes PMIC losses
    and loads fed directly from the 5 V header (USB, HATs).
    """
    voltage = readings.get("EXT5V_V")
    current = readings.get("EXT5V_A", readings.get("EXT5V_I"))
    if voltage is not None and current is not None:
        return PowerSample(voltage * current, voltage, current, "ext5v")

    rail_power = 0.0
    rails = 0
    for label, amps in readings.items():
        if not label.endswith("_A"):
            continue
        volts = readings.get(label[:-2] + "_V")
        if volts is not None:
            rail_power += volts * amps
            rails += 1
    if rails:
        est_current = rail_power / voltage if voltage else None
        return PowerSample(rail_power, voltage, est_current, f"pmic_rails({rails})")

    return PowerSample(None, voltage, None)


# -----------------------------------------------------------------------------
# Raspberry Pi firmware interface
# -----------------------------------------------------------------------------

class VcgencmdReader:
    THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")

    def __init__(self) -> None:
        self.available = shutil.which("vcgencmd") is not None
        if not self.available:
            log("vcgencmd not found: temperature falls back to sysfs; power/throttling disabled.")
        self._power_source: Optional[str] = None
        self._last_throttled: Optional[int] = None

    def soc_temperature(self) -> Optional[float]:
        if self.available:
            out = run_command(["vcgencmd", "measure_temp"])
            if out:
                match = re.search(r"temp=\s*([-+]?\d+(?:\.\d+)?)", out)
                if match:
                    return float(match.group(1))
        try:
            return int(self.THERMAL_ZONE.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            return None

    def throttled(self) -> Optional[int]:
        if not self.available:
            return None
        out = run_command(["vcgencmd", "get_throttled"])
        if not out:
            return None
        match = re.search(r"throttled=\s*(0x[0-9a-fA-F]+|\d+)", out)
        if not match:
            return None
        value = int(match.group(1), 0)
        if value != self._last_throttled:
            flags = ", ".join(decode_throttled(value)) or "none"
            log(f"throttled state changed: {hex(value)} ({flags})")
            self._last_throttled = value
        return value

    def power(self) -> PowerSample:
        if not self.available:
            return PowerSample()
        out = run_command(["vcgencmd", "pmic_read_adc"])
        if not out:
            return PowerSample()
        sample = compute_power(parse_pmic_adc(out))
        if sample.source != self._power_source:
            log(f"power source: {sample.source}")
            self._power_source = sample.source
        return sample


# -----------------------------------------------------------------------------
# Hailo NPU
# -----------------------------------------------------------------------------

class HailoMonitor:
    """Best-effort NPU telemetry.

    Utilisation is parsed from a long-running ``hailortcli monitor`` process
    (the inference process must run with HAILO_MONITOR=1). Temperature is read
    through the ``hailo_platform`` Python API, falling back to the CLI.
    """

    TEMP_RETRY_BACKOFF_S = 30.0
    UTIL_MAX_RESTARTS = 5

    def __init__(self, temp_period_s: float) -> None:
        self._cli = shutil.which("hailortcli")
        self._temp_period_s = temp_period_s
        self._next_temp_poll = 0.0
        self._temp: Optional[float] = None
        self._util: Optional[float] = None
        self._util_stamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._device = None
        self._api_failed = False
        self._monitor_proc: Optional[subprocess.Popen] = None
        self._util_thread: Optional[threading.Thread] = None

        if self._cli is None:
            log("hailortcli not found: NPU utilisation disabled.")
        else:
            self._util_thread = threading.Thread(target=self._util_loop, daemon=True)
            self._util_thread.start()

    # Utilisation ------------------------------------------------------------
    def _util_loop(self) -> None:
        device_row = re.compile(r"^\s*(\S+)\s+(\d+(?:\.\d+)?)\s+HAILO\w*", re.IGNORECASE)
        restarts = 0
        while not self._stop.is_set() and restarts <= self.UTIL_MAX_RESTARTS:
            try:
                self._monitor_proc = subprocess.Popen(
                    [self._cli, "monitor"], stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, bufsize=1)
            except OSError as exc:
                log(f"cannot start 'hailortcli monitor': {exc}")
                return
            assert self._monitor_proc.stdout is not None
            for raw in self._monitor_proc.stdout:
                if self._stop.is_set():
                    break
                match = device_row.match(ANSI_ESCAPE.sub("", raw))
                if match:
                    with self._lock:
                        self._util = float(match.group(2))
                        self._util_stamp = time.monotonic()
            self._terminate_monitor()
            restarts += 1
            self._stop.wait(2.0)
        if restarts > self.UTIL_MAX_RESTARTS:
            log("'hailortcli monitor' keeps exiting: NPU utilisation disabled.")

    def _terminate_monitor(self) -> None:
        proc = self._monitor_proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    def utilisation(self) -> Optional[float]:
        with self._lock:
            # Discard stale values (monitor stalled or inference stopped).
            if self._util is not None and time.monotonic() - self._util_stamp < 5.0:
                return self._util
        return None

    # Temperature ------------------------------------------------------------
    def _read_temp_api(self) -> Optional[float]:
        if self._api_failed:
            return None
        try:
            if self._device is None:
                from hailo_platform import Device  # type: ignore
                device_ids = Device.scan()
                if not device_ids:
                    return None
                self._device = Device(device_ids[0])
            target = getattr(self._device, "control", self._device)
            info = target.get_chip_temperature()
            return max(float(info.ts0_temperature), float(info.ts1_temperature))
        except ImportError:
            self._api_failed = True
            log("hailo_platform not importable: using hailortcli for NPU temperature.")
        except Exception as exc:  # noqa: BLE001 - HailoRT raises many exception types
            log(f"Hailo API temperature read failed: {exc}")
            self._release_device()
        return None

    def _read_temp_cli(self) -> Optional[float]:
        if self._cli is None:
            return None
        out = run_command([self._cli, "fw-control", "--temperature"], timeout=3.0)
        if not out:
            return None
        values = [float(v) for v in re.findall(r"(-?\d+(?:\.\d+)?)\s*(?:°|')?\s*C\b", out)]
        return max(values) if values else None

    def temperature(self) -> Optional[float]:
        now = time.monotonic()
        if now < self._next_temp_poll:
            return self._temp
        temp = self._read_temp_api()
        if temp is None:
            temp = self._read_temp_cli()
        self._temp = temp
        self._next_temp_poll = now + (self._temp_period_s if temp is not None
                                      else self.TEMP_RETRY_BACKOFF_S)
        return temp

    def _release_device(self) -> None:
        device, self._device = self._device, None
        if device is not None and hasattr(device, "release"):
            try:
                device.release()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        self._stop.set()
        self._terminate_monitor()
        if self._util_thread is not None:
            self._util_thread.join(timeout=3.0)
        self._release_device()


# -----------------------------------------------------------------------------
# Process tracking
# -----------------------------------------------------------------------------

@dataclass
class ProcessSample:
    cpu_pct: Optional[float] = None
    rss_mb: Optional[float] = None
    vms_mb: Optional[float] = None


class ProcessTracker:
    """Follows the PRIVES process by PID or by name, re-resolving if it restarts."""

    def __init__(self, pid: Optional[int], name: Optional[str]) -> None:
        self._fixed_pid = pid
        self._name = name
        self._proc: Optional[psutil.Process] = None
        self._ncpu = psutil.cpu_count(logical=True) or 1

    def _resolve(self) -> Optional[psutil.Process]:
        if self._fixed_pid is not None:
            try:
                return psutil.Process(self._fixed_pid)
            except psutil.Error:
                return None
        if not self._name:
            return None
        own_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if proc.info["pid"] == own_pid:
                continue
            cmdline = proc.info.get("cmdline") or []
            exe_name = os.path.basename(cmdline[0]) if cmdline else ""
            if proc.info.get("name") == self._name or exe_name == self._name:
                return proc
        return None

    @staticmethod
    def _read_status(pid: int) -> tuple[Optional[float], Optional[float]]:
        rss = vms = None
        try:
            with open(f"/proc/{pid}/status", encoding="ascii", errors="ignore") as status:
                for line in status:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) / 1024.0
                    elif line.startswith("VmSize:"):
                        vms = int(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        return rss, vms

    def sample(self) -> ProcessSample:
        if self._proc is None or not self._proc.is_running():
            self._proc = self._resolve()
            if self._proc is None:
                return ProcessSample()
            try:
                self._proc.cpu_percent(None)  # Prime the CPU counter.
            except psutil.Error:
                self._proc = None
                return ProcessSample()
            rss, vms = self._read_status(self._proc.pid)
            return ProcessSample(None, rss, vms)
        try:
            # Normalised to the whole system (0-100 %) to be comparable with cpu_total_pct.
            cpu = self._proc.cpu_percent(None) / self._ncpu
        except psutil.Error:
            self._proc = None
            return ProcessSample()
        rss, vms = self._read_status(self._proc.pid)
        return ProcessSample(cpu, rss, vms)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def fmt(value: Optional[float], digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def resolve_output(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else RESULTS_DIR / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRIVES hardware telemetry daemon.")
    parser.add_argument("--output", default="hardware_metrics.csv",
                        help="CSV path (relative paths are resolved under benchmarks/results/).")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling period in seconds.")
    parser.add_argument("--pid", type=int, default=None, help="PID of the PRIVES process.")
    parser.add_argument("--process-name", default="prives",
                        help="Process name to track when --pid is not given.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional auto-stop after N seconds.")
    parser.add_argument("--npu-temp-period", type=float, default=2.0,
                        help="NPU temperature polling period in seconds.")
    parser.add_argument("--no-npu", action="store_true", help="Disable Hailo NPU telemetry.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        log("--interval must be positive.")
        return 2

    stop = threading.Event()

    def on_signal(signum: int, _frame) -> None:
        log(f"received {signal.Signals(signum).name}, shutting down.")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    output = resolve_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    vcgencmd = VcgencmdReader()
    hailo = None if args.no_npu else HailoMonitor(args.npu_temp_period)
    tracker = ProcessTracker(args.pid, args.process_name)

    psutil.cpu_percent(None)
    psutil.cpu_percent(None, percpu=True)

    rows = 0
    start = time.monotonic()
    next_tick = start + args.interval
    log(f"sampling every {args.interval * 1000:.0f} ms -> {output}")

    try:
        with open(output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            handle.flush()

            while not stop.wait(max(0.0, next_tick - time.monotonic())):
                now_wall = dt.datetime.now(dt.timezone.utc)
                cpu_total = psutil.cpu_percent(None)
                per_core = psutil.cpu_percent(None, percpu=True)
                cores = [per_core[i] if i < len(per_core) else None
                         for i in range(NUM_CORE_COLUMNS)]
                proc = tracker.sample()
                vmem = psutil.virtual_memory()
                temp = vcgencmd.soc_temperature()
                throttled = vcgencmd.throttled()
                power = vcgencmd.power()
                npu_temp = hailo.temperature() if hailo else None
                npu_util = hailo.utilisation() if hailo else None

                writer.writerow([
                    now_wall.isoformat(timespec="milliseconds"),
                    int(now_wall.timestamp() * 1000),
                    fmt(cpu_total, 1), fmt(proc.cpu_pct, 1),
                    *(fmt(c, 1) for c in cores),
                    fmt(proc.rss_mb, 2), fmt(proc.vms_mb, 2),
                    fmt(temp, 2), fmt(power.power_w, 3),
                    fmt(power.voltage_v, 3), fmt(power.current_a, 3),
                    "" if throttled is None else hex(throttled),
                    fmt(npu_temp, 2),
                    fmt((vmem.total - vmem.available) / 2**20, 1), fmt(vmem.percent, 1),
                    fmt(npu_util, 1),
                ])
                handle.flush()
                rows += 1

                if args.duration is not None and time.monotonic() - start >= args.duration:
                    break

                next_tick += args.interval
                if next_tick < time.monotonic():
                    # Sampling overran: realign instead of bursting.
                    next_tick = time.monotonic() + args.interval
            os.fsync(handle.fileno())
    finally:
        if hailo:
            hailo.close()
        log(f"{rows} samples written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
