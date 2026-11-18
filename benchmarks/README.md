# PRIVES Hardware Benchmarking & Stress Testing Suite

Automated benchmarking, hardware telemetry, memory leak verification, and RNF compliance analysis suite for the **PRIVES** pipeline on **Raspberry Pi 5** (8 GB RAM, ARM Cortex-A76 @ 2.4 GHz) with **Raspberry Pi AI HAT+** (Hailo-8 NPU), running Raspberry Pi OS Lite (Debian 12 Bookworm, 64-bit ARM64) headless.

---

## 1. Directory Structure

```text
benchmarks/
├── include/
│   └── prives_profiler.hpp      # Header-only C++ per-frame latency instrumentation (MPSC ring buffer)
├── scripts/
│   ├── hw_monitor.py            # Hardware daemon: 500 ms sampling (CPU, RAM, Temp, PMIC Power, Hailo NPU)
│   ├── run_stress_suite.sh      # Master unattended orchestrator for validation scenarios (E0-E3, L0)
│   └── check_leaks.sh           # Heap memory profiling via Heaptrack and Valgrind Massif
├── analysis/
│   └── generate_report.py       # Offline processing: CSV parser, RNF LaTeX tables & publication-quality PDF plots
├── results/                     # Output directory for telemetry data, logs, and reports (git-ignored)
└── README.md                    # Compilation, execution, and scenario reference
```

---

## 2. Non-Functional Requirements (NFR / RNF) Validated

| Requirement | Metric | Theoretical Threshold | Target Stage / Evaluation Metric |
| :--- | :--- | :--- | :--- |
| **RNF-01** | Frame Rate | $\ge 30.0$ FPS sustained | Sustained mean $\ge 29.5$ FPS with 0 accumulated frame drops (`dropped_flag == 0`) |
| **RNF-02** | End-to-End Latency | $\le 25.0$ ms / frame | P95 end-to-end latency ($T_{capture}$ hand-off to vital signs update) $\le 25.0$ ms |
| **RNF-03** | CPU Utilization | $< 35.0\%$ system-wide | 4-core average $< 35\%$, zero threads saturated at $100\%$ ($5\,\text{s}$ rolling max $< 99\%$) |
| **RNF-04** | RAM Footprint & Leaks | $\le 350.0$ MB Resident (RSS) | Peak RSS $\le 350.0$ MB, post-warmup linear drift $|\text{slope}| \le 2.0$ MB/h, 0 leaks |
| **RNF-13** | Thermal & Power | $P < 12.0$ W cont., $T < 70^\circ\text{C}$ | PMIC DA9091 $10\,\text{s}$ rolling avg $< 12$ W, SoC temp $< 70^\circ\text{C}$, 0 throttling events |

---

## 3. System Prerequisites & Dependencies

### Operating System Packages (APT)
```bash
sudo apt update
sudo apt install -y \
    build-essential cmake \
    libopencv-dev \
    libgstreamer1.0-dev libgstreamer-app1.0-dev \
    libfftw3-dev libiir-dev \
    stress-ng heaptrack valgrind \
    libraspberrypi-bin
```

### Python 3 Environment
In Raspberry Pi OS (Debian 12 Bookworm), you can install the pre-compiled system packages directly via APT (recommended, fastest and avoids build/PEP-668 conflicts):
```bash
sudo apt install -y python3-psutil python3-pandas python3-matplotlib python3-seaborn
```

Or install via `pip`:
```bash
pip install --break-system-packages psutil pandas matplotlib seaborn pandas-stubs
```

*Note: For Hailo-8 NPU telemetry, ensure `hailortcli` is installed and in your `PATH`. The Python package `hailo_platform` is optional; if not present, `hw_monitor.py` automatically falls back to `hailortcli fw-control --temperature`.*

---

## 4. Building with Latency Profiling

The C++ instrumentation (`benchmarks/include/prives_profiler.hpp`) uses a zero-allocation, lock-free MPSC ring buffer and a decoupled background writer thread. By default, profiling is compiled out (`no-op`).

To enable profiling in the `prives` binary:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPRIVES_ENABLE_PROFILING=ON ..
cmake --build . -j$(nproc)
```

The resulting `build/prives` binary will have per-frame timing hooks enabled and will stream metrics to `benchmarks/results/<run_id>/pipeline_telemetry.csv` (or the path defined by `PRIVES_TELEMETRY_CSV`).

---

## 5. Running the Automated Stress Suite

The master script `benchmarks/scripts/run_stress_suite.sh` coordinates cool-downs, process management, scenario execution, hardware monitoring, and automatic report generation.

### Full Execution (Nominal Duration: ~100 min)
```bash
bash benchmarks/scripts/run_stress_suite.sh
```

### Smoke Test / Rapid Validation (`--scale` factor)
You can scale the duration of all scenarios linearly using `--scale`:
```bash
# Runs ~5% of nominal time (~5 min total) to verify end-to-end functionality:
bash benchmarks/scripts/run_stress_suite.sh --scale 0.05
```

### Command-Line Options
| Option | Default | Description |
| :--- | :--- | :--- |
| `--bin PATH` | `build/prives` | Path to compiled PRIVES executable |
| `--workdir DIR` | Binary dir | Working directory containing `.hef` model and configs |
| `--scenarios LIST`| `E0,E1,E2,E3` | Comma-separated list of scenarios to execute |
| `--scale FACTOR` | `1.0` | Multiplier for scenario durations (e.g. `0.1` for 10% duration) |
| `--interval SEC` | `0.5` | Hardware telemetry sampling rate (seconds) |
| `--occlusion SEC`| `5.0` | Face on/off half-period for dynamic scene scenario (E2) |
| `--cooldown-temp`| `50.0` | Target SoC temperature (°C) to wait for before starting a scenario |
| `--leak-tool TOOL`| `auto` | Heap profiler: `heaptrack` (preferred) or `massif` |
| `--skip-leaks` | Disabled | Skip the 5-minute initial heap allocation profiling run |
| `--no-report` | Disabled | Do not run `generate_report.py` automatically at the end |
| `--run-id NAME` | `run_<timestamp>` | Custom folder name in `benchmarks/results/` |

---

## 6. Experimental Validation Scenarios

1. **L0: Heap Allocation Profiling (5 min)**
   - Executed via `check_leaks.sh` before stress tests.
   - Profiled under `heaptrack` (or `valgrind --tool=massif`).
   - Produces heap allocation snapshots, peak consumption, and identifies memory leaks.

2. **E0: Idle Baseline (10 min)**
   - Headless system at idle (camera pipeline stopped).
   - Establishes baseline operating metrics for OS CPU, quiescent RAM, idle power, and ambient SoC temperature.

3. **E1: Nominal Soak Test (60 min)**
   - Continuous 30 FPS pipeline execution.
   - Evaluates asymptotic RSS stability (linear regression slope $|\text{slope}| \le 2$ MB/h post-warmup) and thermal steady-state equilibrium.

4. **E2: Dynamic Scene Stress (15 min)**
   - Injects periodic synthetic face occlusions via `PRIVES_BENCH_OCCLUSION_PERIOD_S`.
   - Forces repeated Face Mesh re-initialization and signal buffer resets (`anonymizer.reset()`, `pos_extractor.reset()`, `estimator.reset()`).
   - Verifies lack of buffer accumulation or state leaks during tracking recovery.

5. **E3: Memory Bus Contention (10 min)**
   - Executes `prives` alongside synthetic LPDDR4X bus contention:
     ```bash
     stress-ng --vm 1 --vm-bytes 1G --vm-keep --timeout 10m
     ```
   - Measures inter-frame jitter ($\Delta t = t_i - t_{i-1}$) and checks if DMA-BUF/memory bus saturation degrades throughput below 30 FPS.

---

## 7. Individual Script Usage

### Hardware Telemetry Daemon (`hw_monitor.py`)
Can be run independently to monitor system metrics or a specific process:
```bash
python3 benchmarks/scripts/hw_monitor.py \
    --output benchmarks/results/custom_metrics.csv \
    --interval 0.5 \
    --process-name prives
```
- Captures system and per-core CPU%, target process RSS/VMS via `/proc/[PID]/status`, SoC temp via `vcgencmd measure_temp`, throttling bitmasks via `vcgencmd get_throttled`, voltage and current via `vcgencmd pmic_read_adc` (DA9091 PMIC), and Hailo NPU stats.
- Safely handles `SIGINT`/`SIGTERM` to flush and close CSV files.

### Standalone Leak Checker (`check_leaks.sh`)
```bash
bash benchmarks/scripts/check_leaks.sh --duration 300 --tool heaptrack
```
Outputs `heaptrack_summary.txt` and `leak_report.txt` in `benchmarks/results/<output-dir>/`.

### Offline Analysis & Report Generation (`generate_report.py`)
```bash
# Analyze latest run:
python3 benchmarks/analysis/generate_report.py

# Or specify a specific run directory:
python3 benchmarks/analysis/generate_report.py --run-dir benchmarks/results/run_20261009_120000
```

#### Generated Deliverables in `<run-dir>/report/`:
- `rnf_summary.txt`: Plaintext tabular validation of all metrics against theoretical thresholds.
- `table_rnf.tex`: LaTeX booktabs table for academic papers comparing Mean, $\sigma$, P95, Worst-case, and Pass/Fail status.
- `latency_breakdown.pdf`: Vector stacked area chart of $T_{npu}$, $T_{anon}$, $T_{pos}$, $T_{filter}$, $T_{fft}$ vs. 25.0 ms and 33.33 ms reference limits.
- `memory_stability.pdf`: RSS time-series over 60 min with warmup exclusion and least-squares regression fit.
- `thermal_power_profile.pdf`: Dual Y-axis plot showing SoC/NPU temperature (°C) against 70°C limit and PMIC power (W) against 12.0 W limit.
- `interframe_jitter_hist.pdf`: High-resolution histogram + KDE of inter-frame delta ($\Delta t = t_i - t_{i-1}$) around the 33.33 ms (30 Hz) target.
