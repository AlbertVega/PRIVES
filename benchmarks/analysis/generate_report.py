#!/usr/bin/env python3
"""Offline analysis of the PRIVES benchmarking suite.

Reads the CSV files produced by the stress suite and generates:
  - rnf_summary.txt                RNF compliance per scenario (plain text)
  - table_rnf.tex                  LaTeX (booktabs) table for the primary scenario
  - table_rnf_<scenario>.tex       LaTeX table for every scenario
  - latency_breakdown.pdf          Stacked per-stage latency
  - memory_stability.pdf           RSS time series + linear regression (soak test)
  - thermal_power_profile.pdf      SoC temperature and power (dual Y axis)
  - interframe_jitter_hist.pdf     Inter-frame interval histogram + KDE

Expected layout (created by run_stress_suite.sh):
  benchmarks/results/<run>/E0_idle/hardware_metrics.csv
  benchmarks/results/<run>/E1_soak/{pipeline_telemetry,hardware_metrics}.csv
  ...
A flat directory containing both CSV files directly is also accepted.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# -----------------------------------------------------------------------------
# Requirements and validation criteria
# -----------------------------------------------------------------------------
TARGET_FPS = 30.0
FRAME_PERIOD_MS = 1000.0 / TARGET_FPS
FPS_TOLERANCE = 0.5            # "approximately >= 30 FPS"
LATENCY_MAX_MS = 25.0          # RNF-02, evaluated on P95
CPU_MAX_PCT = 35.0             # RNF-03, evaluated on the mean
CORE_SATURATION_PCT = 99.0     # RNF-03, 5 s rolling mean of any core
RSS_MAX_MB = 350.0             # RNF-04, evaluated on the worst case
RSS_SLOPE_MAX_MB_H = 2.0       # RNF-04, post-warm-up drift
POWER_MAX_W = 12.0             # RNF-13, 10 s rolling mean ("continuous")
TEMP_MAX_C = 70.0              # RNF-13, worst case
THROTTLE_ACTIVE_MASK = 0xF     # Under-voltage, freq capped, throttled, soft temp limit

RSS_WARMUP_S = 300.0
CORE_WINDOW_S = 5.0
POWER_WINDOW_S = 10.0
SENSOR_GAP_FACTOR = 1.5

STATUS_PASS = "CUMPLE"
STATUS_FAIL = "FALLA"
STATUS_NA = "N/D"

SCENARIO_LABELS = {
    "E0": "E0 Idle",
    "E1": "E1 Soak nominal",
    "E2": "E2 Escena dinámica",
    "E3": "E3 Contención de memoria",
    "RUN": "Ejecución",
}

STAGE_COLUMNS = {
    "t_npu_ms": r"$T_{npu}$ (SCRFD/Hailo)",
    "t_anon_ms": r"$T_{anon}$ (permutación)",
    "t_pos_ms": r"$T_{pos}$ (POS + detrend)",
    "t_filter_ms": r"$T_{filter}$ (IIR)",
    "t_fft_ms": r"$T_{fft}$ (FFTW3)",
}

PIPELINE_NUMERIC = ["frame_id", "timestamp_epoch_ms", "t_capture_ms", *STAGE_COLUMNS,
                    "t_total_ms", "dropped_flag"]


def warn(message: str) -> None:
    print(f"[generate_report] WARNING: {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"[generate_report] {message}")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

@dataclass
class Scenario:
    key: str
    name: str
    pipeline: Optional[pd.DataFrame]
    hardware: Optional[pd.DataFrame]

    @property
    def label(self) -> str:
        return SCENARIO_LABELS.get(self.key, self.name)


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        warn(f"cannot parse {path}: {exc}")
        return None
    return frame if not frame.empty else None


def load_pipeline(path: Path) -> Optional[pd.DataFrame]:
    frame = _read_csv(path)
    if frame is None:
        return None
    for column in PIPELINE_NUMERIC:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["frame_id", "timestamp_epoch_ms"]).sort_values("frame_id")
    frame["dropped_flag"] = frame["dropped_flag"].fillna(0).astype(int)
    frame["t_s"] = (frame["timestamp_epoch_ms"] - frame["timestamp_epoch_ms"].iloc[0]) / 1000.0
    return frame.reset_index(drop=True)


def _parse_hex(value) -> float:
    try:
        return float(int(str(value), 0))
    except (TypeError, ValueError):
        return np.nan


def load_hardware(path: Path) -> Optional[pd.DataFrame]:
    frame = _read_csv(path)
    if frame is None:
        return None
    for column in frame.columns:
        if column not in ("timestamp", "throttled_hex"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["epoch_ms"]).sort_values("epoch_ms")
    frame["throttled"] = frame.get("throttled_hex", pd.Series(dtype=object)).map(_parse_hex)
    frame["t_s"] = (frame["epoch_ms"] - frame["epoch_ms"].iloc[0]) / 1000.0
    return frame.reset_index(drop=True)


def discover_scenarios(run_dir: Path) -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}
    for sub in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        match = re.match(r"^(E\d+)", sub.name)
        if not match:
            continue
        scenario = Scenario(match.group(1), sub.name,
                            load_pipeline(sub / "pipeline_telemetry.csv"),
                            load_hardware(sub / "hardware_metrics.csv"))
        if scenario.pipeline is None and scenario.hardware is None:
            warn(f"no data in {sub}")
            continue
        scenarios[scenario.key] = scenario

    if not scenarios:
        flat = Scenario("RUN", run_dir.name,
                        load_pipeline(run_dir / "pipeline_telemetry.csv"),
                        load_hardware(run_dir / "hardware_metrics.csv"))
        if flat.pipeline is not None or flat.hardware is not None:
            scenarios["RUN"] = flat
    return scenarios


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

@dataclass
class Stats:
    mean: float
    std: float
    p95: float
    worst: float


def describe(series: Optional[pd.Series], higher_is_better: bool = False) -> Optional[Stats]:
    """Mean, std, tail percentile and worst case. For higher-is-better metrics
    the tail is the 5th percentile and the worst case is the minimum."""
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return None
    tail = np.percentile(values, 5 if higher_is_better else 95)
    worst = values.min() if higher_is_better else values.max()
    std = values.std(ddof=1) if values.size > 1 else 0.0
    return Stats(float(values.mean()), float(std), float(tail), float(worst))


def per_second_fps(pipeline: pd.DataFrame) -> pd.Series:
    """Completed frames per 1 s bin, discarding the partial first/last bins."""
    done = pipeline.loc[pipeline["dropped_flag"] == 0, "t_s"]
    if done.empty:
        return pd.Series(dtype=float)
    counts = np.floor(done).astype(int).value_counts().sort_index()
    full_range = pd.RangeIndex(counts.index.min(), counts.index.max() + 1)
    counts = counts.reindex(full_range, fill_value=0)
    return counts.iloc[1:-1].astype(float) if len(counts) > 2 else counts.astype(float)


def interframe_intervals(pipeline: pd.DataFrame) -> pd.Series:
    """Delta between consecutive captured frames (ms), including dropped ones."""
    return pipeline["timestamp_epoch_ms"].diff().dropna()


def sensor_gaps(intervals: pd.Series) -> int:
    """Estimated frames lost before reaching the application (camera/appsink)."""
    late = intervals[intervals > SENSOR_GAP_FACTOR * FRAME_PERIOD_MS]
    return int(np.maximum(np.round(late / FRAME_PERIOD_MS) - 1, 0).sum())


def rolling_window(frame: pd.DataFrame, column: str, window_s: float) -> pd.Series:
    period = frame["t_s"].diff().median()
    samples = max(1, int(round(window_s / period))) if period and period > 0 else 1
    return frame[column].rolling(samples, min_periods=samples).mean()


def linear_slope(t_s: pd.Series, values: pd.Series) -> Optional[tuple[float, float]]:
    """Least-squares fit; returns (slope MB/h, intercept MB)."""
    mask = values.notna() & t_s.notna()
    if mask.sum() < 3:
        return None
    slope, intercept = np.polyfit(t_s[mask] / 3600.0, values[mask], 1)
    return float(slope), float(intercept)


def warmup_cut(hardware: pd.DataFrame) -> float:
    duration = hardware["t_s"].max()
    return RSS_WARMUP_S if duration > 2 * RSS_WARMUP_S else 0.2 * duration


# -----------------------------------------------------------------------------
# RNF evaluation
# -----------------------------------------------------------------------------

@dataclass
class RnfRow:
    metric: str
    metric_tex: str
    rnf: str
    threshold: str
    threshold_tex: str
    stats: Optional[Stats]
    status: str
    note: str = ""


def _status(condition: Optional[bool]) -> str:
    if condition is None:
        return STATUS_NA
    return STATUS_PASS if condition else STATUS_FAIL


def evaluate(scenario: Scenario) -> tuple[list[RnfRow], dict[str, float]]:
    pipe, hw = scenario.pipeline, scenario.hardware
    rows: list[RnfRow] = []
    extra: dict[str, float] = {}

    # RNF-01 Frame rate
    fps_stats = dropped = gaps = None
    if pipe is not None:
        fps_stats = describe(per_second_fps(pipe), higher_is_better=True)
        dropped = int(pipe["dropped_flag"].sum())
        intervals = interframe_intervals(pipe)
        gaps = sensor_gaps(intervals)
        extra.update(frames=float(len(pipe)), dropped=float(dropped), sensor_gaps=float(gaps),
                     jitter_std_ms=float(intervals.std()) if len(intervals) > 1 else np.nan)
    rows.append(RnfRow(
        "FPS", "Tasa de cuadros (FPS)", "RNF-01",
        f">= {TARGET_FPS:.1f}", rf"$\geq {TARGET_FPS:.1f}$", fps_stats,
        _status(None if fps_stats is None else
                fps_stats.mean >= TARGET_FPS - FPS_TOLERANCE and dropped == 0),
        "" if fps_stats is None else f"descartados={dropped}, huecos de sensor={gaps}; P95=P5"))

    # RNF-02 Latency
    lat_stats = None
    if pipe is not None and "t_total_ms" in pipe:
        lat_stats = describe(pipe.loc[pipe["dropped_flag"] == 0, "t_total_ms"])
    rows.append(RnfRow(
        "Latencia (ms)", "Latencia por cuadro (ms)", "RNF-02",
        f"<= {LATENCY_MAX_MS:.1f}", rf"$\leq {LATENCY_MAX_MS:.1f}$", lat_stats,
        _status(None if lat_stats is None else lat_stats.p95 <= LATENCY_MAX_MS),
        "criterio: P95"))

    # RNF-03 CPU
    cpu_stats = None
    max_core = np.nan
    if hw is not None and hw["cpu_total_pct"].notna().any():
        cpu_stats = describe(hw["cpu_total_pct"])
        core_cols = [c for c in hw.columns if c.startswith("cpu_core") and hw[c].notna().any()]
        if core_cols:
            max_core = max(float(rolling_window(hw, c, CORE_WINDOW_S).max()) for c in core_cols)
        if "cpu_prives_pct" in hw and hw["cpu_prives_pct"].notna().any():
            extra["cpu_prives_mean"] = float(hw["cpu_prives_pct"].mean())
        extra["max_core_rolling_pct"] = max_core
    rows.append(RnfRow(
        "CPU (%)", r"Uso global de CPU (\%)", "RNF-03",
        f"< {CPU_MAX_PCT:.1f}", rf"$< {CPU_MAX_PCT:.1f}$", cpu_stats,
        _status(None if cpu_stats is None else
                cpu_stats.mean < CPU_MAX_PCT and not (max_core >= CORE_SATURATION_PCT)),
        "" if cpu_stats is None else f"núcleo máx. (media {CORE_WINDOW_S:.0f} s)={max_core:.1f}%"))

    # RNF-04 RAM
    rss_stats = None
    slope = None
    if hw is not None and "ram_rss_mb" in hw and hw["ram_rss_mb"].notna().any():
        rss_stats = describe(hw["ram_rss_mb"])
        steady = hw[hw["t_s"] >= warmup_cut(hw)]
        fit = linear_slope(steady["t_s"], steady["ram_rss_mb"])
        slope = fit[0] if fit else None
        if slope is not None:
            extra["rss_slope_mb_h"] = slope
    rows.append(RnfRow(
        "RAM RSS (MB)", "Memoria residente RSS (MB)", "RNF-04",
        f"<= {RSS_MAX_MB:.1f}", rf"$\leq {RSS_MAX_MB:.1f}$", rss_stats,
        _status(None if rss_stats is None else
                rss_stats.worst <= RSS_MAX_MB and (slope is None or abs(slope) <= RSS_SLOPE_MAX_MB_H)),
        "" if slope is None else f"pendiente post-calentamiento={slope:+.3f} MB/h"))

    # RNF-13 Power
    pwr_stats = None
    if hw is not None and "power_w" in hw and hw["power_w"].notna().any():
        pwr_stats = describe(hw["power_w"])
        extra["power_rolling_max_w"] = float(rolling_window(hw, "power_w", POWER_WINDOW_S).max())
    rows.append(RnfRow(
        "Potencia (W)", "Potencia (W)", "RNF-13",
        f"< {POWER_MAX_W:.1f}", rf"$< {POWER_MAX_W:.1f}$", pwr_stats,
        _status(None if pwr_stats is None else extra["power_rolling_max_w"] < POWER_MAX_W),
        "" if pwr_stats is None else
        f"máx. media {POWER_WINDOW_S:.0f} s={extra['power_rolling_max_w']:.2f} W"))

    # RNF-13 Temperature and throttling
    tmp_stats = None
    throttled = None
    if hw is not None and "temp_soc_c" in hw and hw["temp_soc_c"].notna().any():
        tmp_stats = describe(hw["temp_soc_c"])
        flags = hw["throttled"].dropna().astype(int)
        throttled = bool((flags & THROTTLE_ACTIVE_MASK).any()) if not flags.empty else False
        if "npu_temp_c" in hw and hw["npu_temp_c"].notna().any():
            extra["npu_temp_max_c"] = float(hw["npu_temp_c"].max())
    rows.append(RnfRow(
        "Temperatura (°C)", r"Temperatura SoC ($^\circ$C)", "RNF-13",
        f"< {TEMP_MAX_C:.1f}", rf"$< {TEMP_MAX_C:.1f}$", tmp_stats,
        _status(None if tmp_stats is None else tmp_stats.worst < TEMP_MAX_C and not throttled),
        "" if tmp_stats is None else f"throttling={'SÍ' if throttled else 'no'}"))

    return rows, extra


# -----------------------------------------------------------------------------
# Text and LaTeX output
# -----------------------------------------------------------------------------

def _num(value: Optional[float], digits: int = 2) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def write_summary(path: Path, results: dict[str, tuple[Scenario, list[RnfRow], dict]]) -> None:
    lines = ["PRIVES - Resumen de cumplimiento de requerimientos no funcionales", "=" * 100, ""]
    baseline = results.get("E0")
    header = f"{'Métrica':<18}{'RNF':<8}{'Umbral':<10}{'Media':>9}{'Desv.':>9}" \
             f"{'P95':>9}{'Peor':>9}  {'Estado':<8}Notas"

    for key, (scenario, rows, extra) in results.items():
        lines += [f"[{scenario.label}]  ({scenario.name})", header, "-" * 100]
        for row in rows:
            s = row.stats
            lines.append(
                f"{row.metric:<18}{row.rnf:<8}{row.threshold:<10}"
                f"{_num(s and s.mean):>9}{_num(s and s.std):>9}{_num(s and s.p95):>9}"
                f"{_num(s and s.worst):>9}  {row.status:<8}{row.note}")
        if extra:
            lines.append("  " + ", ".join(f"{k}={_num(v, 3)}" for k, v in extra.items()))
        if baseline and key != "E0":
            deltas = []
            for metric in ("CPU (%)", "Potencia (W)", "Temperatura (°C)"):
                base = next((r.stats for r in baseline[1] if r.metric == metric), None)
                cur = next((r.stats for r in rows if r.metric == metric), None)
                if base and cur:
                    deltas.append(f"{metric}: {cur.mean - base.mean:+.2f}")
            if deltas:
                lines.append("  Incremento sobre E0 (media): " + ", ".join(deltas))
        lines.append("")

    lines += ["Criterios: FPS media >= 29.5 y 0 cuadros descartados; latencia P95 <= 25 ms;",
              f"CPU media < 35 % y ningún núcleo >= {CORE_SATURATION_PCT:.0f} % (media {CORE_WINDOW_S:.0f} s);",
              f"RSS peor caso <= 350 MB y |pendiente| <= {RSS_SLOPE_MAX_MB_H} MB/h;",
              f"potencia media móvil {POWER_WINDOW_S:.0f} s < 12 W; temperatura máx. < 70 °C sin throttling.",
              "Para FPS la columna P95 contiene el percentil 5 y 'Peor' el mínimo."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, scenario: Scenario, rows: list[RnfRow]) -> None:
    body = []
    for row in rows:
        s = row.stats
        status = rf"\textbf{{{row.status}}}"
        body.append(
            f"    {row.metric_tex} & {row.rnf} & {row.threshold_tex} & {_num(s and s.mean)} & "
            f"{_num(s and s.std)} & {_num(s and s.p95)} & {_num(s and s.worst)} & {status} \\\\")
    label = scenario.key.lower()
    content = "\n".join([
        "% Auto-generated by benchmarks/analysis/generate_report.py",
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Validación de requerimientos no funcionales -- {scenario.label}.}}",
        f"  \\label{{tab:rnf_{label}}}",
        "  \\begin{tabular}{llcrrrrc}",
        "    \\toprule",
        "    Métrica & RNF & Umbral & Media & $\\sigma$ & P95$^{\\dagger}$ & Peor caso & Estado \\\\",
        "    \\midrule",
        *body,
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\par\\smallskip",
        "  {\\footnotesize $^{\\dagger}$Para FPS se reporta el percentil 5 y el peor caso es el mínimo.}",
        "\\end{table}",
        "",
    ])
    path.write_text(content, encoding="utf-8")


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    info(f"figure -> {path.name}")


def plot_latency_breakdown(scenario: Scenario, path: Path) -> None:
    pipe = scenario.pipeline
    done = pipe[pipe["dropped_flag"] == 0].copy()
    stages = [c for c in STAGE_COLUMNS if c in done]
    if done.empty or not stages:
        warn("latency_breakdown: no completed frames.")
        return
    done[stages] = done[stages].fillna(0.0)
    done["bin"] = np.floor(done["t_s"]).astype(int)
    binned = done.groupby("bin")[[*stages, "t_total_ms"]].mean()
    minutes = binned.index.to_numpy() / 60.0

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    palette = sns.color_palette("crest", len(stages))
    ax.stackplot(minutes, *(binned[c].to_numpy() for c in stages),
                 labels=[STAGE_COLUMNS[c] for c in stages], colors=palette, alpha=0.9)
    ax.plot(minutes, binned["t_total_ms"].to_numpy(), color="0.25", lw=0.8,
            label=r"$T_{total}$ extremo a extremo")
    ax.axhline(LATENCY_MAX_MS, color="#d1495b", ls="--", lw=1.2, label="Límite RNF-02 (25 ms)")
    ax.axhline(FRAME_PERIOD_MS, color="#edae49", ls="-.", lw=1.2, label="Presupuesto 30 FPS (33.33 ms)")
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Latencia media por segundo (ms)")
    ax.set_title(f"Descomposición de latencia por etapa — {scenario.label}")
    ax.set_xlim(minutes.min(), minutes.max())
    ax.set_ylim(0, max(FRAME_PERIOD_MS * 1.2, float(binned["t_total_ms"].max()) * 1.1))
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=True)
    _save(fig, path)


def plot_memory_stability(scenario: Scenario, path: Path) -> None:
    hw = scenario.hardware
    data = hw[["t_s", "ram_rss_mb"]].dropna()
    if data.empty:
        warn("memory_stability: no RSS samples.")
        return
    minutes = data["t_s"] / 60.0
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(minutes, data["ram_rss_mb"], color="#30638e", lw=0.8, label="RSS")

    full = linear_slope(data["t_s"], data["ram_rss_mb"])
    cut = warmup_cut(hw)
    steady = data[data["t_s"] >= cut]
    post = linear_slope(steady["t_s"], steady["ram_rss_mb"])
    if full:
        ax.plot(minutes, full[1] + full[0] * data["t_s"] / 3600.0, color="#edae49", ls="--",
                lw=1.2, label=f"Regresión global: {full[0]:+.3f} MB/h")
    if post:
        t_post = steady["t_s"]
        ax.plot(t_post / 60.0, post[1] + post[0] * t_post / 3600.0, color="#d1495b", lw=1.5,
                label=f"Regresión post-calentamiento: {post[0]:+.3f} MB/h")
    ax.axvspan(0, cut / 60.0, color="0.85", alpha=0.6, label="Calentamiento")

    span = data["ram_rss_mb"].max() - data["ram_rss_mb"].min()
    margin = max(5.0, span * 0.5)
    upper = data["ram_rss_mb"].max() + margin
    if upper >= 0.8 * RSS_MAX_MB:
        ax.axhline(RSS_MAX_MB, color="black", ls=":", lw=1.0, label="Límite RNF-04 (350 MB)")
        upper = max(upper, RSS_MAX_MB * 1.05)
    ax.set_ylim(max(0.0, data["ram_rss_mb"].min() - margin), upper)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("RSS (MB)")
    ax.set_title(f"Estabilidad de memoria — {scenario.label}")
    ax.legend(loc="lower right", fontsize=7)
    _save(fig, path)


def plot_thermal_power(scenario: Scenario, path: Path) -> None:
    hw = scenario.hardware
    if hw["temp_soc_c"].isna().all() and hw["power_w"].isna().all():
        warn("thermal_power_profile: no temperature/power samples.")
        return
    minutes = hw["t_s"] / 60.0
    fig, ax_t = plt.subplots(figsize=(7.0, 3.6))
    ax_p = ax_t.twinx()
    temp_color, power_color = "#d1495b", "#30638e"

    ax_t.plot(minutes, hw["temp_soc_c"], color=temp_color, lw=1.0, label="Temperatura SoC")
    if "npu_temp_c" in hw and hw["npu_temp_c"].notna().any():
        ax_t.plot(minutes, hw["npu_temp_c"], color="#a05195", lw=0.8, ls="-.", label="Temperatura NPU")
    ax_t.axhline(TEMP_MAX_C, color=temp_color, ls="--", lw=1.0, label="Límite 70 °C")

    if hw["power_w"].notna().any():
        ax_p.plot(minutes, hw["power_w"], color=power_color, lw=0.4, alpha=0.35)
        ax_p.plot(minutes, rolling_window(hw, "power_w", POWER_WINDOW_S), color=power_color,
                  lw=1.2, label=f"Potencia (media {POWER_WINDOW_S:.0f} s)")
    ax_p.axhline(POWER_MAX_W, color=power_color, ls=":", lw=1.0, label="Límite 12 W")

    flags = hw["throttled"].fillna(0).astype(int) & THROTTLE_ACTIVE_MASK
    if flags.any():
        ax_t.fill_between(minutes, 0, 1, where=flags > 0, transform=ax_t.get_xaxis_transform(),
                          color="orange", alpha=0.25, label="Throttling activo")

    temps = hw["temp_soc_c"].dropna()
    t_low = (temps.min() - 5) if not temps.empty else 30
    t_high = max(TEMP_MAX_C + 5, (temps.max() + 5) if not temps.empty else 0)
    ax_t.set_ylim(t_low, t_high)
    p_max = hw["power_w"].max()
    ax_p.set_ylim(0, max(POWER_MAX_W * 1.15, (p_max * 1.1) if np.isfinite(p_max) else 0))

    ax_t.set_xlabel("Tiempo (min)")
    ax_t.set_ylabel("Temperatura (°C)", color=temp_color)
    ax_p.set_ylabel("Potencia (W)", color=power_color)
    ax_t.tick_params(axis="y", colors=temp_color)
    ax_p.tick_params(axis="y", colors=power_color)
    ax_p.grid(False)
    ax_t.set_title(f"Perfil térmico y de potencia — {scenario.label}")
    handles = ax_t.get_legend_handles_labels()
    handles_p = ax_p.get_legend_handles_labels()
    ax_t.legend(handles[0] + handles_p[0], handles[1] + handles_p[1], loc="lower right",
                fontsize=7, ncol=2)
    _save(fig, path)


def plot_jitter(scenarios: list[Scenario], primary: Scenario, path: Path) -> None:
    records = []
    for scenario in scenarios:
        deltas = interframe_intervals(scenario.pipeline)
        records.append(pd.DataFrame({"dt_ms": deltas, "Escenario": scenario.label}))
    data = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if data.empty:
        warn("interframe_jitter_hist: no pipeline data.")
        return
    upper = 3 * FRAME_PERIOD_MS
    clipped = data[(data["dt_ms"] > 0) & (data["dt_ms"] <= upper)]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    sns.histplot(clipped, x="dt_ms", hue="Escenario", stat="density", common_norm=False,
                 kde=True, binwidth=0.5, element="step", alpha=0.3, ax=ax,
                 palette="deep", kde_kws={"bw_adjust": 1.5})
    ax.axvline(FRAME_PERIOD_MS, color="black", ls="--", lw=1.0, label="33.33 ms (30 Hz)")

    prim = data.loc[data["Escenario"] == primary.label, "dt_ms"]
    if not prim.empty:
        outside = (prim > upper).mean() * 100.0
        ax.text(0.98, 0.95,
                f"{primary.label}\n$\\mu$ = {prim.mean():.2f} ms\n$\\sigma$ = {prim.std():.2f} ms\n"
                f"P99 = {prim.quantile(0.99):.2f} ms\n> {upper:.0f} ms: {outside:.2f} %",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_xlim(0, upper)
    ax.set_xlabel(r"$\Delta t = t_i - t_{i-1}$ (ms)")
    ax.set_ylabel("Densidad")
    ax.set_title("Jitter intercuadro")
    _save(fig, path)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def resolve_run_dir(arg: Optional[str]) -> Path:
    if arg is None:
        latest = RESULTS_DIR / "latest"
        return latest.resolve() if latest.exists() else RESULTS_DIR
    path = Path(arg)
    return path if path.is_absolute() else (RESULTS_DIR / path if (RESULTS_DIR / path).exists()
                                            else path.resolve())


def pick(scenarios: dict[str, Scenario], preferred: str,
         need: Callable[[Scenario], bool]) -> Optional[Scenario]:
    if preferred in scenarios and need(scenarios[preferred]):
        return scenarios[preferred]
    return next((s for s in scenarios.values() if need(s)), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PRIVES benchmark report.")
    parser.add_argument("--run-dir", default=None,
                        help="Run directory (absolute, or relative to benchmarks/results/). "
                             "Default: benchmarks/results/latest")
    parser.add_argument("--output-dir", default=None,
                        help="Report directory (default: <run-dir>/report)")
    parser.add_argument("--primary", default="E1",
                        help="Scenario used for table_rnf.tex and single-scenario figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    scenarios = discover_scenarios(run_dir)
    if not scenarios:
        print(f"No benchmark data found in {run_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "report"
    if not out_dir.is_absolute() and args.output_dir:
        out_dir = RESULTS_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    info(f"run: {run_dir} | scenarios: {', '.join(scenarios)} | output: {out_dir}")

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150})

    results = {key: (s, *evaluate(s)) for key, s in scenarios.items()}
    write_summary(out_dir / "rnf_summary.txt", results)
    info("summary -> rnf_summary.txt")

    for key, (scenario, rows, _extra) in results.items():
        write_latex(out_dir / f"table_rnf_{key}.tex", scenario, rows)

    has_pipe = lambda s: s.pipeline is not None  # noqa: E731
    has_hw = lambda s: s.hardware is not None  # noqa: E731
    has_any = lambda s: has_pipe(s) or has_hw(s)  # noqa: E731

    primary = pick(scenarios, args.primary.upper(), has_any)
    if primary:
        write_latex(out_dir / "table_rnf.tex", primary, results[primary.key][1])
        info(f"LaTeX -> table_rnf.tex ({primary.key})")

    pipe_primary = pick(scenarios, args.primary.upper(), has_pipe)
    hw_primary = pick(scenarios, args.primary.upper(), has_hw)
    if pipe_primary:
        plot_latency_breakdown(pipe_primary, out_dir / "latency_breakdown.pdf")
        plot_jitter([s for s in scenarios.values() if has_pipe(s)], pipe_primary,
                    out_dir / "interframe_jitter_hist.pdf")
    else:
        warn("no pipeline telemetry: latency and jitter figures skipped.")

    mem_source = pick(scenarios, args.primary.upper(),
                      lambda s: has_hw(s) and s.hardware["ram_rss_mb"].notna().any())
    if mem_source:
        plot_memory_stability(mem_source, out_dir / "memory_stability.pdf")
    if hw_primary:
        plot_thermal_power(hw_primary, out_dir / "thermal_power_profile.pdf")

    print((out_dir / "rnf_summary.txt").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
