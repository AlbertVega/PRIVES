#!/usr/bin/env python3
"""Offline rPPG accuracy evaluation of PRIVES on the UBFC-rPPG dataset.

Compares the per-frame CSV exported by PRIVES (`./prives --video ... --output ...`)
against an UBFC-rPPG ground-truth file (`ground_truth.txt` from DATASET_2 or
`gtdump.xmp` from DATASET_1) and produces:

  - ubfc_metrics_summary.txt       Complete plain-text summary (MAE, RMSE, SNR, etc.)
  - table_ubfc.tex                 LaTeX (booktabs) table ready for thesis/paper
  - per_window_metrics.csv         1-second windowed HR estimates, errors, and SNR
  - ppg_comparison.{pdf,png}       Full + zoomed PPG waveform comparison (PRIVES vs GT)
  - psd_snr_comparison.{pdf,png}   Normalized PSD + de Haan SNR harmonic windows
  - hr_tracking.{pdf,png}          Temporal HR tracking (BPM) and absolute error
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import seaborn as sns  # type: ignore
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

# Physiological constants (matching estimation.hpp and de Haan & Jeanne, 2013)
PRIVES_HR_WINDOW_FRAMES = 300        # 10 s @ 30 Hz (estimation::kHrBufLen)
CARDIAC_BAND_PRIVES_HZ = (0.70, 3.00)  # [42, 180] BPM
SNR_BAND_DE_HAAN_HZ = (0.70, 4.00)     # [42, 240] BPM (includes 2nd harmonic)
SNR_FUND_HALF_WIDTH_BPM = 5.0          # +/- 5 BPM around f_GT
SNR_HARM_HALF_WIDTH_BPM = 10.0         # +/- 10 BPM around 2 * f_GT


def info(msg: str) -> None:
    print(f"[evaluate_ubfc] {msg}")


def warn(msg: str) -> None:
    print(f"[evaluate_ubfc] WARNING: {msg}", file=sys.stderr)


# -----------------------------------------------------------------------------
# Ground-Truth Loading (UBFC-rPPG DATASET_2 & DATASET_1)
# -----------------------------------------------------------------------------

@dataclass
class GroundTruthData:
    ppg: np.ndarray        # Contact pulse-oximeter PPG waveform
    hr_bpm: np.ndarray     # Contact pulse-oximeter HR (BPM)
    time_s: np.ndarray     # Timestamps in seconds (starting at 0.0)
    fmt: str               # "DATASET_2_PER_FRAME", "DATASET_1_XMP", or "CSV"
    source_path: Path


def resolve_gt_file(gt_arg: Path) -> Path:
    """Accepts either a direct ground-truth file or a subject folder."""
    if gt_arg.is_file():
        return gt_arg
    if gt_arg.is_dir():
        for candidate in ("ground_truth.txt", "gtdump.xmp", "ground_truth.csv"):
            p = gt_arg / candidate
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"Ground-truth file not found at '{gt_arg}'. "
        "Provide 'ground_truth.txt' (UBFC DATASET_2) or 'gtdump.xmp' (UBFC DATASET_1)."
    )


def _normalize_timestamps_to_seconds(raw_t: np.ndarray) -> np.ndarray:
    t = np.asarray(raw_t, dtype=float).copy()
    if t.size < 2:
        return np.zeros_like(t)
    diffs = np.diff(t)
    pos_diffs = diffs[diffs > 0]
    dt_med = float(np.median(pos_diffs)) if pos_diffs.size > 0 else 1.0 / 30.0
    # If median delta > 1.0 (e.g. ~16 ms or ~33 ms), timestamps are in milliseconds.
    if dt_med > 1.0:
        t = t / 1000.0
    t = t - t[0]
    return t


def load_ubfc_ground_truth(gt_path: Path) -> GroundTruthData:
    """Loads UBFC-rPPG ground truth automatically detecting DATASET_2 vs DATASET_1."""
    path = resolve_gt_file(gt_path)
    suffix = path.suffix.lower()

    # Case 1: UBFC-rPPG DATASET_1 (gtdump.xmp: comma-separated, 4 columns)
    if suffix == ".xmp":
        mat = np.loadtxt(path, delimiter=",")
        if mat.ndim != 2 or mat.shape[1] < 4:
            raise ValueError(f"Unexpected gtdump.xmp shape {mat.shape} in {path}")
        time_s = _normalize_timestamps_to_seconds(mat[:, 0])
        hr_bpm = mat[:, 1].astype(float)
        ppg = mat[:, 3].astype(float)
        return GroundTruthData(ppg=ppg, hr_bpm=hr_bpm, time_s=time_s,
                               fmt="DATASET_1_XMP", source_path=path)

    # Read non-empty lines to distinguish 3-row DATASET_2 text vs tabular CSV
    raw_lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                 if ln.strip()]
    if not raw_lines:
        raise ValueError(f"Ground-truth file is empty: {path}")

    # Case 2: UBFC-rPPG DATASET_2 (ground_truth.txt: 3 whitespace-separated rows)
    #   Row 0: gtTrace (PPG waveform)
    #   Row 1: gtHR    (Heart rate in BPM)
    #   Row 2: gtTime  (Timestamps in seconds)
    first_char = raw_lines[0][0]
    if len(raw_lines) == 3 and (first_char.isdigit() or first_char in "+-."):
        rows = [np.fromstring(ln.replace(",", " "), sep=" ", dtype=float) for ln in raw_lines]
        min_len = min(len(r) for r in rows)
        if min_len < 10:
            raise ValueError(f"Parsed fewer than 10 samples from 3-row file {path}")
        ppg = rows[0][:min_len]
        hr_bpm = rows[1][:min_len]
        time_s = _normalize_timestamps_to_seconds(rows[2][:min_len])
        return GroundTruthData(ppg=ppg, hr_bpm=hr_bpm, time_s=time_s,
                               fmt="DATASET_2_PER_FRAME", source_path=path)

    # Case 3: Generic CSV with header or 3-column whitespace/comma matrix
    if any(ch.isalpha() for ch in raw_lines[0]):
        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}
        ppg_col = next((cols[k] for k in ("ppg", "bvp", "gttrace", "gt_ppg", "wave") if k in cols), None)
        hr_col = next((cols[k] for k in ("hr", "hr_bpm", "gthr", "gt_hr") if k in cols), None)
        t_col = next((cols[k] for k in ("time_s", "time", "timestamp", "gttime") if k in cols), None)
        if ppg_col is None or hr_col is None:
            raise ValueError(f"CSV ground truth must contain PPG and HR columns. Found: {list(df.columns)}")
        ppg = pd.to_numeric(df[ppg_col], errors="coerce").to_numpy(dtype=float)
        hr_bpm = pd.to_numeric(df[hr_col], errors="coerce").to_numpy(dtype=float)
        if t_col is not None:
            time_s = _normalize_timestamps_to_seconds(pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float))
        else:
            time_s = np.arange(len(ppg), dtype=float) / 30.0
        return GroundTruthData(ppg=ppg, hr_bpm=hr_bpm, time_s=time_s,
                               fmt="CSV", source_path=path)

    # Transposed N x 3 numeric matrix fallback
    mat = np.loadtxt(path, delimiter="," if "," in raw_lines[0] else None)
    if mat.ndim == 2 and mat.shape[1] >= 3:
        ppg = mat[:, 0].astype(float)
        hr_bpm = mat[:, 1].astype(float)
        time_s = _normalize_timestamps_to_seconds(mat[:, 2])
        return GroundTruthData(ppg=ppg, hr_bpm=hr_bpm, time_s=time_s,
                               fmt="MATRIX_Nx3", source_path=path)

    raise ValueError(f"Unrecognised ground-truth file structure in {path}")


# -----------------------------------------------------------------------------
# PRIVES CSV Loading & Temporal Alignment
# -----------------------------------------------------------------------------

REQUIRED_PRIVES_COLS = {
    "frame_id", "video_time_s", "face_detected", "bvp_ready",
    "bvp_pos", "ppg_filtered", "hr_ready", "vitals_ready", "hr_updated", "hr_bpm"
}


def load_prives_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"PRIVES estimations CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = REQUIRED_PRIVES_COLS - set(df.columns)
    if missing:
        raise ValueError(f"PRIVES CSV is missing columns {sorted(missing)} in {csv_path}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["frame_id"]).sort_values("frame_id").reset_index(drop=True)
    df["frame_id"] = df["frame_id"].astype(int)
    for flag_col in ("face_detected", "bvp_ready", "hr_ready", "vitals_ready", "hr_updated"):
        df[flag_col] = df[flag_col].fillna(0).astype(int)
    return df


def align_prives_and_gt(prives_df: pd.DataFrame, gt: GroundTruthData) -> tuple[pd.DataFrame, float]:
    """Aligns PRIVES frame records with UBFC-rPPG ground truth and returns (merged_df, fs_hz)."""
    df = prives_df.copy()

    if gt.fmt == "DATASET_2_PER_FRAME":
        # Direct 1-to-1 frame correspondence in UBFC-rPPG DATASET_2
        valid_mask = df["frame_id"] < len(gt.ppg)
        if not valid_mask.all():
            dropped_tail = int((~valid_mask).sum())
            warn(f"Video had {dropped_tail} frames beyond ground_truth.txt length ({len(gt.ppg)}); trimming tail.")
            df = df.loc[valid_mask].reset_index(drop=True)

        fids = df["frame_id"].to_numpy()
        df["time_s"] = gt.time_s[fids]
        df["gt_ppg"] = gt.ppg[fids]
        df["gt_hr_bpm"] = gt.hr_bpm[fids]
    else:
        # Time-based interpolation (e.g. DATASET_1 gtdump.xmp @ ~60 Hz)
        t_vid = df["video_time_s"].to_numpy(dtype=float)
        t_vid = t_vid - t_vid[0]
        max_t = float(gt.time_s[-1]) if gt.time_s.size > 0 else 0.0
        if max_t > 0 and t_vid[-1] > max_t + 0.5:
            warn(f"Trimming video frames exceeding GT duration ({max_t:.2f} s).")
            df = df.loc[t_vid <= max_t].reset_index(drop=True)
            t_vid = df["video_time_s"].to_numpy(dtype=float)
            t_vid = t_vid - t_vid[0]

        df["time_s"] = t_vid
        df["gt_ppg"] = np.interp(t_vid, gt.time_s, gt.ppg)
        df["gt_hr_bpm"] = np.interp(t_vid, gt.time_s, gt.hr_bpm)

    diffs = np.diff(df["time_s"].to_numpy(dtype=float))
    pos_diffs = diffs[diffs > 1e-6]
    fs_hz = (1.0 / float(np.median(pos_diffs))) if pos_diffs.size > 0 else 30.0
    return df, fs_hz


# -----------------------------------------------------------------------------
# Spectral Analysis, SNR (de Haan & Jeanne 2013), and Waveform Metrics
# -----------------------------------------------------------------------------

def compute_power_spectrum(signal: np.ndarray, fs_hz: float, min_nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Hamming-windowed, zero-padded R2C power spectrum P(f) = |FFT(w * (x - mean(x)))|^2."""
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return np.array([]), np.array([])
    x = x - np.mean(x)
    win = np.hamming(x.size)
    xw = x * win
    n_fft = max(min_nfft, 1 << int(np.ceil(np.log2(xw.size))))
    spec = np.fft.rfft(xw, n=n_fft)
    power = (spec.real * spec.real) + (spec.imag * spec.imag)
    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
    return freqs_hz, power


def dominant_frequency_hz(
    signal: np.ndarray,
    fs_hz: float,
    band_hz: tuple[float, float] = CARDIAC_BAND_PRIVES_HZ,
) -> float:
    """Finds dominant spectral peak (Hz) with parabolic sub-bin interpolation."""
    freqs, power = compute_power_spectrum(signal, fs_hz, min_nfft=4096)
    if freqs.size == 0:
        return np.nan
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(mask):
        return np.nan
    idxs = np.flatnonzero(mask)
    rel_peak = int(np.argmax(power[idxs]))
    k = int(idxs[rel_peak])
    peak_bin = float(k)
    if idxs[0] < k < idxs[-1]:
        alpha = np.sqrt(power[k - 1])
        beta = np.sqrt(power[k])
        gamma = np.sqrt(power[k + 1])
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            peak_bin += 0.5 * (alpha - gamma) / denom
    df = freqs[1] - freqs[0]
    return peak_bin * float(df)


def compute_snr_de_haan(
    signal: np.ndarray,
    fs_hz: float,
    f_gt_hz: float,
    band_hz: tuple[float, float] = SNR_BAND_DE_HAAN_HZ,
    fund_half_width_bpm: float = SNR_FUND_HALF_WIDTH_BPM,
    harm_half_width_bpm: float = SNR_HARM_HALF_WIDTH_BPM,
) -> float:
    """Computes spectral rPPG SNR (dB) as defined by de Haan & Jeanne (IEEE TBME, 2013).

    Signal energy integrates power within +/- 5 BPM around the reference fundamental
    frequency f_GT and +/- 10 BPM around the second harmonic 2*f_GT; noise energy
    integrates all remaining bins in [42, 240] BPM ([0.70, 4.00] Hz).
    """
    if not np.isfinite(f_gt_hz) or f_gt_hz <= 0.0:
        return np.nan
    freqs, power = compute_power_spectrum(signal, fs_hz, min_nfft=4096)
    if freqs.size == 0:
        return np.nan

    in_band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(in_band):
        return np.nan

    f_b = freqs[in_band]
    p_b = power[in_band]

    df1 = fund_half_width_bpm / 60.0
    df2 = harm_half_width_bpm / 60.0
    sig_mask = (np.abs(f_b - f_gt_hz) <= df1) | (np.abs(f_b - 2.0 * f_gt_hz) <= df2)

    p_sig = float(np.sum(p_b[sig_mask]))
    p_noise = float(np.sum(p_b[~sig_mask]))
    if p_sig <= 1e-20 or p_noise <= 1e-20:
        return np.nan
    return float(10.0 * np.log10(p_sig / p_noise))


def zscore_on_mask(values: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Z-score normalizes an array using mean & std computed on ref_mask."""
    ref = values[ref_mask & np.isfinite(values)]
    if ref.size < 2:
        ref = values[np.isfinite(values)]
    if ref.size < 2:
        return np.zeros_like(values, dtype=float)
    mu = float(np.mean(ref))
    sd = float(np.std(ref))
    if sd < 1e-12:
        return values - mu
    return (values - mu) / sd


def bandpass_fft_zero_phase(
    signal: np.ndarray,
    fs_hz: float,
    band_hz: tuple[float, float] = CARDIAC_BAND_PRIVES_HZ,
) -> np.ndarray:
    """Zero-phase cardiac bandpass for clean ground-truth baseline removal when correlating."""
    x = np.asarray(signal, dtype=float).copy()
    if x.size < 8:
        return x
    finite = np.isfinite(x)
    if not np.all(finite):
        if not np.any(finite):
            return np.zeros_like(x)
        idx = np.arange(x.size)
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    x_demean = x - np.mean(x)
    spec = np.fft.rfft(x_demean)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    # Smooth cosine taper at band edges (0.05 Hz) to avoid ringing
    mask = np.zeros_like(freqs)
    lo, hi = band_hz
    taper = 0.05
    inside = (freqs >= lo) & (freqs <= hi)
    mask[inside] = 1.0
    low_taper = (freqs >= lo - taper) & (freqs < lo)
    high_taper = (freqs > hi) & (freqs <= hi + taper)
    mask[low_taper] = 0.5 * (1.0 - np.cos(np.pi * (freqs[low_taper] - (lo - taper)) / taper))
    mask[high_taper] = 0.5 * (1.0 + np.cos(np.pi * (freqs[high_taper] - hi) / taper))
    return np.fft.irfft(spec * mask, n=x.size)


def estimate_waveform_alignment(
    prives_ppg: np.ndarray,
    gt_ppg: np.ndarray,
    fs_hz: float,
    max_lag_s: float = 1.0,
) -> tuple[float, int, float, float]:
    """Estimates optimal lag (seconds), polarity (+1/-1), zero-lag r, and aligned Pearson r.

    Accounts for:
      1. Causal 4th-order Butterworth IIR filter group delay in PRIVES.
      2. Physiological Pulse Transit Time (PTT) between facial skin and finger oximeter.
      3. Reflectance rPPG polarity (hemoglobin light absorption vs oximeter convention).
    """
    valid = np.isfinite(prives_ppg) & np.isfinite(gt_ppg)
    p = prives_ppg[valid]
    g = gt_ppg[valid]
    if p.size < int(2 * fs_hz):
        return 0.0, 1, np.nan, np.nan

    # Remove slow non-cardiac baseline drift from GT before cross-correlation
    g_bp = bandpass_fft_zero_phase(g, fs_hz, CARDIAC_BAND_PRIVES_HZ)
    p_norm = (p - np.mean(p)) / (np.std(p) + 1e-12)
    g_norm = (g_bp - np.mean(g_bp)) / (np.std(g_bp) + 1e-12)

    r_zero = float(np.mean(p_norm * g_norm))
    max_lag_samples = min(int(round(max_lag_s * fs_hz)), p_norm.size // 4)

    best_abs_r = -1.0
    best_r = r_zero
    best_lag = 0

    # Positive lag k > 0 means PRIVES lags behind GT by k samples: p_norm[k:] aligns with g_norm[:-k]
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag > 0:
            a = p_norm[lag:]
            b = g_norm[:-lag]
        elif lag < 0:
            a = p_norm[:lag]
            b = g_norm[-lag:]
        else:
            a = p_norm
            b = g_norm
        if a.size < 16:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(r) and abs(r) > best_abs_r:
            best_abs_r = abs(r)
            best_r = r
            best_lag = lag

    polarity = 1 if best_r >= 0.0 else -1
    lag_s = best_lag / fs_hz
    return lag_s, polarity, r_zero, abs(best_r)


def shift_and_orient_signal(sig: np.ndarray, lag_samples: int, polarity: int) -> np.ndarray:
    """Shifts signal earlier by lag_samples and applies polarity for visual pulse overlay."""
    oriented = polarity * sig
    out = np.full_like(oriented, np.nan, dtype=float)
    if lag_samples > 0 and lag_samples < oriented.size:
        out[:-lag_samples] = oriented[lag_samples:]
    elif lag_samples < 0 and (-lag_samples) < oriented.size:
        k = -lag_samples
        out[k:] = oriented[:-k]
    else:
        out[:] = oriented
    return out


# -----------------------------------------------------------------------------
# Evaluation Summary Data Structure
# -----------------------------------------------------------------------------

@dataclass
class EvaluationResults:
    subject: str
    total_frames: int
    face_detected_frames: int
    face_detection_pct: float
    fs_hz: float
    duration_s: float
    eval_mode: str
    num_hr_updates: int
    # HR Error Metrics (Window-matched GT)
    mae_bpm: float
    rmse_bpm: float
    mape_pct: float
    me_bias_bpm: float
    std_err_bpm: float
    max_err_bpm: float
    pearson_hr: float
    # HR Error Metrics (Instantaneous GT reference)
    mae_inst_bpm: float
    rmse_inst_bpm: float
    # Mean HR comparison
    mean_hr_prives_bpm: float
    mean_hr_gt_bpm: float
    # SNR Metrics (de Haan & Jeanne, 2013)
    snr_ppg_window_mean_db: float
    snr_ppg_window_std_db: float
    snr_pos_window_mean_db: float
    snr_pos_window_std_db: float
    snr_ppg_global_db: float
    snr_pos_global_db: float
    # Waveform correlation & lag
    lag_ms: float
    polarity: int
    r_ppg_raw: float
    r_ppg_aligned: float
    # Per-window DataFrame
    windows_df: pd.DataFrame


def evaluate_recording(
    df: pd.DataFrame,
    fs_hz: float,
    subject: str,
    eval_from: str = "vitals_ready",
) -> EvaluationResults:
    total_frames = len(df)
    face_frames = int((df["face_detected"] == 1).sum())
    face_pct = (100.0 * face_frames / total_frames) if total_frames > 0 else 0.0
    duration_s = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if total_frames > 1 else 0.0

    # Determine calibrated update rows
    ready_col = "vitals_ready" if eval_from == "vitals_ready" else "hr_ready"
    upd_mask = (df["hr_updated"] == 1) & (df[ready_col] == 1) & df["hr_bpm"].notna()
    used_mode = ready_col
    if not upd_mask.any() and ready_col == "vitals_ready":
        warn("No HR updates reached 'vitals_ready' (12 s); falling back to 'hr_ready' (7 s).")
        used_mode = "hr_ready"
        upd_mask = (df["hr_updated"] == 1) & (df["hr_ready"] == 1) & df["hr_bpm"].notna()

    upd_indices = np.flatnonzero(upd_mask.to_numpy())
    if upd_indices.size == 0:
        raise RuntimeError(
            "PRIVES CSV contains 0 calibrated HR updates. Ensure the video is at least "
            "8-12 seconds long and that a face was detected consistently."
        )

    win_len = int(round(10.0 * fs_hz))  # ~300 frames (10 s window matching estimation::kHrBufLen)
    records = []

    gt_hr_arr = df["gt_hr_bpm"].to_numpy(dtype=float)
    gt_ppg_arr = df["gt_ppg"].to_numpy(dtype=float)
    ppg_filt_arr = df["ppg_filtered"].to_numpy(dtype=float)
    bvp_pos_arr = df["bvp_pos"].to_numpy(dtype=float)
    time_arr = df["time_s"].to_numpy(dtype=float)
    hr_est_arr = df["hr_bpm"].to_numpy(dtype=float)

    for idx in upd_indices:
        start = max(0, int(idx) - win_len + 1)
        end = int(idx) + 1

        hr_est = float(hr_est_arr[idx])
        hr_gt_inst = float(gt_hr_arr[idx])
        hr_gt_win = float(np.nanmean(gt_hr_arr[start:end]))

        # Reference fundamental frequency for de Haan SNR in this 10 s window:
        # prefer GT PPG spectral peak; fall back to oximeter windowed HR / 60.
        gt_seg = gt_ppg_arr[start:end]
        f_gt_spec = dominant_frequency_hz(gt_seg, fs_hz, CARDIAC_BAND_PRIVES_HZ)
        f_gt_hz = f_gt_spec if (np.isfinite(f_gt_spec) and abs(f_gt_spec * 60.0 - hr_gt_win) <= 25.0) \
            else (hr_gt_win / 60.0)

        ppg_seg = ppg_filt_arr[start:end]
        pos_seg = bvp_pos_arr[start:end]

        snr_ppg = compute_snr_de_haan(ppg_seg[np.isfinite(ppg_seg)], fs_hz, f_gt_hz)
        snr_pos = compute_snr_de_haan(pos_seg[np.isfinite(pos_seg)], fs_hz, f_gt_hz)

        err_win = hr_est - hr_gt_win
        err_inst = hr_est - hr_gt_inst

        records.append({
            "frame_id": int(df["frame_id"].iloc[idx]),
            "time_s": float(time_arr[idx]),
            "hr_prives_bpm": hr_est,
            "hr_gt_window_bpm": hr_gt_win,
            "hr_gt_inst_bpm": hr_gt_inst,
            "f_gt_ref_bpm": f_gt_hz * 60.0,
            "error_bpm": err_win,
            "abs_error_bpm": abs(err_win),
            "abs_error_inst_bpm": abs(err_inst),
            "snr_ppg_filtered_db": snr_ppg,
            "snr_bvp_pos_db": snr_pos,
        })

    win_df = pd.DataFrame(records)

    abs_err = win_df["abs_error_bpm"].to_numpy(dtype=float)
    err = win_df["error_bpm"].to_numpy(dtype=float)
    hr_p = win_df["hr_prives_bpm"].to_numpy(dtype=float)
    hr_g = win_df["hr_gt_window_bpm"].to_numpy(dtype=float)

    mae_bpm = float(np.mean(abs_err))
    rmse_bpm = float(np.sqrt(np.mean(err ** 2)))
    mape_pct = float(np.mean(abs_err / np.maximum(hr_g, 1e-6)) * 100.0)
    me_bias = float(np.mean(err))
    std_err = float(np.std(err, ddof=1)) if err.size > 1 else 0.0
    max_err = float(np.max(abs_err))

    if hr_p.size > 2 and np.std(hr_p) > 1e-6 and np.std(hr_g) > 1e-6:
        pearson_hr = float(np.corrcoef(hr_p, hr_g)[0, 1])
    else:
        pearson_hr = np.nan

    abs_err_inst = win_df["abs_error_inst_bpm"].to_numpy(dtype=float)
    err_inst = hr_p - win_df["hr_gt_inst_bpm"].to_numpy(dtype=float)
    mae_inst = float(np.mean(abs_err_inst))
    rmse_inst = float(np.sqrt(np.mean(err_inst ** 2)))

    # Global SNR & waveform correlation on the steady-state calibrated waveform
    steady_mask = (df["hr_ready"] == 1).to_numpy() & np.isfinite(ppg_filt_arr)
    if np.sum(steady_mask) < int(3 * fs_hz):
        steady_mask = (df["bvp_ready"] == 1).to_numpy() & np.isfinite(ppg_filt_arr)

    f_gt_global = dominant_frequency_hz(gt_ppg_arr[steady_mask], fs_hz, CARDIAC_BAND_PRIVES_HZ)
    if not np.isfinite(f_gt_global):
        f_gt_global = float(np.nanmean(gt_hr_arr[steady_mask])) / 60.0

    snr_ppg_global = compute_snr_de_haan(ppg_filt_arr[steady_mask], fs_hz, f_gt_global)
    snr_pos_global = compute_snr_de_haan(bvp_pos_arr[steady_mask], fs_hz, f_gt_global)

    snr_ppg_vals = win_df["snr_ppg_filtered_db"].dropna().to_numpy(dtype=float)
    snr_pos_vals = win_df["snr_bvp_pos_db"].dropna().to_numpy(dtype=float)

    steady_indices = np.flatnonzero(steady_mask)
    if steady_indices.size >= int(2 * fs_hz):
        s0 = int(steady_indices[0])
        lag_s, polarity, r_raw, r_aligned = estimate_waveform_alignment(
            np.nan_to_num(ppg_filt_arr[s0:], nan=0.0),
            gt_ppg_arr[s0:],
            fs_hz,
        )
    else:
        lag_s, polarity, r_raw, r_aligned = 0.0, 1, np.nan, np.nan

    return EvaluationResults(
        subject=subject,
        total_frames=total_frames,
        face_detected_frames=face_frames,
        face_detection_pct=face_pct,
        fs_hz=fs_hz,
        duration_s=duration_s,
        eval_mode=used_mode,
        num_hr_updates=len(win_df),
        mae_bpm=mae_bpm,
        rmse_bpm=rmse_bpm,
        mape_pct=mape_pct,
        me_bias_bpm=me_bias,
        std_err_bpm=std_err,
        max_err_bpm=max_err,
        pearson_hr=pearson_hr,
        mae_inst_bpm=mae_inst,
        rmse_inst_bpm=rmse_inst,
        mean_hr_prives_bpm=float(np.mean(hr_p)),
        mean_hr_gt_bpm=float(np.mean(hr_g)),
        snr_ppg_window_mean_db=float(np.mean(snr_ppg_vals)) if snr_ppg_vals.size else np.nan,
        snr_ppg_window_std_db=float(np.std(snr_ppg_vals, ddof=1)) if snr_ppg_vals.size > 1 else 0.0,
        snr_pos_window_mean_db=float(np.mean(snr_pos_vals)) if snr_pos_vals.size else np.nan,
        snr_pos_window_std_db=float(np.std(snr_pos_vals, ddof=1)) if snr_pos_vals.size > 1 else 0.0,
        snr_ppg_global_db=snr_ppg_global,
        snr_pos_global_db=snr_pos_global,
        lag_ms=lag_s * 1000.0,
        polarity=polarity,
        r_ppg_raw=r_raw,
        r_ppg_aligned=r_aligned,
        windows_df=win_df,
    )


# -----------------------------------------------------------------------------
# Figures (PDF + PNG, matching PRIVES academic benchmark aesthetic)
# -----------------------------------------------------------------------------

COLOR_GT = "#30638e"      # Deep slate blue (Ground Truth)
COLOR_PRIVES = "#d1495b"  # Crimson red (PRIVES)
COLOR_POS = "#edae49"     # Warm amber (Raw POS / auxiliary)
COLOR_BAND = "#2a9d8f"    # Teal (SNR harmonic bands)


def configure_plot_style() -> None:
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="paper")
    else:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.rcParams["axes.grid"] = True
            plt.rcParams["grid.alpha"] = 0.35
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
    })


def save_figure(fig: plt.Figure, stem_path: Path) -> None:
    fig.tight_layout()
    pdf_path = stem_path.with_suffix(".pdf")
    png_path = stem_path.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    info(f"figure -> {pdf_path.name}, {png_path.name}")


def plot_ppg_comparison(
    df: pd.DataFrame,
    res: EvaluationResults,
    stem_path: Path,
    zoom_start_s: Optional[float] = None,
    zoom_duration_s: float = 12.0,
) -> None:
    """Three-panel PPG comparison:
       (a) Full-session overview (z-score normalized on steady state, warm-up shaded).
       (b) Zoomed window: direct synchronized waveforms (PRIVES filtered PPG vs GT PPG).
       (c) Zoomed window: delay & reflectance-polarity aligned overlay showing pulse morphology.
    """
    t = df["time_s"].to_numpy(dtype=float)
    ppg_raw = df["ppg_filtered"].to_numpy(dtype=float)
    gt_raw = df["gt_ppg"].to_numpy(dtype=float)

    steady_mask = (df["hr_ready"] == 1).to_numpy() & np.isfinite(ppg_raw)
    if np.sum(steady_mask) < 30:
        steady_mask = (df["bvp_ready"] == 1).to_numpy() & np.isfinite(ppg_raw)

    ppg_z = zscore_on_mask(ppg_raw, steady_mask)
    gt_bp = bandpass_fft_zero_phase(gt_raw, res.fs_hz, CARDIAC_BAND_PRIVES_HZ)
    gt_z = zscore_on_mask(gt_raw, steady_mask)
    gt_bp_z = zscore_on_mask(gt_bp, steady_mask)

    # Determine calibration cut timestamp
    ready_times = t[steady_mask]
    calib_end_s = float(ready_times[0]) if ready_times.size > 0 else 0.0

    # Determine zoom window inside steady-state segment
    t_min_steady = calib_end_s
    t_max = float(t[-1])
    if zoom_start_s is None:
        avail = max(0.0, t_max - t_min_steady)
        dur = min(zoom_duration_s, avail) if avail > 2.0 else min(zoom_duration_s, t_max)
        z_start = t_min_steady + max(0.0, 0.5 * (avail - dur))
    else:
        z_start = max(0.0, float(zoom_start_s))
        dur = min(zoom_duration_s, max(1.0, t_max - z_start))
    z_end = z_start + dur

    lag_samples = int(round((res.lag_ms / 1000.0) * res.fs_hz))
    ppg_aligned = shift_and_orient_signal(ppg_z, lag_samples, res.polarity)

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharey=False)

    # Panel (a): Full recording overview
    ax0 = axes[0]
    ax0.plot(t, gt_z, color=COLOR_GT, lw=0.85, alpha=0.85, label="PPG Ground Truth (oxímetro)")
    ax0.plot(t, ppg_z, color=COLOR_PRIVES, lw=0.85, alpha=0.90, label="PPG estimada por PRIVES (filtrada)")
    if calib_end_s > 0.0:
        ax0.axvspan(float(t[0]), calib_end_s, color="0.85", alpha=0.65, label="Calentamiento POS + IIR")
    ax0.axvspan(z_start, z_end, color=COLOR_POS, alpha=0.18, label=f"Ventana ampliada ({dur:.0f} s)")
    ax0.set_xlim(float(t[0]), t_max)
    ax0.set_ylim(-3.8, 3.8)
    ax0.set_ylabel("Amplitud (z-score)")
    ax0.set_title(
        f"Comparación de señal PPG: PRIVES vs. Ground Truth — {res.subject} "
        f"(SNR = {res.snr_ppg_window_mean_db:.2f} dB, MAE = {res.mae_bpm:.2f} BPM)",
        fontsize=9.5,
    )
    ax0.legend(loc="upper right", fontsize=7.5, ncol=2, frameon=True)

    # Zoom mask
    z_mask = (t >= z_start) & (t <= z_end)

    # Panel (b): Zoomed direct comparison (unshifted)
    ax1 = axes[1]
    ax1.plot(t[z_mask], gt_z[z_mask], color=COLOR_GT, lw=1.3, label="PPG Ground Truth (z-score)")
    ax1.plot(t[z_mask], ppg_z[z_mask], color=COLOR_PRIVES, lw=1.3,
             label="PPG PRIVES directa (salida filtro IIR [0.70, 3.00] Hz)")
    ax1.set_xlim(z_start, z_end)
    ax1.set_ylim(-3.2, 3.2)
    ax1.set_ylabel("Amplitud (z-score)")
    ax1.set_title(
        f"Detalle temporal sincronizado ({z_start:.1f} s – {z_end:.1f} s)",
        fontsize=9.0,
    )
    ax1.legend(loc="upper right", fontsize=7.5, ncol=2, frameon=True)

    # Panel (c): Zoomed phase/polarity-compensated morphological comparison
    ax2 = axes[2]
    pol_str = "invertida por reflectancia" if res.polarity < 0 else "en fase"
    ax2.plot(t[z_mask], gt_bp_z[z_mask], color=COLOR_GT, lw=1.35,
             label="PPG Ground Truth (banda cardíaca)")
    ax2.plot(t[z_mask], ppg_aligned[z_mask], color=COLOR_PRIVES, lw=1.35, ls="--",
             label=f"PPG PRIVES alineada (Δτ = {res.lag_ms:+.0f} ms, {pol_str}, r = {res.r_ppg_aligned:.2f})")
    ax2.set_xlim(z_start, z_end)
    ax2.set_ylim(-3.2, 3.2)
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Amplitud (z-score)")
    ax2.set_title(
        "Comparación morfológica compensando retardo de grupo IIR + PTT rostro-dedo",
        fontsize=9.0,
    )
    ax2.legend(loc="upper right", fontsize=7.5, frameon=True)

    save_figure(fig, stem_path)


def plot_psd_snr(df: pd.DataFrame, res: EvaluationResults, stem_path: Path) -> None:
    """Plots normalized Power Spectral Density (BPM) of PRIVES PPG vs GT PPG
    with de Haan & Jeanne (2013) fundamental and 2nd-harmonic SNR windows shaded.
    """
    ppg_filt = df["ppg_filtered"].to_numpy(dtype=float)
    bvp_pos = df["bvp_pos"].to_numpy(dtype=float)
    gt_ppg = df["gt_ppg"].to_numpy(dtype=float)

    steady = (df["hr_ready"] == 1).to_numpy() & np.isfinite(ppg_filt)
    if np.sum(steady) < 30:
        steady = (df["bvp_ready"] == 1).to_numpy() & np.isfinite(ppg_filt)

    f_gt, p_gt = compute_power_spectrum(gt_ppg[steady], res.fs_hz)
    f_pr, p_pr = compute_power_spectrum(ppg_filt[steady], res.fs_hz)
    f_po, p_po = compute_power_spectrum(bvp_pos[steady], res.fs_hz)

    if f_pr.size == 0 or f_gt.size == 0:
        warn("plot_psd_snr: insufficient waveform samples for spectrum plot.")
        return

    band_gt = (f_gt >= SNR_BAND_DE_HAAN_HZ[0]) & (f_gt <= SNR_BAND_DE_HAAN_HZ[1])
    band_pr = (f_pr >= SNR_BAND_DE_HAAN_HZ[0]) & (f_pr <= SNR_BAND_DE_HAAN_HZ[1])
    band_po = (f_po >= SNR_BAND_DE_HAAN_HZ[0]) & (f_po <= SNR_BAND_DE_HAAN_HZ[1])

    bpm_gt = f_gt[band_gt] * 60.0
    bpm_pr = f_pr[band_pr] * 60.0
    bpm_po = f_po[band_po] * 60.0

    norm_gt = p_gt[band_gt] / (np.max(p_gt[band_gt]) + 1e-20)
    norm_pr = p_pr[band_pr] / (np.max(p_pr[band_pr]) + 1e-20)
    norm_po = p_po[band_po] / (np.max(p_po[band_po]) + 1e-20)

    f0_hz = dominant_frequency_hz(gt_ppg[steady], res.fs_hz, CARDIAC_BAND_PRIVES_HZ)
    f0_bpm = f0_hz * 60.0 if np.isfinite(f0_hz) else res.mean_hr_gt_bpm

    fig, ax = plt.subplots(figsize=(7.6, 3.8))

    # Shade de Haan SNR fundamental & 2nd harmonic windows
    ax.axvspan(f0_bpm - SNR_FUND_HALF_WIDTH_BPM, f0_bpm + SNR_FUND_HALF_WIDTH_BPM,
               color=COLOR_BAND, alpha=0.20,
               label=f"Ventana fundamental SNR ({f0_bpm:.1f} ± {SNR_FUND_HALF_WIDTH_BPM:.0f} BPM)")
    if 2.0 * f0_bpm - SNR_HARM_HALF_WIDTH_BPM <= 240.0:
        ax.axvspan(2.0 * f0_bpm - SNR_HARM_HALF_WIDTH_BPM,
                   min(240.0, 2.0 * f0_bpm + SNR_HARM_HALF_WIDTH_BPM),
                   color=COLOR_BAND, alpha=0.11,
                   label=f"2.º armónico SNR ({2.0 * f0_bpm:.1f} ± {SNR_HARM_HALF_WIDTH_BPM:.0f} BPM)")

    ax.plot(bpm_gt, norm_gt, color=COLOR_GT, lw=1.4, label="PSD Ground Truth PPG")
    ax.plot(bpm_po, norm_po, color=COLOR_POS, lw=1.0, alpha=0.75, ls=":",
            label=f"PSD BVP POS cruda (SNR = {res.snr_pos_window_mean_db:.2f} dB)")
    ax.plot(bpm_pr, norm_pr, color=COLOR_PRIVES, lw=1.4,
            label=f"PSD PPG PRIVES filtrada (SNR = {res.snr_ppg_window_mean_db:.2f} dB)")

    ax.axvline(180.0, color="0.45", lw=0.9, ls="--", label="Corte superior filtro HR PRIVES (180 BPM)")

    ax.set_xlim(42.0, 240.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Frecuencia (BPM)")
    ax.set_ylabel("Potencia espectral normalizada")
    ax.set_title(f"Densidad espectral de potencia y bandas SNR (de Haan) — {res.subject}")
    ax.legend(loc="upper right", fontsize=7.3, frameon=True)

    save_figure(fig, stem_path)


def plot_hr_tracking(df: pd.DataFrame, res: EvaluationResults, stem_path: Path) -> None:
    """Two-panel HR tracking figure:
       Top   : Continuous GT HR + windowed GT HR vs PRIVES 1-second HR estimates.
       Bottom: Instantaneous absolute error |HR_PRIVES - HR_GT| (BPM) + MAE line.
    """
    t_all = df["time_s"].to_numpy(dtype=float)
    gt_hr_all = df["gt_hr_bpm"].to_numpy(dtype=float)

    wdf = res.windows_df
    t_u = wdf["time_s"].to_numpy(dtype=float)
    hr_p = wdf["hr_prives_bpm"].to_numpy(dtype=float)
    hr_g = wdf["hr_gt_window_bpm"].to_numpy(dtype=float)
    abs_e = wdf["abs_error_bpm"].to_numpy(dtype=float)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7.8, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.1]}
    )

    ax0.plot(t_all, gt_hr_all, color=COLOR_GT, lw=0.9, alpha=0.45, label="HR Ground Truth (instantáneo)")
    ax0.plot(t_u, hr_g, color=COLOR_GT, lw=1.6, label="HR Ground Truth (media ventana 10 s)")
    ax0.fill_between(t_u, hr_g - 5.0, hr_g + 5.0, color=COLOR_GT, alpha=0.12, label="Banda ±5 BPM")
    ax0.plot(t_u, hr_p, color=COLOR_PRIVES, lw=1.6, marker="o", ms=3.2,
             label="HR estimado por PRIVES (1 Hz)")

    if t_u.size > 0:
        ax0.axvspan(float(t_all[0]), float(t_u[0]), color="0.85", alpha=0.6, label="Calibración inicial")

    stats_box = (
        f"MAE  = {res.mae_bpm:.2f} BPM\n"
        f"RMSE = {res.rmse_bpm:.2f} BPM\n"
        f"SNR  = {res.snr_ppg_window_mean_db:.2f} dB\n"
        f"N    = {res.num_hr_updates} ventanas"
    )
    ax0.text(
        0.02, 0.95, stats_box, transform=ax0.transAxes, va="top", ha="left",
        fontsize=8.0, bbox=dict(boxstyle="round", facecolor="white", alpha=0.88)
    )

    y_vals = np.concatenate([gt_hr_all[np.isfinite(gt_hr_all)], hr_p[np.isfinite(hr_p)]])
    if y_vals.size > 0:
        margin = max(8.0, 0.15 * (np.max(y_vals) - np.min(y_vals)))
        ax0.set_ylim(max(35.0, float(np.min(y_vals)) - margin), float(np.max(y_vals)) + margin)

    ax0.set_ylabel("Frecuencia cardíaca (BPM)")
    ax0.set_title(f"Seguimiento temporal de frecuencia cardíaca — {res.subject}")
    ax0.legend(loc="upper right", fontsize=7.3, ncol=2, frameon=True)

    # Bottom panel: Absolute Error
    ax1.bar(t_u, abs_e, width=0.72, color=COLOR_PRIVES, alpha=0.72, label=r"$|\widehat{\mathrm{HR}} - \mathrm{HR}_{\mathrm{GT}}|$")
    ax1.axhline(res.mae_bpm, color="black", ls="--", lw=1.2, label=f"MAE = {res.mae_bpm:.2f} BPM")
    ax1.axhline(5.0, color=COLOR_BAND, ls=":", lw=1.1, label="Referencia clínica 5 BPM (ANSI/AAMI)")
    ax1.set_xlim(float(t_all[0]), float(t_all[-1]))
    ax1.set_ylim(0.0, max(7.0, float(np.max(abs_e)) * 1.2) if abs_e.size else 10.0)
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Error abs. (BPM)")
    ax1.legend(loc="upper right", fontsize=7.3, ncol=3, frameon=True)

    save_figure(fig, stem_path)


# -----------------------------------------------------------------------------
# Text Summary & LaTeX Table Generation
# -----------------------------------------------------------------------------

def _fmt(val: float, digits: int = 2) -> str:
    return "--" if not np.isfinite(val) else f"{val:.{digits}f}"


def write_summary_text(path: Path, res: EvaluationResults, prives_csv: Path, gt_path: Path) -> str:
    pol_note = "Invertida (-1, absorción por reflectancia)" if res.polarity < 0 else "En fase (+1)"
    lines = [
        "==============================================================================",
        f"  PRIVES - Evaluación con Ground Truth UBFC-rPPG ({res.subject})",
        "==============================================================================",
        f"Archivo PRIVES CSV : {prives_csv}",
        f"Archivo GroundTruth: {gt_path}",
        f"Duración analizada : {res.duration_s:.2f} s ({res.total_frames} cuadros @ {res.fs_hz:.2f} Hz)",
        f"Detección facial   : {res.face_detected_frames}/{res.total_frames} cuadros ({res.face_detection_pct:.2f} %)",
        f"Ventanas evaluadas : {res.num_hr_updates} actualizaciones de 1 Hz (modo: {res.eval_mode})",
        "",
        "--- 1. Exactitud de Frecuencia Cardíaca (HR) ---------------------------------",
        f"  HR medio Ground Truth (ventana 10 s) : {_fmt(res.mean_hr_gt_bpm)} BPM",
        f"  HR medio estimado por PRIVES         : {_fmt(res.mean_hr_prives_bpm)} BPM",
        f"  MAE  (Error Absoluto Medio)          : {_fmt(res.mae_bpm)} BPM",
        f"  RMSE (Error Cuadrático Medio)        : {_fmt(res.rmse_bpm)} BPM",
        f"  MAPE (Error Porcentual Medio)        : {_fmt(res.mape_pct)} %",
        f"  Sesgo (ME) ± Desv. Estándar          : {_fmt(res.me_bias_bpm)} ± {_fmt(res.std_err_bpm)} BPM",
        f"  Error absoluto máximo                : {_fmt(res.max_err_bpm)} BPM",
        f"  Correlación de Pearson (r_HR)        : {_fmt(res.pearson_hr, 3)}",
        f"  [Ref. GT instantáneo] MAE / RMSE     : {_fmt(res.mae_inst_bpm)} / {_fmt(res.rmse_inst_bpm)} BPM",
        "",
        "--- 2. Calidad Espectral de Señal rPPG (SNR de Haan & Jeanne, 2013) ----------",
        f"  SNR PPG filtrada PRIVES (media±desv) : {_fmt(res.snr_ppg_window_mean_db)} ± {_fmt(res.snr_ppg_window_std_db)} dB",
        f"  SNR PPG filtrada PRIVES (global)     : {_fmt(res.snr_ppg_global_db)} dB",
        f"  SNR BVP POS cruda (media±desv)       : {_fmt(res.snr_pos_window_mean_db)} ± {_fmt(res.snr_pos_window_std_db)} dB",
        f"  SNR BVP POS cruda (global)           : {_fmt(res.snr_pos_global_db)} dB",
        "",
        "--- 3. Comparación Temporal de Forma de Onda PPG -----------------------------",
        f"  Retardo óptimo estimado (IIR + PTT)  : {_fmt(res.lag_ms, 1)} ms",
        f"  Polaridad relativa rPPG vs oxímetro  : {pol_note}",
        f"  Correlación forma de onda (alineada) : r = {_fmt(res.r_ppg_aligned, 3)} (sin alinear: {_fmt(res.r_ppg_raw, 3)})",
        "==============================================================================",
        "",
    ]
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def write_latex_table(path: Path, res: EvaluationResults) -> None:
    safe_subj = res.subject.replace("_", r"\_")
    label_key = "".join(ch.lower() if ch.isalnum() else "_" for ch in res.subject)
    lines = [
        "% Auto-generated by benchmarks/analysis/evaluate_ubfc.py",
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Evaluación fisiológica del sistema PRIVES sobre UBFC-rPPG ({safe_subj}).}}",
        f"  \\label{{tab:ubfc_{label_key}}}",
        "  \\begin{tabular}{lrc}",
        "    \\toprule",
        "    Métrica & Valor & Unidad \\\\",
        "    \\midrule",
        f"    HR medio Ground Truth & {_fmt(res.mean_hr_gt_bpm)} & BPM \\\\",
        f"    HR medio PRIVES & {_fmt(res.mean_hr_prives_bpm)} & BPM \\\\",
        f"    Error absoluto medio (MAE) & \\textbf{{{_fmt(res.mae_bpm)}}} & BPM \\\\",
        f"    Error cuadrático medio (RMSE) & {_fmt(res.rmse_bpm)} & BPM \\\\",
        f"    Error porcentual absoluto medio (MAPE) & {_fmt(res.mape_pct)} & \\% \\\\",
        f"    Sesgo ($\\mu_e \\pm \\sigma_e$) & {_fmt(res.me_bias_bpm)} $\\pm$ {_fmt(res.std_err_bpm)} & BPM \\\\",
        f"    SNR espectral (PPG filtrada, ventana 10 s) & \\textbf{{{_fmt(res.snr_ppg_window_mean_db)}}} & dB \\\\",
        f"    SNR espectral (PPG filtrada, global) & {_fmt(res.snr_ppg_global_db)} & dB \\\\",
        f"    SNR espectral (BVP POS cruda, ventana 10 s) & {_fmt(res.snr_pos_window_mean_db)} & dB \\\\",
        f"    Tasa de detección facial (SCRFD NPU) & {_fmt(res.face_detection_pct)} & \\% \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PRIVES CSV outputs against UBFC-rPPG Ground Truth (MAE, SNR, PPG plots)."
    )
    parser.add_argument(
        "--prives-csv", required=True,
        help="Path to the CSV generated by './prives --video <vid.avi> --output <estimations.csv>'."
    )
    parser.add_argument(
        "--gt", required=True,
        help="Path to UBFC-rPPG 'ground_truth.txt' (DATASET_2), 'gtdump.xmp' (DATASET_1), "
             "or the subject directory containing it."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save plots, tables, and summary (default: benchmarks/results/ubfc_<subject>)."
    )
    parser.add_argument(
        "--subject", default=None,
        help="Subject identifier for plot titles and tables (e.g. 'subject1'). Auto-inferred if omitted."
    )
    parser.add_argument(
        "--eval-from", choices=("vitals_ready", "hr_ready"), default="vitals_ready",
        help="Start MAE/SNR evaluation from full 10 s calibration ('vitals_ready', default) "
             "or 5 s HR warm-up ('hr_ready')."
    )
    parser.add_argument(
        "--zoom-start", type=float, default=None,
        help="Start timestamp (seconds) for the zoomed PPG waveform panels (default: auto middle)."
    )
    parser.add_argument(
        "--zoom-duration", type=float, default=12.0,
        help="Duration (seconds) of the zoomed PPG waveform panels (default: 12.0)."
    )
    return parser.parse_args()


def infer_subject_name(gt_file: Path, prives_csv: Path) -> str:
    if gt_file.parent.name and gt_file.parent.name.lower() not in ("build", "results", ".", ""):
        return gt_file.parent.name
    return prives_csv.stem


def main() -> int:
    args = parse_args()
    prives_csv_path = Path(args.prives_csv).resolve()
    gt_input_path = Path(args.gt).resolve()

    try:
        gt = load_ubfc_ground_truth(gt_input_path)
        prives_df = load_prives_csv(prives_csv_path)
    except Exception as exc:
        print(f"[evaluate_ubfc] ERROR loading inputs: {exc}", file=sys.stderr)
        return 1

    subject = args.subject or infer_subject_name(gt.source_path, prives_csv_path)
    out_dir = Path(args.output_dir).resolve() if args.output_dir else (RESULTS_ROOT / f"ubfc_{subject}")
    out_dir.mkdir(parents=True, exist_ok=True)

    info(f"Subject: {subject} | GT format: {gt.fmt} ({len(gt.ppg)} samples)")
    info(f"PRIVES frames: {len(prives_df)} | Output dir: {out_dir}")

    merged_df, fs_hz = align_prives_and_gt(prives_df, gt)

    try:
        res = evaluate_recording(merged_df, fs_hz=fs_hz, subject=subject, eval_from=args.eval_from)
    except Exception as exc:
        print(f"[evaluate_ubfc] ERROR during evaluation: {exc}", file=sys.stderr)
        return 1

    configure_plot_style()

    # Save tabular outputs
    res.windows_df.to_csv(out_dir / "per_window_metrics.csv", index=False)
    merged_df.to_csv(out_dir / "aligned_signals.csv", index=False)
    summary_txt = write_summary_text(
        out_dir / "ubfc_metrics_summary.txt", res, prives_csv_path, gt.source_path
    )
    write_latex_table(out_dir / "table_ubfc.tex", res)

    # Save plots (PDF + PNG)
    plot_ppg_comparison(
        merged_df, res, out_dir / "ppg_comparison",
        zoom_start_s=args.zoom_start, zoom_duration_s=args.zoom_duration,
    )
    plot_psd_snr(merged_df, res, out_dir / "psd_snr_comparison")
    plot_hr_tracking(merged_df, res, out_dir / "hr_tracking")

    print("\n" + summary_txt)
    info(f"All deliverables written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
