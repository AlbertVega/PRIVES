#!/usr/bin/env bash
# End-to-end UBFC-rPPG evaluation runner for PRIVES (Raspberry Pi 5 + Hailo-8).
#
# Executes the PRIVES C++ binary on an UBFC-rPPG video using the Hailo-8 NPU,
# exports per-frame BVP/PPG waveforms and HR/FR estimations to CSV, and runs
# benchmarks/analysis/evaluate_ubfc.py to compute MAE, RMSE, de Haan SNR, and
# generate PPG + PSD + HR comparison plots.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BENCH_DIR="$(dirname "${SCRIPT_DIR}")"
readonly REPO_ROOT="$(dirname "${BENCH_DIR}")"
readonly RESULTS_ROOT="${BENCH_DIR}/results"

PRIVES_BIN="${PRIVES_BIN:-${REPO_ROOT}/build/prives}"
HEF_PATH="${HEF_PATH:-${REPO_ROOT}/build/scrfd_2.5g.hef}"
SUBJECT_DIR=""
VIDEO_PATH=""
GT_PATH=""
SUBJECT_NAME=""
OUTPUT_DIR=""
REALTIME=0
ENABLE_STREAM=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Input selection (choose one):
  --subject-dir DIR   UBFC-rPPG subject folder containing vid.avi and ground_truth.txt (or gtdump.xmp)
  --video PATH        Explicit path to input video (e.g. build/vid.avi)
  --gt PATH           Explicit path to ground_truth.txt or gtdump.xmp (optional if only running C++ step)

Options:
  --subject NAME      Subject label for plots & tables (default: inferred from folder/video)
  --output-dir DIR    Output folder (default: benchmarks/results/ubfc_<subject>)
  --bin PATH          PRIVES executable (default: ${PRIVES_BIN})
  --hef PATH          SCRFD .hef model path (default: ${HEF_PATH})
  --realtime          Pace video at 1x native FPS (useful with --stream)
  --stream            Keep MJPEG HTTP server active on port 8080 during execution
  -h, --help          Show this help message
EOF
}

log() {
    echo "[run_ubfc_eval $(date '+%H:%M:%S')] $*"
}

die() {
    echo "[run_ubfc_eval ERROR] $*" >&2
    exit 1
}

while (( $# )); do
    case "$1" in
        --subject-dir) SUBJECT_DIR="$2"; shift 2 ;;
        --video)       VIDEO_PATH="$2"; shift 2 ;;
        --gt)          GT_PATH="$2"; shift 2 ;;
        --subject)     SUBJECT_NAME="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --bin)         PRIVES_BIN="$2"; shift 2 ;;
        --hef)         HEF_PATH="$2"; shift 2 ;;
        --realtime)    REALTIME=1; shift ;;
        --stream)      ENABLE_STREAM=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             usage; die "Unknown option: $1" ;;
    esac
done

# Auto-discover video and ground truth when --subject-dir is given
if [[ -n "${SUBJECT_DIR}" ]]; then
    [[ -d "${SUBJECT_DIR}" ]] || die "Subject directory not found: ${SUBJECT_DIR}"
    if [[ -z "${VIDEO_PATH}" ]]; then
        for cand in "${SUBJECT_DIR}/vid.avi" "${SUBJECT_DIR}"/*.avi "${SUBJECT_DIR}"/*.mp4; do
            if [[ -f "${cand}" ]]; then
                VIDEO_PATH="${cand}"
                break
            fi
        done
    fi
    if [[ -z "${GT_PATH}" ]]; then
        for cand in "${SUBJECT_DIR}/ground_truth.txt" "${SUBJECT_DIR}/gtdump.xmp"; do
            if [[ -f "${cand}" ]]; then
                GT_PATH="${cand}"
                break
            fi
        done
    fi
    if [[ -z "${SUBJECT_NAME}" ]]; then
        SUBJECT_NAME="$(basename "${SUBJECT_DIR}")"
    fi
fi

[[ -n "${VIDEO_PATH}" ]] || { usage; die "Provide --subject-dir <dir> or --video <vid.avi>."; }
[[ -f "${VIDEO_PATH}" ]] || die "Video file not found: ${VIDEO_PATH}"
[[ -x "${PRIVES_BIN}" ]] || die "PRIVES binary not found or not executable at '${PRIVES_BIN}'. Compile first with: cmake -B build && cmake --build build --parallel"
[[ -f "${HEF_PATH}" ]]   || die "HEF model not found at '${HEF_PATH}'. Place scrfd_2.5g.hef in build/ or pass --hef <path>."

if [[ -z "${SUBJECT_NAME}" ]]; then
    parent_name="$(basename "$(dirname "${VIDEO_PATH}")")"
    if [[ "${parent_name}" != "." && "${parent_name}" != "build" ]]; then
        SUBJECT_NAME="${parent_name}"
    else
        base="$(basename "${VIDEO_PATH}")"
        SUBJECT_NAME="${base%.*}"
    fi
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${RESULTS_ROOT}/ubfc_${SUBJECT_NAME}"
fi
mkdir -p "${OUTPUT_DIR}"

CSV_OUT="${OUTPUT_DIR}/prives_estimations.csv"

PRIVES_ARGS=(
    --video "${VIDEO_PATH}"
    --output "${CSV_OUT}"
    --hef "${HEF_PATH}"
)
(( REALTIME ))      && PRIVES_ARGS+=(--realtime)
(( !ENABLE_STREAM )) && PRIVES_ARGS+=(--no-stream)

log "Running PRIVES on '${VIDEO_PATH}' (subject: ${SUBJECT_NAME})..."
"${PRIVES_BIN}" "${PRIVES_ARGS[@]}"
log "PRIVES finished. Estimations saved to: ${CSV_OUT}"

if [[ -n "${GT_PATH}" ]]; then
    [[ -f "${GT_PATH}" ]] || die "Ground-truth file not found: ${GT_PATH}"
    if python3 -c "import numpy, pandas, matplotlib" >/dev/null 2>&1; then
        log "Running offline evaluation (MAE, SNR, PPG & HR plots)..."
        python3 "${BENCH_DIR}/analysis/evaluate_ubfc.py" \
            --prives-csv "${CSV_OUT}" \
            --gt "${GT_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            --subject "${SUBJECT_NAME}"
    else
        log "WARNING: Python packages (numpy, pandas, matplotlib) are not installed on this machine."
        log "Install them with: sudo apt install -y python3-numpy python3-pandas python3-matplotlib"
        log "Or copy '${CSV_OUT}' to your PC and run:"
        log "  python benchmarks/analysis/evaluate_ubfc.py --prives-csv \"${CSV_OUT}\" --gt \"${GT_PATH}\" --subject \"${SUBJECT_NAME}\""
    fi
else
    log "No --gt ground-truth file provided; skipping Python metric calculation."
    log "To compute MAE, SNR, and PPG plots later, run:"
    log "  python3 benchmarks/analysis/evaluate_ubfc.py --prives-csv \"${CSV_OUT}\" --gt <path/to/ground_truth.txt> --subject \"${SUBJECT_NAME}\""
fi

