#!/usr/bin/env bash
# Heap profiling run of the PRIVES binary (default 5 minutes).
#
# Uses heaptrack when available (low overhead, real-time capable) and falls
# back to valgrind --tool=massif. The process is stopped with SIGTERM so the
# pipeline shuts down gracefully and the profiler can flush its data.
#
# Output (relative to benchmarks/results/ unless absolute):
#   <output-dir>/heaptrack.prives.*.{zst,gz} + heaptrack_summary.txt   or
#   <output-dir>/massif.out.prives + massif_summary.txt
#   <output-dir>/leak_report.txt   (condensed verdict)

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BENCH_DIR="$(dirname "${SCRIPT_DIR}")"
readonly REPO_ROOT="$(dirname "${BENCH_DIR}")"
readonly RESULTS_ROOT="${BENCH_DIR}/results"

PRIVES_BIN="${PRIVES_BIN:-${REPO_ROOT}/build/prives}"
PRIVES_WORKDIR="${PRIVES_WORKDIR:-}"
OUTPUT_DIR="heap_profile_$(date +%Y%m%d_%H%M%S)"
DURATION_S=300
TOOL="auto"

TARGET_PID=""
LAUNCHER_PID=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --bin PATH          PRIVES binary (default: ${PRIVES_BIN})
  --workdir DIR       Working directory containing the .hef model (default: binary dir)
  --output-dir DIR    Output directory, relative to benchmarks/results/ (default: ${OUTPUT_DIR})
  --duration SECONDS  Profiling duration (default: ${DURATION_S})
  --tool TOOL         auto | heaptrack | massif (default: ${TOOL})
  -h, --help          Show this help
EOF
}

log() {
    echo "[check_leaks $(date '+%H:%M:%S')] $*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

parse_args() {
    while (( $# )); do
        case "$1" in
            --bin)        PRIVES_BIN="$2"; shift 2 ;;
            --workdir)    PRIVES_WORKDIR="$2"; shift 2 ;;
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            --duration)   DURATION_S="$2"; shift 2 ;;
            --tool)       TOOL="$2"; shift 2 ;;
            -h|--help)    usage; exit 0 ;;
            *)            usage; die "Unknown option: $1" ;;
        esac
    done
    [[ "${DURATION_S}" =~ ^[0-9]+$ ]] && (( DURATION_S > 0 )) || die "--duration must be a positive integer."
}

select_tool() {
    case "${TOOL}" in
        auto)
            if command -v heaptrack >/dev/null 2>&1; then
                TOOL="heaptrack"
            elif command -v valgrind >/dev/null 2>&1; then
                TOOL="massif"
            else
                die "Neither heaptrack nor valgrind is installed (apt install heaptrack valgrind)."
            fi
            ;;
        heaptrack) command -v heaptrack >/dev/null 2>&1 || die "heaptrack not installed." ;;
        massif)    command -v valgrind  >/dev/null 2>&1 || die "valgrind not installed." ;;
        *)         die "Unknown tool '${TOOL}'." ;;
    esac
}

stop_target() {
    local pid="${TARGET_PID:-${LAUNCHER_PID}}"
    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        return 0
    fi
    log "Sending SIGTERM to PID ${pid}."
    kill -TERM "${pid}" 2>/dev/null || true

    # Valgrind needs time to write the massif snapshot; heaptrack to flush.
    local waited=0
    while kill -0 "${LAUNCHER_PID}" 2>/dev/null && (( waited < 1200 )); do
        sleep 0.1
        (( waited += 1 ))
    done
    if kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
        log "Profiler did not exit after 120 s, sending SIGKILL."
        kill -KILL "${pid}" "${LAUNCHER_PID}" 2>/dev/null || true
    fi
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_target
    wait "${LAUNCHER_PID}" 2>/dev/null || true
    exit "${status}"
}

# heaptrack is a wrapper script: the real PRIVES process is one of its descendants.
find_heaptrack_target() {
    local name pid
    name="$(basename "${PRIVES_BIN}")"
    for _ in $(seq 1 100); do
        pid="$(pgrep -n -x "${name}" || true)"
        if [[ -n "${pid}" ]]; then
            echo "${pid}"
            return 0
        fi
        sleep 0.1
    done
    return 1
}

run_heaptrack() {
    local out_dir="$1"
    (
        cd "${PRIVES_WORKDIR}"
        exec heaptrack -o "${out_dir}/heaptrack.prives" "${PRIVES_BIN}"
    ) > "${out_dir}/prives.log" 2>&1 &
    LAUNCHER_PID=$!
    TARGET_PID="$(find_heaptrack_target)" || die "Could not find the PRIVES process under heaptrack."
    log "heaptrack running (PRIVES PID ${TARGET_PID}) for ${DURATION_S}s."

    sleep "${DURATION_S}" & wait $! || true
    stop_target
    wait "${LAUNCHER_PID}" 2>/dev/null || true

    local data
    data="$(ls -t "${out_dir}"/heaptrack.prives*.{zst,gz} 2>/dev/null | head -n 1 || true)"
    [[ -n "${data}" ]] || die "heaptrack produced no data file."
    log "Data file: ${data}"

    if command -v heaptrack_print >/dev/null 2>&1; then
        heaptrack_print --print-leaks 1 "${data}" > "${out_dir}/heaptrack_summary.txt" 2>&1 || true
    else
        heaptrack --analyze "${data}" > "${out_dir}/heaptrack_summary.txt" 2>&1 || true
    fi

    {
        echo "tool=heaptrack"
        echo "duration_s=${DURATION_S}"
        echo "data_file=$(basename "${data}")"
        grep -Ei "total runtime|calls to allocation|temporary memory|peak heap memory|peak RSS|total memory leaked|suppressed leaks" \
            "${out_dir}/heaptrack_summary.txt" || echo "summary lines not found (see heaptrack_summary.txt)"
    } > "${out_dir}/leak_report.txt"
}

run_massif() {
    local out_dir="$1"
    local massif_out="${out_dir}/massif.out.prives"
    (
        cd "${PRIVES_WORKDIR}"
        exec valgrind --tool=massif --massif-out-file="${massif_out}" \
            --time-unit=ms --detailed-freq=10 --max-snapshots=200 "${PRIVES_BIN}"
    ) > "${out_dir}/prives.log" 2>&1 &
    LAUNCHER_PID=$!
    TARGET_PID="${LAUNCHER_PID}"   # valgrind runs the guest in-process.
    log "valgrind massif running (PID ${TARGET_PID}) for ${DURATION_S}s."
    log "Note: valgrind slows the pipeline down ~20-50x; FPS figures from this run are not representative."

    sleep "${DURATION_S}" & wait $! || true
    stop_target
    wait "${LAUNCHER_PID}" 2>/dev/null || true

    [[ -s "${massif_out}" ]] || die "massif produced no output."
    if command -v ms_print >/dev/null 2>&1; then
        ms_print "${massif_out}" > "${out_dir}/massif_summary.txt" 2>&1 || true
    fi

    # Heap growth between the first and second half of the run hints at leaks.
    awk -v dur="${DURATION_S}" '
        /^time=/           { t = substr($0, 6) + 0 }
        /^mem_heap_B=/     { h = substr($0, 12) + 0 }
        /^mem_heap_extra_B=/ {
            total = h + substr($0, 18)
            if (total > peak) peak = total
            n++; last = total
            if (n == 1) first = total
        }
        END {
            printf "tool=massif\nduration_s=%s\nsnapshots=%d\n", dur, n
            printf "first_heap_mb=%.2f\nlast_heap_mb=%.2f\npeak_heap_mb=%.2f\n", first/1048576, last/1048576, peak/1048576
            printf "growth_mb=%.2f\n", (last - first)/1048576
        }' "${massif_out}" > "${out_dir}/leak_report.txt"
}

main() {
    parse_args "$@"
    select_tool

    [[ -x "${PRIVES_BIN}" ]] || die "PRIVES binary not found: ${PRIVES_BIN}"
    PRIVES_BIN="$(cd "$(dirname "${PRIVES_BIN}")" && pwd)/$(basename "${PRIVES_BIN}")"
    PRIVES_WORKDIR="${PRIVES_WORKDIR:-$(dirname "${PRIVES_BIN}")}"

    local out_dir="${OUTPUT_DIR}"
    [[ "${out_dir}" = /* ]] || out_dir="${RESULTS_ROOT}/${out_dir}"
    mkdir -p "${out_dir}"
    out_dir="$(cd "${out_dir}" && pwd)"

    trap cleanup EXIT
    trap 'exit 130' INT TERM

    log "Heap profiling with ${TOOL} -> ${out_dir}"
    case "${TOOL}" in
        heaptrack) run_heaptrack "${out_dir}" ;;
        massif)    run_massif "${out_dir}" ;;
    esac

    log "Leak report:"
    sed 's/^/    /' "${out_dir}/leak_report.txt" >&2
}

main "$@"
