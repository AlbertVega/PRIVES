#!/usr/bin/env bash
# PRIVES stress & benchmarking orchestrator (Raspberry Pi 5 + Hailo-8).
#
# Runs, unattended and in sequence:
#   L0  Heap profiling (check_leaks.sh, 5 min)          [optional]
#   E0  Idle baseline                         (10 min)
#   E1  Nominal soak test                     (60 min)
#   E2  Dynamic scene / tracking re-init      (15 min)
#   E3  LPDDR4X bus contention with stress-ng (10 min)
#
# Every artefact is written under benchmarks/results/run_<timestamp>/.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BENCH_DIR="$(dirname "${SCRIPT_DIR}")"
readonly REPO_ROOT="$(dirname "${BENCH_DIR}")"
readonly RESULTS_ROOT="${BENCH_DIR}/results"

# ----------------------------------------------------------------------------
# Defaults (overridable through environment variables or CLI flags)
# ----------------------------------------------------------------------------
PRIVES_BIN="${PRIVES_BIN:-${REPO_ROOT}/build/prives}"
PRIVES_WORKDIR="${PRIVES_WORKDIR:-}"
SCENARIOS="${SCENARIOS:-E0,E1,E2,E3}"
DURATION_SCALE="${DURATION_SCALE:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.5}"
OCCLUSION_PERIOD_S="${OCCLUSION_PERIOD_S:-5}"
STRESS_VM_BYTES="${STRESS_VM_BYTES:-1G}"
COOLDOWN_TEMP_C="${COOLDOWN_TEMP_C:-50}"
COOLDOWN_MAX_S="${COOLDOWN_MAX_S:-600}"
WARMUP_S="${WARMUP_S:-10}"
RUN_LEAKS=1
LEAK_TOOL="auto"
RUN_REPORT=1
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"

declare -A SCENARIO_DIR=(
    [E0]="E0_idle"
    [E1]="E1_soak"
    [E2]="E2_dynamic_scene"
    [E3]="E3_memory_contention"
)
declare -A SCENARIO_MINUTES=([E0]=10 [E1]=60 [E2]=15 [E3]=10)
readonly LEAK_MINUTES=5

PRIVES_PID=""
MONITOR_PID=""
STRESS_PID=""
RUN_DIR=""
RUN_LOG=""
declare -a FAILED_SCENARIOS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --bin PATH            PRIVES binary (default: ${PRIVES_BIN})
  --workdir DIR         Working directory for PRIVES, must contain the .hef model
                        (default: directory of the binary)
  --scenarios LIST      Comma-separated subset of E0,E1,E2,E3 (default: ${SCENARIOS})
  --scale FACTOR        Multiply every scenario duration (e.g. 0.05 for a smoke test)
  --interval SECONDS    Hardware sampling period (default: ${SAMPLE_INTERVAL})
  --occlusion SECONDS   Face on/off half-period for E2 (default: ${OCCLUSION_PERIOD_S})
  --cooldown-temp C     Wait until the SoC is below this temperature before each scenario
                        (default: ${COOLDOWN_TEMP_C}; 0 disables)
  --cooldown-max S      Maximum cool-down wait (default: ${COOLDOWN_MAX_S})
  --leak-tool TOOL      auto | heaptrack | massif (default: auto)
  --skip-leaks          Do not run the heap profiling stage
  --no-report           Do not run generate_report.py at the end
  --run-id NAME         Name of the output directory under benchmarks/results/
  -h, --help            Show this help
EOF
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log() {
    local line
    line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${line}" >&2
    if [[ -n "${RUN_LOG}" ]]; then
        echo "${line}" >> "${RUN_LOG}"
    fi
}

die() {
    log "ERROR: $*"
    exit 1
}

scaled_seconds() {
    awk -v m="$1" -v s="${DURATION_SCALE}" 'BEGIN { v = int(m * 60 * s + 0.5); print (v < 1 ? 1 : v) }'
}

has_scenario() {
    [[ ",${SCENARIOS}," == *",$1,"* ]]
}

# Interruptible sleep: traps run as soon as a signal arrives.
wait_seconds() {
    sleep "$1" &
    wait $! || true
}

needs_prives() {
    has_scenario E1 || has_scenario E2 || has_scenario E3 || (( RUN_LEAKS ))
}

read_soc_temp() {
    local out
    if command -v vcgencmd >/dev/null 2>&1 && out="$(vcgencmd measure_temp 2>/dev/null)"; then
        sed -E 's/[^0-9.]*([0-9.]+).*/\1/' <<< "${out}"
    elif [[ -r /sys/class/thermal/thermal_zone0/temp ]]; then
        awk '{ printf "%.1f\n", $1 / 1000 }' /sys/class/thermal/thermal_zone0/temp
    else
        echo ""
    fi
}

stop_process() {
    # stop_process PID NAME TIMEOUT_S
    local pid="$1" name="$2" timeout_s="${3:-15}"
    if [[ -z "${pid}" ]]; then
        return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null || true
        return 0
    fi

    kill -TERM "${pid}" 2>/dev/null || true
    local waited=0
    while kill -0 "${pid}" 2>/dev/null && (( waited < timeout_s * 10 )); do
        sleep 0.1
        (( waited += 1 ))
    done
    if kill -0 "${pid}" 2>/dev/null; then
        log "${name} (PID ${pid}) did not exit after ${timeout_s}s, sending SIGKILL."
        kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_process "${STRESS_PID}" "stress-ng" 5
    stop_process "${PRIVES_PID}" "prives" 15
    stop_process "${MONITOR_PID}" "hw_monitor" 5
    STRESS_PID="" PRIVES_PID="" MONITOR_PID=""
    exit "${status}"
}

on_interrupt() {
    log "Interrupted, stopping child processes."
    exit 130
}

# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------
parse_args() {
    while (( $# )); do
        case "$1" in
            --bin)           PRIVES_BIN="$2"; shift 2 ;;
            --workdir)       PRIVES_WORKDIR="$2"; shift 2 ;;
            --scenarios)     SCENARIOS="${2^^}"; shift 2 ;;
            --scale)         DURATION_SCALE="$2"; shift 2 ;;
            --interval)      SAMPLE_INTERVAL="$2"; shift 2 ;;
            --occlusion)     OCCLUSION_PERIOD_S="$2"; shift 2 ;;
            --cooldown-temp) COOLDOWN_TEMP_C="$2"; shift 2 ;;
            --cooldown-max)  COOLDOWN_MAX_S="$2"; shift 2 ;;
            --leak-tool)     LEAK_TOOL="$2"; shift 2 ;;
            --skip-leaks)    RUN_LEAKS=0; shift ;;
            --no-report)     RUN_REPORT=0; shift ;;
            --run-id)        RUN_ID="$2"; shift 2 ;;
            -h|--help)       usage; exit 0 ;;
            *)               usage; die "Unknown option: $1" ;;
        esac
    done

    local s
    IFS=',' read -ra requested <<< "${SCENARIOS}"
    for s in "${requested[@]}"; do
        [[ -n "${SCENARIO_DIR[${s}]:-}" ]] || die "Unknown scenario '${s}'."
    done
    awk -v s="${DURATION_SCALE}" 'BEGIN { exit !(s > 0) }' || die "--scale must be > 0."
}

preflight() {
    command -v python3 >/dev/null 2>&1 || die "python3 is required."
    python3 -c "import psutil" 2>/dev/null || die "Python package 'psutil' is required."
    command -v vcgencmd >/dev/null 2>&1 || log "WARNING: vcgencmd not found; power/throttle data unavailable."

    if has_scenario E3; then
        command -v stress-ng >/dev/null 2>&1 || die "stress-ng is required for E3 (apt install stress-ng)."
    fi

    if needs_prives; then
        [[ -x "${PRIVES_BIN}" ]] || die "PRIVES binary not found or not executable: ${PRIVES_BIN}"
        PRIVES_BIN="$(cd "$(dirname "${PRIVES_BIN}")" && pwd)/$(basename "${PRIVES_BIN}")"
        PRIVES_WORKDIR="${PRIVES_WORKDIR:-$(dirname "${PRIVES_BIN}")}"
        [[ -d "${PRIVES_WORKDIR}" ]] || die "Working directory not found: ${PRIVES_WORKDIR}"
        if ! compgen -G "${PRIVES_WORKDIR}/*.hef" >/dev/null; then
            log "WARNING: no .hef model found in ${PRIVES_WORKDIR}."
        fi
        if [[ ! -e /dev/video0 ]]; then
            log "WARNING: /dev/video0 not present; PRIVES will fail to open the camera."
        fi
    fi
}

capture_system_info() {
    local out="${RUN_DIR}/system_info.txt"
    {
        echo "run_id=${RUN_ID}"
        echo "date=$(date --iso-8601=seconds)"
        echo "duration_scale=${DURATION_SCALE}"
        echo "scenarios=${SCENARIOS}"
        echo "prives_bin=${PRIVES_BIN}"
        echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "--- uname"; uname -a
        echo "--- os-release"; cat /etc/os-release 2>/dev/null || true
        echo "--- cpu governor"
        cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c || true
        echo "--- memory"; free -m 2>/dev/null || true
        if command -v vcgencmd >/dev/null 2>&1; then
            echo "--- firmware"; vcgencmd version 2>/dev/null || true
            echo "--- arm_freq"; vcgencmd get_config arm_freq 2>/dev/null || true
            echo "--- throttled at start"; vcgencmd get_throttled 2>/dev/null || true
        fi
        if command -v hailortcli >/dev/null 2>&1; then
            echo "--- hailo"; timeout 10 hailortcli fw-control identify 2>&1 || true
        fi
    } > "${out}" 2>&1
}

# ----------------------------------------------------------------------------
# Process control
# ----------------------------------------------------------------------------
cooldown() {
    (( $(awk -v t="${COOLDOWN_TEMP_C}" 'BEGIN { print (t > 0) }') )) || return 0
    local temp elapsed=0
    temp="$(read_soc_temp)"
    [[ -z "${temp}" ]] && return 0
    while awk -v a="${temp}" -v b="${COOLDOWN_TEMP_C}" 'BEGIN { exit !(a > b) }'; do
        if (( elapsed >= COOLDOWN_MAX_S )); then
            log "Cool-down timeout at ${temp} C, continuing."
            return 0
        fi
        if (( elapsed == 0 )); then
            log "Cooling down: SoC ${temp} C > ${COOLDOWN_TEMP_C} C."
        fi
        wait_seconds 10
        (( elapsed += 10 ))
        temp="$(read_soc_temp)"
    done
    log "SoC temperature ${temp} C, starting scenario."
}

start_monitor() {
    # start_monitor OUTPUT_CSV [PID]
    local args=(--output "$1" --interval "${SAMPLE_INTERVAL}")
    if [[ -n "${2:-}" ]]; then
        args+=(--pid "$2")
    else
        args+=(--process-name "$(basename "${PRIVES_BIN}")")
    fi
    python3 "${SCRIPT_DIR}/hw_monitor.py" "${args[@]}" 2>> "$(dirname "$1")/hw_monitor.log" &
    MONITOR_PID=$!
}

start_prives() {
    # start_prives SCENARIO_PATH [ENV=VALUE ...]
    local scenario_path="$1"; shift
    (
        cd "${PRIVES_WORKDIR}"
        exec env HAILO_MONITOR=1 \
            PRIVES_TELEMETRY_CSV="${scenario_path}/pipeline_telemetry.csv" \
            "$@" "${PRIVES_BIN}"
    ) > "${scenario_path}/prives.log" 2>&1 &
    PRIVES_PID=$!
}

# Sleeps for DURATION seconds while checking that PRIVES stays alive.
supervise() {
    local duration="$1" deadline
    deadline=$(( $(date +%s) + duration ))
    while (( $(date +%s) < deadline )); do
        if [[ -n "${PRIVES_PID}" ]] && ! kill -0 "${PRIVES_PID}" 2>/dev/null; then
            log "PRIVES exited prematurely (see prives.log)."
            return 1
        fi
        sleep 1
    done
}

write_scenario_meta() {
    # write_scenario_meta PATH KEY DURATION START END STATUS
    cat > "$1/scenario.env" <<EOF
scenario=$2
duration_s=$3
start_epoch=$4
end_epoch=$5
status=$6
EOF
}

# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------
run_scenario() {
    local key="$1"
    local path="${RUN_DIR}/${SCENARIO_DIR[${key}]}"
    local duration status="ok" start_epoch
    duration="$(scaled_seconds "${SCENARIO_MINUTES[${key}]}")"
    mkdir -p "${path}"

    cooldown
    log "=== ${key} (${SCENARIO_DIR[${key}]}) - ${duration}s ==="
    start_epoch="$(date +%s)"

    case "${key}" in
        E0)
            start_monitor "${path}/hardware_metrics.csv"
            wait_seconds "${duration}"
            ;;
        E1)
            start_prives "${path}"
            start_monitor "${path}/hardware_metrics.csv" "${PRIVES_PID}"
            supervise "${duration}" || status="prives_crashed"
            ;;
        E2)
            start_prives "${path}" "PRIVES_BENCH_OCCLUSION_PERIOD_S=${OCCLUSION_PERIOD_S}"
            start_monitor "${path}/hardware_metrics.csv" "${PRIVES_PID}"
            supervise "${duration}" || status="prives_crashed"
            ;;
        E3)
            start_prives "${path}"
            start_monitor "${path}/hardware_metrics.csv" "${PRIVES_PID}"
            # Let the pipeline reach steady state before injecting contention.
            sleep "${WARMUP_S}"
            stress-ng --vm 1 --vm-bytes "${STRESS_VM_BYTES}" --vm-keep \
                --timeout "${duration}s" --metrics-brief \
                > "${path}/stress-ng.log" 2>&1 &
            STRESS_PID=$!
            supervise "${duration}" || status="prives_crashed"
            stop_process "${STRESS_PID}" "stress-ng" 10
            STRESS_PID=""
            ;;
    esac

    stop_process "${PRIVES_PID}" "prives" 15
    PRIVES_PID=""
    stop_process "${MONITOR_PID}" "hw_monitor" 5
    MONITOR_PID=""

    write_scenario_meta "${path}" "${key}" "${duration}" "${start_epoch}" "$(date +%s)" "${status}"
    if [[ "${status}" != "ok" ]]; then
        FAILED_SCENARIOS+=("${key}")
        log "${key} finished with status: ${status}"
    else
        log "${key} completed."
    fi
}

run_leak_check() {
    local duration
    duration="$(scaled_seconds "${LEAK_MINUTES}")"
    cooldown
    log "=== L0 heap profiling (${LEAK_TOOL}) - ${duration}s ==="
    if ! bash "${SCRIPT_DIR}/check_leaks.sh" \
            --bin "${PRIVES_BIN}" --workdir "${PRIVES_WORKDIR}" \
            --output-dir "${RUN_ID}/L0_heap_profile" \
            --duration "${duration}" --tool "${LEAK_TOOL}" 2>&1 | tee -a "${RUN_LOG}"; then
        log "WARNING: heap profiling stage failed, continuing with the suite."
        FAILED_SCENARIOS+=("L0")
    fi
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
main() {
    parse_args "$@"
    preflight

    RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
    mkdir -p "${RUN_DIR}"
    RUN_LOG="${RUN_DIR}/suite.log"
    trap cleanup EXIT
    trap on_interrupt INT TERM

    log "PRIVES stress suite - results in ${RUN_DIR}"
    capture_system_info

    if (( RUN_LEAKS )); then
        run_leak_check
    fi

    local key
    for key in E0 E1 E2 E3; do
        if has_scenario "${key}"; then
            run_scenario "${key}"
        fi
    done

    ln -sfn "${RUN_ID}" "${RESULTS_ROOT}/latest"

    if (( RUN_REPORT )); then
        log "Generating report."
        python3 "${BENCH_DIR}/analysis/generate_report.py" --run-dir "${RUN_DIR}" 2>&1 \
            | tee -a "${RUN_LOG}" || log "WARNING: report generation failed."
    fi

    if (( ${#FAILED_SCENARIOS[@]} )); then
        log "Suite finished with failures in: ${FAILED_SCENARIOS[*]}"
        exit 1
    fi
    log "Suite finished successfully."
}

main "$@"
