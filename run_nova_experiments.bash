#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-isaac}"
REMOTE_REPO="${REMOTE_REPO:-/home/isaac/Projects/TIC-VLA}"
REMOTE_DISPLAY="${REMOTE_DISPLAY:-:1}"
SCENARIO="DynaNav/configs/icra_scenarios_v4/scenario3_5.yaml"
MODE="ablation"
CONTROLLER_DT="0.75"
TIME_LOOK_AHEAD="1.0"
INFERENCE_METHOD="${INFERENCE_METHOD:-NaVILA}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-600}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/logs/orchestration/${BATCH_ID}"
RUN_ATTEMPT_ID="${RUN_ATTEMPT_ID:-$(date +%Y%m%d_%H%M%S)_$$}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mode normal|steered|no-velocity|both|ablation] [--scenario FILE] [--dt SECONDS] [--inference-method NAME] [--time-look-ahead SECONDS]

Run one experiment mode, both primary modes, or an ablation of normal,
steered, and steered without velocity compensation.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --mode) (( $# >= 2 )) || { echo "--mode requires a value." >&2; exit 2; }; MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#*=}"; shift ;;
    -s|--scenario) (( $# >= 2 )) || { echo "$1 requires a file." >&2; exit 2; }; SCENARIO="$2"; shift 2 ;;
    --scenario=*) SCENARIO="${1#*=}"; shift ;;
    --dt) (( $# >= 2 )) || { echo "--dt requires a value." >&2; exit 2; }; CONTROLLER_DT="$2"; shift 2 ;;
    --dt=*) CONTROLLER_DT="${1#*=}"; shift ;;
    --inference-method) (( $# >= 2 )) || { echo "--inference-method requires a value." >&2; exit 2; }; INFERENCE_METHOD="$2"; shift 2 ;;
    --inference-method=*) INFERENCE_METHOD="${1#*=}"; shift ;;
    --time-look-ahead) (( $# >= 2 )) || { echo "--time-look-ahead requires a value." >&2; exit 2; }; TIME_LOOK_AHEAD="$2"; shift 2 ;;
    --time-look-ahead=*) TIME_LOOK_AHEAD="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${MODE}" == "normal" || "${MODE}" == "steered" || "${MODE}" == "no-velocity" || "${MODE}" == "both" || "${MODE}" == "ablation" ]] || {
  echo "Invalid mode: ${MODE}" >&2; exit 2
}
[[ "${SCENARIO}" =~ ^[A-Za-z0-9_./-]+\.yaml$ ]] || { echo "Invalid scenario path." >&2; exit 2; }
[[ "${REMOTE_REPO}" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo "Invalid REMOTE_REPO." >&2; exit 2; }
[[ "${REMOTE_DISPLAY}" =~ ^:[0-9]+(\.[0-9]+)?$ ]] || { echo "Invalid REMOTE_DISPLAY." >&2; exit 2; }
[[ "${BATCH_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid BATCH_ID." >&2; exit 2; }
[[ "${RUN_ATTEMPT_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid RUN_ATTEMPT_ID." >&2; exit 2; }
[[ "${INFERENCE_METHOD}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid inference method." >&2; exit 2; }
[[ "${CONTROLLER_DT}" =~ ^[0-9]+([.][0-9]+)?$ && "${CONTROLLER_DT}" =~ [1-9] ]] || {
  echo "Controller dt must be a positive number." >&2; exit 2
}
[[ "${READY_TIMEOUT_SECONDS}" =~ ^[0-9]+$ && "${READY_TIMEOUT_SECONDS}" -gt 0 ]] || {
  echo "READY_TIMEOUT_SECONDS must be positive." >&2; exit 2
}
[[ "${TIME_LOOK_AHEAD}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "SteerNav time look-ahead must be a non-negative number." >&2; exit 2
}
command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed." >&2; exit 1; }

mkdir -p "${LOG_DIR}"
active_session=""
active_remote_job=""

stop_remote_job() {
  [[ -n "${active_remote_job}" ]] || return 0
  echo "Stopping remote simulator job ${active_remote_job} on ${REMOTE_HOST}..."
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" \
    "cd ${REMOTE_REPO} && ./run_sim.bash --stop ${active_remote_job}" || \
    echo "Warning: remote job stop was not confirmed." >&2
  active_remote_job=""
}

cleanup_session() {
  stop_remote_job
  if [[ -n "${active_session}" ]] && tmux has-session -t "${active_session}" 2>/dev/null; then
    tmux kill-session -t "${active_session}"
  fi
  active_session=""
}
trap cleanup_session EXIT
trap 'exit 130' INT TERM

wait_until_ready() {
  local mode="$1" session="$2" deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  local pane_output pane_id
  echo "Waiting up to ${READY_TIMEOUT_SECONDS}s for local ${mode} nodes to initialize..."
  while (( SECONDS < deadline )); do
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      echo "Local tmux session exited before becoming ready: ${session}" >&2
      return 1
    fi
    pane_output=""
    while IFS= read -r pane_id; do
      pane_output+="$(tmux capture-pane -p -S -200 -t "${pane_id}" 2>/dev/null || true)"$'\n'
    done < <(tmux list-panes -t "${session}:0" -F '#{pane_id}')
    if [[ "${pane_output}" == *"Model ready."* && "${pane_output}" == *"Registered with master node."* ]]; then
      if [[ "${mode}" == "normal" || "${pane_output}" == *"Waiting for simulation episodes"* ]]; then
        echo "Local ${mode} pipeline is ready."
        return 0
      fi
    fi
    sleep 2
  done
  echo "Timed out waiting for local ${mode} pipeline. Recent pane output:" >&2
  while IFS= read -r pane_id; do
    tmux capture-pane -p -S -100 -t "${pane_id}" >&2 || true
  done < <(tmux list-panes -t "${session}:0" -F '#{pane_id}')
  return 1
}

run_mode() {
  local mode="$1" run_id="${BATCH_ID}_$1"
  local session="navila_${run_id//./_}_${RUN_ATTEMPT_ID//./_}"
  local run_log_dir="${LOG_DIR}/${run_id}/${RUN_ATTEMPT_ID}"
  local status_file="${run_log_dir}/.remote.status"
  local remote_command remote_pane watcher_pid remote_status

  echo
  echo "=== ${mode}: run_id=${run_id} ==="
  mkdir -p "${run_log_dir}"
  NOVA_ROS_LOG_ROOT="${run_log_dir}" \
    "${REPO_ROOT}/nova_ros.bash" "${mode}" --detach --session-name "${session}" \
    --dt "${CONTROLLER_DT}" --time-look-ahead "${TIME_LOOK_AHEAD}"
  active_session="${session}"
  wait_until_ready "${mode}" "${session}"

  echo "Starting simulator benchmark on ${REMOTE_HOST} in a visible tmux pane..."
  active_remote_job="${run_id}"
  remote_command="set -o pipefail; ssh -o BatchMode=yes -o ConnectTimeout=10 ${REMOTE_HOST} 'cd ${REMOTE_REPO} && DISPLAY=${REMOTE_DISPLAY} ./run_sim.bash --scenario ${SCENARIO} --job-id ${run_id} --inference-method ${INFERENCE_METHOD} -- --run_id ${run_id} --experiment_mode ${mode}' 2>&1 | tee ${run_log_dir}/remote.log; status=\${PIPESTATUS[0]}; echo \${status} > ${status_file}; exit \${status}"
  remote_pane="$(tmux split-window -v -P -F '#{pane_id}' -t "${session}:0" -c "${REPO_ROOT}" /bin/bash -lc "${remote_command}")"
  tmux select-pane -t "${remote_pane}" -T "Isaac ${mode} status"
  tmux select-layout -t "${session}:0" tiled

  (
    while tmux has-session -t "${session}" 2>/dev/null && [[ ! -s "${status_file}" ]]; do sleep 1; done
    if tmux has-session -t "${session}" 2>/dev/null; then sleep 3; tmux kill-session -t "${session}"; fi
  ) &
  watcher_pid=$!

  echo "Attaching to ${session}; it will close shortly after the remote benchmark exits."
  if [[ -n "${TMUX:-}" ]]; then
    echo "Run this orchestrator outside an existing tmux client." >&2
    tmux kill-session -t "${session}"
    wait "${watcher_pid}" || true
    return 1
  fi
  tmux attach-session -t "${session}" || true
  wait "${watcher_pid}" || true
  active_session=""
  [[ -s "${status_file}" ]] || { echo "The ${mode} tmux session ended before the remote benchmark reported status." >&2; return 1; }
  remote_status="$(<"${status_file}")"
  active_remote_job=""
  if [[ "${remote_status}" != "0" ]]; then
    echo "Remote ${mode} benchmark failed with status ${remote_status}." >&2
    return "${remote_status}"
  fi
  echo "Remote ${mode} benchmark completed."
  sleep 5
}

echo "Batch ID: ${BATCH_ID}"
echo "Run attempt: ${RUN_ATTEMPT_ID}"
echo "Mode: ${MODE}"
echo "Inference method: ${INFERENCE_METHOD}"
echo "SteerNav time look-ahead: ${TIME_LOOK_AHEAD}s"
echo "Scenario: ${SCENARIO}"
echo "Remote simulator: ${REMOTE_HOST}:${REMOTE_REPO}"
echo "Controller dt: ${CONTROLLER_DT}s"
ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" \
  "cd ${REMOTE_REPO} && DISPLAY=${REMOTE_DISPLAY} ./run_sim.bash --check --scenario ${SCENARIO}"

if [[ "${MODE}" == "steered" || "${MODE}" == "both" || "${MODE}" == "ablation" ]]; then run_mode steered; fi
if [[ "${MODE}" == "no-velocity" || "${MODE}" == "ablation" ]]; then run_mode no-velocity; fi
if [[ "${MODE}" == "normal" || "${MODE}" == "both" || "${MODE}" == "ablation" ]]; then run_mode normal; fi

echo
echo "Requested NaVILA experiment mode '${MODE}' completed successfully."
if [[ "${MODE}" == "ablation" ]]; then
  echo "Remote run IDs: ${BATCH_ID}_normal, ${BATCH_ID}_steered, ${BATCH_ID}_no-velocity"
elif [[ "${MODE}" == "both" ]]; then
  echo "Remote run IDs: ${BATCH_ID}_normal, ${BATCH_ID}_steered"
else
  echo "Remote run ID: ${BATCH_ID}_${MODE}"
fi
echo "Local orchestration logs: ${LOG_DIR}"
