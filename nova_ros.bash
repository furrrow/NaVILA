#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_CONFIG="${REPO_ROOT}/config/robot.yaml"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
STEERNAV_ROOT="${STEERNAV_ROOT:-/home/jim/Projects/steernav}"
MODE="normal"
DETACH=false
SESSION_NAME=""
CONTROLLER_DT="0.75"
TIME_LOOK_AHEAD="1.0"
FIX_END=false
LOG_ROOT="${NOVA_ROS_LOG_ROOT:-${REPO_ROOT}/logs/nova_ros}"

usage() {
  echo "Usage: $0 [normal|steered|no-velocity] [--detach] [--session-name NAME] [--dt SECONDS] [--time-look-ahead SECONDS] [--fix-end|--no-fix-end]"
}

if [[ $# -gt 0 && ( "$1" == "normal" || "$1" == "steered" || "$1" == "no-velocity" ) ]]; then
  MODE="$1"
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=true; shift ;;
    --session-name)
      [[ $# -ge 2 ]] || { echo "--session-name requires a value" >&2; exit 2; }
      SESSION_NAME="$2"; shift 2 ;;
    --dt)
      [[ $# -ge 2 ]] || { echo "--dt requires a value" >&2; exit 2; }
      CONTROLLER_DT="$2"; shift 2 ;;
    --time-look-ahead)
      [[ $# -ge 2 ]] || { echo "--time-look-ahead requires a value" >&2; exit 2; }
      TIME_LOOK_AHEAD="$2"; shift 2 ;;
    --time-look-ahead=*) TIME_LOOK_AHEAD="${1#*=}"; shift ;;
    --fix-end) FIX_END=true; shift ;;
    --no-fix-end) FIX_END=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -z "${SESSION_NAME}" || "${SESSION_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "Invalid session name." >&2; exit 2
}
[[ "${CONTROLLER_DT}" =~ ^[0-9]+([.][0-9]+)?$ && "${CONTROLLER_DT}" =~ [1-9] ]] || {
  echo "Controller dt must be a positive number." >&2; exit 2
}
[[ "${TIME_LOOK_AHEAD}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "SteerNav time look-ahead must be a non-negative number." >&2; exit 2
}
echo "SteerNav fix end: ${FIX_END}"
[[ -f "${ROS_SETUP}" ]] || { echo "ROS setup not found: ${ROS_SETUP}" >&2; exit 1; }
[[ -f "${ROBOT_CONFIG}" ]] || { echo "Robot config not found: ${ROBOT_CONFIG}" >&2; exit 1; }
command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed." >&2; exit 1; }
if [[ "${MODE}" != "normal" && ! -f "${STEERNAV_ROOT}/steernav/ros_inference.py" ]]; then
  echo "Set STEERNAV_ROOT to a SteerNav checkout." >&2; exit 1
fi

session_name="${SESSION_NAME:-navila_nova_$(date +%s)_$$}"
if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  exit 1
fi
log_dir="${LOG_ROOT}/${session_name}"
mkdir -p "${log_dir}"
inference_log="${log_dir}/inference.log"
controller_log="${log_dir}/controller.log"
steernav_log="${log_dir}/steernav.log"

run_in_pane() {
  local pane="$1" log_file="$2" command="$3" quoted_log
  printf -v quoted_log '%q' "${log_file}"
  tmux send-keys -t "${pane}" \
    "set -o pipefail; export PYTHONUNBUFFERED=1; ( ${command} ) 2>&1 | tee -a ${quoted_log}" C-m
}

tmux new-session -d -s "${session_name}" -c "${REPO_ROOT}"
inference_pane="$(tmux display-message -p -t "${session_name}:0.0" '#{pane_id}')"
controller_pane="$(tmux split-window -h -P -F '#{pane_id}' -t "${inference_pane}" -c "${REPO_ROOT}")"
tmux set-option -t "${session_name}" mouse on
tmux select-pane -t "${inference_pane}" -T "NaVILA inference"
tmux select-pane -t "${controller_pane}" -T "Waypoint controller"
inference_flags=""
if [[ "${MODE}" == "normal" ]]; then
  inference_flags="--visualize"
fi
run_in_pane "${inference_pane}" "${inference_log}" \
  "source ${ROS_SETUP} && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && uv run python ros_inference.py --robot sim_nova --config ${ROBOT_CONFIG} ${inference_flags} --ros-args -p use_sim_time:=true"

controller_flags="--simulation-episode-control"
if [[ "${MODE}" != "normal" ]]; then
  controller_flags="--steer ${controller_flags}"
  steernav_flags="--time-look-ahead ${TIME_LOOK_AHEAD}"
  if [[ "${FIX_END}" == true ]]; then
    steernav_flags="${steernav_flags} --fix-end"
  fi
  if [[ "${MODE}" == "no-velocity" ]]; then
    steernav_flags="${steernav_flags} --no-velocity"
  fi
  steernav_pane="$(tmux split-window -v -P -F '#{pane_id}' -t "${controller_pane}" -c "${STEERNAV_ROOT}")"
  tmux select-pane -t "${steernav_pane}" -T "SteerNav"
  run_in_pane "${steernav_pane}" "${steernav_log}" \
    "source ${ROS_SETUP} && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && uv run steernav/ros_inference.py --robot sim_nova ${steernav_flags} --ros-args -p use_sim_time:=true"
  tmux select-layout -t "${session_name}:0" tiled
fi

run_in_pane "${controller_pane}" "${controller_log}" \
  "source ${ROS_SETUP} && export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && uv run python custom_utils/custom_utils/kinematic_controller_naive.py --robot sim_nova --config ${ROBOT_CONFIG} --dt ${CONTROLLER_DT} ${controller_flags} --ros-args -p use_sim_time:=true"

tmux select-pane -t "${inference_pane}"
echo "TMUX_SESSION=${session_name}"
echo "LOG_DIR=${log_dir}"
if [[ "${DETACH}" == true ]]; then
  exit 0
elif [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "${session_name}"
else
  tmux attach-session -t "${session_name}"
fi
