#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_DIR="DynaNav/configs/icra_scenarios_v4"
BATCH_PREFIX="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
FORWARD_ARGS=()
SCENARIOS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mode MODE] [--inference-method NAME] SCENARIO [SCENARIO ...]

Run run_nova_experiments.bash once for each scenario. A scenario may be a
YAML path or a short name from ${SCENARIO_DIR}, with or without .yaml.

Example:
  $(basename "$0") --mode ablation scenario1_5 scenario2_5 scenario3_5
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --mode|--inference-method)
      if (( $# < 2 )); then
        echo "$1 requires a value." >&2
        usage >&2
        exit 2
      fi
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --mode=*|--inference-method=*)
      FORWARD_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SCENARIOS+=("$1")
      shift
      ;;
  esac
done

if (( ${#SCENARIOS[@]} == 0 )); then
  echo "Provide at least one scenario." >&2
  usage >&2
  exit 2
fi

if [[ ! -x "${REPO_ROOT}/run_nova_experiments.bash" ]]; then
  echo "Missing executable launcher: ${REPO_ROOT}/run_nova_experiments.bash" >&2
  exit 2
fi

for requested_scenario in "${SCENARIOS[@]}"; do
  scenario_path="${requested_scenario}"
  if [[ "${requested_scenario}" != */* ]]; then
    scenario_name="${requested_scenario%.yaml}.yaml"
    scenario_path="${SCENARIO_DIR}/${scenario_name}"
  fi

  if [[ ! "${scenario_path}" =~ ^[A-Za-z0-9_./-]+\.yaml$ ]]; then
    echo "Invalid relative YAML scenario path: ${scenario_path}" >&2
    exit 2
  fi

  scenario_id="$(basename "${scenario_path}" .yaml)"
  if [[ ! "${scenario_id}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Scenario filename cannot be used in a run ID: ${scenario_id}" >&2
    exit 2
  fi

  echo
  echo "### Running ${scenario_path} (batch ${BATCH_PREFIX}_${scenario_id}) ###"
  BATCH_ID="${BATCH_PREFIX}_${scenario_id}" \
    "${REPO_ROOT}/run_nova_experiments.bash" \
    --scenario "${scenario_path}" \
    "${FORWARD_ARGS[@]}"
done

echo
echo "All ${#SCENARIOS[@]} scenarios completed successfully."
