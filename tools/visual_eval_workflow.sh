#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/configs/monodetr_detect.yaml"
DEFAULT_EVAL_SCRIPT="${REPO_ROOT}/lib/datasets/dataset-dev-kit/src/eval/evaluation.py"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") check --model-name <name> [--camera-id <id>]
  $(basename "$0") visualize [--config <abs_path>]
  $(basename "$0") evaluate --camera-id <south1|south2> --pred-dir <abs_path> [--scene <all|day|night>] [--use-superclasses]

Examples:
  $(basename "$0") check --model-name warm3d_6class_4cam_1
  $(basename "$0") check --model-name warm3d_6class_4cam_1 --camera-id s110_o
  $(basename "$0") visualize --config "${DEFAULT_CONFIG}"
  $(basename "$0") evaluate --camera-id south1 --pred-dir "${REPO_ROOT}/outputs/<model_name>/outputs/south1"
EOF
}

cmd_check() {
  local model_name=""
  local camera_id=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-name) model_name="$2"; shift 2 ;;
      --camera-id) camera_id="$2"; shift 2 ;;
      *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
  done

  if [[ -z "${model_name}" ]]; then
    echo "--model-name is required."
    exit 1
  fi

  local base_dir="${REPO_ROOT}/outputs/${model_name}/outputs"
  local pred_dir="${base_dir}"
  if [[ -n "${camera_id}" ]]; then
    pred_dir="${base_dir}/${camera_id}"
  fi

  echo "Checking predictions in: ${pred_dir}"
  if [[ ! -d "${pred_dir}" ]]; then
    echo "Directory not found: ${pred_dir}"
    exit 1
  fi

  local txt_count
  txt_count=$(find "${pred_dir}" -maxdepth 1 -type f -name "*.txt" | wc -l | tr -d ' ')
  echo "Found ${txt_count} prediction txt files."
  find "${pred_dir}" -maxdepth 1 -type f -name "*.txt" | head -10
}

cmd_visualize() {
  local config="${DEFAULT_CONFIG}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config) config="$2"; shift 2 ;;
      *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
  done

  if [[ ! -f "${config}" ]]; then
    echo "Config not found: ${config}"
    exit 1
  fi

  echo "Running visualization with: ${config}"
  python "${REPO_ROOT}/tools/detect_image.py" --config "${config}"
}

cmd_evaluate() {
  local camera_id=""
  local pred_dir=""
  local scene="all"
  local use_superclasses="false"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --camera-id) camera_id="$2"; shift 2 ;;
      --pred-dir) pred_dir="$2"; shift 2 ;;
      --scene) scene="$2"; shift 2 ;;
      --use-superclasses) use_superclasses="true"; shift 1 ;;
      *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
  done

  if [[ -z "${camera_id}" || -z "${pred_dir}" ]]; then
    echo "--camera-id and --pred-dir are required."
    exit 1
  fi
  if [[ ! -d "${pred_dir}" ]]; then
    echo "Prediction directory not found: ${pred_dir}"
    exit 1
  fi

  local cmd=(python "${DEFAULT_EVAL_SCRIPT}" --camera_id "${camera_id}" --output_dir "${pred_dir}" --scene "${scene}")
  if [[ "${use_superclasses}" == "true" ]]; then
    cmd+=(--use_superclasses)
  fi

  echo "Running evaluation:"
  printf '  %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local action="$1"
  shift

  case "${action}" in
    check) cmd_check "$@" ;;
    visualize) cmd_visualize "$@" ;;
    evaluate) cmd_evaluate "$@" ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
