#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/samuel/policy_risk_persona_simulator}"
RUNS_ROOT="${RUNS_ROOT:-${PROJECT_DIR}/outputs/youtube_stance_kcelectra_binary_law_single_oversample_s2_o1}"
AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-${RUNS_ROOT}/audit}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-youtube-stance-kcelectra-binary-law-original-single-oversample-s2-o1}"

cd "${PROJECT_DIR}"

ARGS=(
  --runs-root "${RUNS_ROOT}"
  --output-dir "${AUDIT_OUTPUT_DIR}"
  --experiment-name "${EXPERIMENT_NAME}"
  --min-seeds "${MIN_SEEDS:-3}"
  --max-test-macro-f1-std "${MAX_TEST_MACRO_F1_STD:-0.05}"
  --max-test-macro-f1-range "${MAX_TEST_MACRO_F1_RANGE:-0.10}"
)

if [[ "${REQUIRE_MULTISEED:-0}" == "1" ]]; then
  ARGS+=(--require-multiseed)
fi

exec "${PYTHON_BIN}" scripts/training/audit_youtube_stance_runs.py "${ARGS[@]}"
