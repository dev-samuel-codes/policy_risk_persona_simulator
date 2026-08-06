#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/samuel/policy_risk_persona_simulator}"
SEEDS="${SEEDS:-7 19 42 73 123}"
SPLIT_SEED="${SPLIT_SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/youtube_stance_kcelectra_binary_law_e08_01_five_seeds}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-youtube-stance-kcelectra-binary-law-e08-01-five-seeds}"

cd "${PROJECT_DIR}"

read -r -a SEED_LIST <<< "${SEEDS}"
seed_index=0
seed_total="${#SEED_LIST[@]}"

for seed in "${SEED_LIST[@]}"; do
  seed_index=$((seed_index + 1))
  echo "[E08-01] ${seed_index}/${seed_total}: training seed=${seed}, fixed split seed=${SPLIT_SEED}"
  SEED="${seed}" \
  SPLIT_SEED="${SPLIT_SEED}" \
  RUN_TAG="E08-01" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  EXPERIMENT_NOTE="E08-01 ${seed_index}/${seed_total}, training seed=${seed}" \
  bash scripts/training/run_youtube_stance_kcelectra_binary_law_e08_ampm.sh
done
