#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/samuel/policy_risk_persona_simulator}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/training/youtube_comments_LLM_label.csv}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S_KST)_E03_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/youtube_stance_kcelectra_base/${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-youtube-stance-kcelectra-base}"
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-E03}"

cd "${PROJECT_DIR}"

ARGS=(
  --data "${DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --experiment-name "${EXPERIMENT_NAME}"
  --experiment-note "${EXPERIMENT_NOTE}"
  --model-name "beomi/KcELECTRA-base"
  --seed "${SEED}"
  --train-ratio 0.80
  --dev-ratio 0.10
  --test-ratio 0.10
  --max-length 256
  --epochs 5
  --train-batch-size 16
  --eval-batch-size 32
  --gradient-accumulation-steps 1
  --learning-rate 2e-5
  --weight-decay 0.01
  --warmup-ratio 0.10
  --early-stopping-patience 2
  --class-weighting sqrt_balanced
)

if [[ -n "${EXCLUDE_VIDEO_IDS:-}" ]]; then
  ARGS+=(--exclude-video-ids "${EXCLUDE_VIDEO_IDS}")
fi

exec "${PYTHON_BIN}" scripts/training/train_youtube_stance_kcelectra.py "${ARGS[@]}"
