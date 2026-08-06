#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/samuel/policy_risk_persona_simulator}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/training/youtube_comments_LLM_label.csv}"
LAW_BODY_DIR="${LAW_BODY_DIR:-${PROJECT_DIR}/data/raw/laws/bodies}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S_KST)_E07_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/youtube_stance_kcelectra_binary_law_cv3_e5/${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-youtube-stance-kcelectra-binary-law-original-cv3-e5}"
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-E07}"

cd "${PROJECT_DIR}"

ARGS=(
  --data "${DATA_PATH}"
  --law-body-dir "${LAW_BODY_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --run-id "${RUN_ID}"
  --experiment-name "${EXPERIMENT_NAME}"
  --experiment-note "${EXPERIMENT_NOTE}"
  --model-name "beomi/KcELECTRA-base"
  --seed "${SEED}"
  --train-ratio 0.80
  --dev-ratio 0.10
  --test-ratio 0.10
  --num-folds 3
  --text-mode comment_context
  --context-mode law_original
  --law-original-top-sections 3
  --law-original-max-chars 600
  --max-length 512
  --epochs 5
  --train-batch-size 4
  --eval-batch-size 8
  --gradient-accumulation-steps 4
  --learning-rate 2e-5
  --weight-decay 0.01
  --warmup-ratio 0.10
  --class-weighting sqrt_balanced
  --threshold-min 0.20
  --threshold-max 0.80
  --threshold-step 0.02
)

if [[ -n "${EXCLUDE_VIDEO_IDS:-}" ]]; then
  ARGS+=(--exclude-video-ids "${EXCLUDE_VIDEO_IDS}")
fi

exec "${PYTHON_BIN}" \
  scripts/training/train_youtube_stance_kcelectra_binary_law_cv.py \
  "${ARGS[@]}"
