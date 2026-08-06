#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/samuel/policy_risk_persona_simulator}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/training/youtube_comments_LLM_label.csv}"
LAW_BODY_DIR="${LAW_BODY_DIR:-${PROJECT_DIR}/data/raw/laws/bodies}"
LAW_ORIGINAL_TOP_SECTIONS="${LAW_ORIGINAL_TOP_SECTIONS:-3}"
LAW_ORIGINAL_MAX_CHARS="${LAW_ORIGINAL_MAX_CHARS:-600}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S_KST)_E06_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/youtube_stance_kcelectra_binary_law_single/${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-youtube-stance-kcelectra-binary-law-original-single}"
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-E06}"

export PROJECT_DIR LAW_BODY_DIR LAW_ORIGINAL_TOP_SECTIONS LAW_ORIGINAL_MAX_CHARS
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
  --text-mode comment_context
  --max-length 512
  --epochs 5
  --early-stopping-patience 2
  --train-batch-size 4
  --eval-batch-size 8
  --gradient-accumulation-steps 4
  --learning-rate 2e-5
  --weight-decay 0.01
  --warmup-ratio 0.10
  --class-weighting sqrt_balanced
)

if [[ -n "${EXCLUDE_VIDEO_IDS:-}" ]]; then
  ARGS+=(--exclude-video-ids "${EXCLUDE_VIDEO_IDS}")
fi

exec "${PYTHON_BIN}" \
  scripts/training/train_youtube_stance_kcelectra_binary_law.py \
  "${ARGS[@]}"
