#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${PROJECT_ROOT}"

# Choose 32 or 64 to match the downloaded StereoBind checkpoint variant.
SUMMARY_TOKEN_NUM="${SUMMARY_TOKEN_NUM:-32}"

# Model paths: these defaults match the README download commands.
DISTILLED_CHECKPOINT_PATH="${PROJECT_ROOT}/models/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
SPATIAL_UPSAMPLER_PATH="${PROJECT_ROOT}/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="${PROJECT_ROOT}/models/gemma-3-12b-it-qat-q4_0-unquantized"
CHECKPOINT_DIR="${PROJECT_ROOT}/models/StereoBindCheckpoint/StereoBindCheckpointToken${SUMMARY_TOKEN_NUM}"

# Replace your_samples.json with an actual JSON file from the downloaded dataset.
BATCH_JSON="${PROJECT_ROOT}/inference_samples/your_samples.json"
OUTPUT_PATH="${PROJECT_ROOT}/outputs"

# Use all three weights from the same variant and training step.
CHECKPOINT_STEP="08000"
LORA_PATH="${CHECKPOINT_DIR}/lora_weights_step_${CHECKPOINT_STEP}.safetensors"
TRACK_CONDITIONING_WEIGHTS="${CHECKPOINT_DIR}/track_conditioning_step_${CHECKPOINT_STEP}.safetensors"
SIMPLE_SUMMARY_WEIGHTS="${CHECKPOINT_DIR}/simple_summary_step_${CHECKPOINT_STEP}.safetensors"
LORA_SCALE="1.0"
TRACK_STE_KWARGS='{"hidden_dim":256,"num_layers":2,"num_heads":8,"input_num_frames":121,"dropout":0.0}'
TRACK_ROPE_KWARGS='{"track_rope_ratio":1,"use_x":true,"use_y":true}'

# Fail before loading large models if the local configuration is incomplete.
case "${SUMMARY_TOKEN_NUM}" in
  32|64) ;;
  *) printf 'SUMMARY_TOKEN_NUM must be 32 or 64.\n' >&2; exit 1 ;;
esac
for required_file in \
  "${DISTILLED_CHECKPOINT_PATH}" \
  "${SPATIAL_UPSAMPLER_PATH}" \
  "${LORA_PATH}" \
  "${TRACK_CONDITIONING_WEIGHTS}" \
  "${SIMPLE_SUMMARY_WEIGHTS}" \
  "${BATCH_JSON}"; do
  if [[ ! -f "${required_file}" ]]; then
    printf 'Required file not found: %s\nUpdate run_scripts/inference.sh; see README.md for downloads.\n' "${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${GEMMA_ROOT}" ]]; then
  printf 'Gemma model directory not found: %s\n' "${GEMMA_ROOT}" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  printf 'FFmpeg is required by --save-stage-outputs. Install ffmpeg and add it to PATH.\n' >&2
  exit 1
fi

# Override the GPU selection with: CUDA_VISIBLE_DEVICES=1 bash run_scripts/inference.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m ltx_pipelines.ic_lora_customize_batch_two_stage \
  --distilled-checkpoint-path "${DISTILLED_CHECKPOINT_PATH}" \
  --spatial-upsampler-path "${SPATIAL_UPSAMPLER_PATH}" \
  --gemma-root "${GEMMA_ROOT}" \
  --lora "${LORA_PATH}" "${LORA_SCALE}" \
  --track-injection v2 \
  --use-track-rope \
  --track-rope-kwargs "${TRACK_ROPE_KWARGS}" \
  --track-conditioning-weights "${TRACK_CONDITIONING_WEIGHTS}" \
  --simple-summary-weights "${SIMPLE_SUMMARY_WEIGHTS}" \
  --track-ste-kwargs "${TRACK_STE_KWARGS}" \
  --batch-json "${BATCH_JSON}" \
  --output-path "${OUTPUT_PATH}" \
  --height 768 \
  --width 1280 \
  --num-frames 121 \
  --frame-rate 24 \
  --offload none \
  --trackadaln-version "V3" \
  --track-rope-ratio 1.0 \
  --summary-token-num "${SUMMARY_TOKEN_NUM}" \
  --save-stage-outputs
