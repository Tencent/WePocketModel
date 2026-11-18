#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL="${WEPM_MODEL:-tencent/WePocketModel1.0-4B}"
OUTPUT_DIR="${WEPM_OUTPUT_DIR:-$SCRIPT_DIR/results}"
GPU_ARGS=()
if [[ -n "${WEPM_MIN_FREE_GIB:-}" ]]; then
  GPU_ARGS=(--min-free-gib "$WEPM_MIN_FREE_GIB")
fi

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$REPO_ROOT/inference/infer.py" \
  --model "$MODEL" --task decision --decision-json "$SCRIPT_DIR/../demo_data/decision/request.json" \
  "${GPU_ARGS[@]}" \
  --output "$OUTPUT_DIR/choice-$(date -u +%Y%m%dT%H%M%SZ).json"
