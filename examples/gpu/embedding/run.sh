#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL="${WEPM_MODEL:-$REPO_ROOT/checkpoints/huggingface}"
INPUT="$SCRIPT_DIR/../generation/single_image.json"
OUTPUT_DIR="${WEPM_OUTPUT_DIR:-$SCRIPT_DIR/results}"

mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$REPO_ROOT/inference/infer.py" \
  --model "$MODEL" --task embed --content-json "$INPUT" \
  --embedding-dim 1024 \
  --output "$OUTPUT_DIR/single_image-$(date -u +%Y%m%dT%H%M%SZ).json"
