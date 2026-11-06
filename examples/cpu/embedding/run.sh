#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL="${WEPM_MODEL:-$REPO_ROOT/checkpoints/huggingface}"
INPUT="$SCRIPT_DIR/../generation/single_image.json"
OUTPUT_DIR="${WEPM_OUTPUT_DIR:-$SCRIPT_DIR/results}"

mkdir -p "$OUTPUT_DIR"
python "$REPO_ROOT/inference/infer_cpu.py" \
  --model "$MODEL" --task embed --content-json "$INPUT" \
  --threads "${WEPM_THREADS:-8}" --embedding-dim 1024 \
  --output "$OUTPUT_DIR/single_image-$(date -u +%Y%m%dT%H%M%SZ).json"
