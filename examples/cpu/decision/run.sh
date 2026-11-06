#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL="${WEPM_MODEL:-$REPO_ROOT/checkpoints/huggingface}"
OUTPUT_DIR="${WEPM_OUTPUT_DIR:-$SCRIPT_DIR/results}"

mkdir -p "$OUTPUT_DIR"
python "$REPO_ROOT/inference/infer_cpu.py" \
  --model "$MODEL" --task decision --decision-json "$SCRIPT_DIR/../demo_data/decision/request.json" \
  --threads "${WEPM_THREADS:-8}" \
  --output "$OUTPUT_DIR/choice-$(date -u +%Y%m%dT%H%M%SZ).json"
