#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MODEL="${WEPM_MODEL:-tencent/WePocketModel1.0-4B}"
CASE="${1:-single_text}"
INPUT="$SCRIPT_DIR/../demo_data/generation/$CASE.json"
OUTPUT_DIR="${WEPM_OUTPUT_DIR:-$SCRIPT_DIR/results}"

if [[ ! -f "$INPUT" ]]; then
  echo "Unknown generation case: $CASE" >&2
  echo "Choose single_text, single_image, or interleaved." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
python "$REPO_ROOT/inference/infer_cpu.py" \
  --model "$MODEL" --task generate --content-json "$INPUT" \
  --threads "${WEPM_THREADS:-8}" --max-new-tokens 32 \
  --output "$OUTPUT_DIR/$CASE-$(date -u +%Y%m%dT%H%M%SZ).json"
