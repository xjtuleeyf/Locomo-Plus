#!/bin/bash
# LLM-as-judge: read prediction JSON, score by category templates in prompt.py, write JSON with judge_label/judge_reason/judge_score and summary.
# Scoring: correct=1, partial=0.5, wrong=0.
# Usage: from repo root or evaluation_framework: ./evaluation_framework/scripts/judge.sh [input_json] [out_json] [model] [concurrency] [summary_json]
# Example: ./evaluation_framework/scripts/judge.sh output/unified_predictions.json output/judged.json gpt-4o-mini 4 output/judge_summary.json
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$FRAMEWORK_DIR"
source scripts/env.sh

INPUT_FILE="${1:-${OUT_DIR}/unified_predictions.json}"
OUT_FILE="${2:-${OUT_DIR}/judged.json}"
MODEL="${3:-gpt-4o-mini}"
CONCURRENCY="${4:-1}"
SUMMARY_FILE="${5:-}"

echo "Input: $INPUT_FILE | Output: $OUT_FILE | Model: $MODEL | Concurrency: $CONCURRENCY"

EXTRA=()
if [[ -n "$SUMMARY_FILE" ]]; then
  EXTRA=(--summary-file "$SUMMARY_FILE")
fi

python3 task_eval/llm_as_judge.py \
    --input-file "$INPUT_FILE" \
    --out-file "$OUT_FILE" \
    --model "$MODEL" \
    --backend call_llm \
    --concurrency "$CONCURRENCY" \
    "${EXTRA[@]}"

echo "Done. Judged output written to $OUT_FILE (score: correct=1, partial=0.5, wrong=0)"
