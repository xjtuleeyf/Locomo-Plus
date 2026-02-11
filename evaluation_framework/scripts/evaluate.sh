#!/bin/bash
# Run model on unified input (six categories), write predictions to fixed JSON.
# Usage: from repo root or evaluation_framework directory.
#   ./scripts/evaluate.sh                         # mock + call_test, concurrency=1
#   ./scripts/evaluate.sh gpt-5-nano               # gpt-5-nano + call_llm, temp=0.3
#   ./scripts/evaluate.sh gpt-5-nano call_llm 0.3 4   # temp=0.3, concurrency=4
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(dirname "$SCRIPT_DIR")"
cd "$FRAMEWORK_DIR"
source scripts/env.sh

MODEL="${1:-mock}"
BACKEND="${2:-call_test}"
TEMP="${3:-0.3}"
CONCURRENCY="${4:-1}"
OUT_FILE="${OUT_DIR}/${QA_OUTPUT_FILE}"

# If model is not mock and backend not set, use call_llm when using a real model name
if [[ "$BACKEND" == "call_test" && "$MODEL" != "mock" ]]; then
    BACKEND="call_llm"
fi

echo "Data file: $DATA_FILE_PATH"
echo "Output file: $OUT_FILE"
echo "Model: $MODEL | Backend: $BACKEND | Temperature: $TEMP | Concurrency: $CONCURRENCY"

python3 task_eval/evaluate_qa.py \
    --data-file "$DATA_FILE_PATH" \
    --out-file "$OUT_FILE" \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --temperature "$TEMP" \
    --concurrency "$CONCURRENCY"

echo "Done. Predictions written to $OUT_FILE"
