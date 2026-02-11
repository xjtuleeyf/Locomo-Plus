# evaluation_framework env
# Path to unified input JSON (from data/unified_input.py output)
DATA_FILE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data/unified_input_samples.json"

# Output directory for prediction JSON (category, evidence, prediction, etc.)
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/evaluation_framework/output"

# Output filename for QA predictions
QA_OUTPUT_FILE=unified_predictions.json

# For --backend call_llm: load from env.local.sh if present (do not commit env.local.sh)
# env.local.sh can set: OPENAI_BASE_URL (required for custom endpoint), OPENAI_API_KEY
if [ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ]; then
  source "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"
fi
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export GOOGLE_API_KEY=
export ANTHROPIC_API_KEY=
export HF_TOKEN=
