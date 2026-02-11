# Locomo-Plus

Locomo-Plus extends the [LoCoMo](https://github.com/xxx) dialogue benchmark with a sixth task category **Cognitive**, which evaluates long-context memory and implicit recall: models are tested on whether they can connect a later trigger query to an earlier cue dialogue embedded in multi-session conversations.

## Overview

- **Data pipeline**: Generate cue dialogues and trigger queries (with optional human filtering and similarity-based ranking), then build unified inputs by stitching cue/query into LoCoMo conversations.
- **Unified input**: Merge LoCoMo’s five original categories (multi-hop, temporal, common-sense, single-hop, adversarial) with Cognitive into a single JSON format for evaluation.
- **Evaluation**: Run model predictions on the unified dataset and score them with an LLM-as-judge (correct=1, partial=0.5, wrong=0).

## Repository structure

```
Locomo-Plus/
├── data/                    # Data and unified input construction
│   ├── build_conv.py        # Stitch cue/query into LoCoMo sessions by time_gap
│   ├── unified_input.py     # Merge five LoCoMo categories + Cognitive → unified JSON
│   ├── locomo_plus.json     # Cognitive samples (cue_dialogue, trigger_query, time_gap)
│   ├── locomo10.json        # Original LoCoMo multi-session conversations
│   └── README.md
├── generation_pipeline/     # Data generation (steps 1, 3, 4; steps 2 & 5 are manual)
│   ├── cue_dialogue.py      # Step 1: generate cue dialogues
│   ├── trigger_query.py     # Step 3: generate trigger queries from filtered cues
│   ├── rank.py              # Step 4: similarity ranking and filtering
│   └── README.md
├── evaluation_framework/    # Six-category evaluation
│   ├── scripts/
│   │   ├── env.sh           # DATA_FILE_PATH, OUT_DIR, sources env.local.sh
│   │   ├── env.local.sh     # Local API config (do not commit secrets)
│   │   ├── evaluate.sh      # Run model → predictions JSON
│   │   └── judge.sh         # LLM-as-judge → judged JSON + summary
│   ├── task_eval/           # evaluate_qa.py, llm_as_judge.py, utils, prompts
│   └── output/              # Predictions and judged results (path set in env.sh)
└── README.md
```

## Requirements

- **Generation pipeline**: `openai`, `tqdm`; for `rank.py`: `rank_bm25`, `numpy`, `sentence-transformers`.
- **Evaluation**: Python 3; for `call_llm`, set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) via environment or `evaluation_framework/scripts/env.local.sh`.

## Configuration (no hardcoded secrets)

- **Generation (cue_dialogue, trigger_query)**: Use environment variables `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. No API keys in repo.
- **trigger_query input**: Set `CUE_QUERY_INPUT` to the path of your filtered cue JSON; default is `generation_pipeline/selected_cue_query.json`.
- **rank.py**: Set `RANK_INPUT` for the full cue–query JSON; default is `complete_data_all_models.json` in the script directory. Optionally set `SENTENCE_TRANSFORMER_MPNET` and `SENTENCE_TRANSFORMER_BGE` for local model paths (defaults use HuggingFace IDs).
- **Evaluation**: In `evaluation_framework/scripts/env.sh` set `DATA_FILE_PATH` to your unified input JSON. Copy `env.local.sh` and set `OPENAI_BASE_URL` / `OPENAI_API_KEY` locally; do not commit real credentials.

## Pipeline (quick start)

### A. Data generation (see `generation_pipeline/README.md`)

1. Set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if needed). Run `cue_dialogue.py` → `cue_dialogue_data.json`.
2. **Manual**: Filter cues → produce `selected_cue_query.json`; set `CUE_QUERY_INPUT` if not using default path.
3. Run `trigger_query.py` → `complete_data_all_models.json`.
4. Run `rank.py` (set `RANK_INPUT` if needed) → `evaluated_similarity_results.json`.
5. **Manual**: Final validation → curate `data/locomo_plus.json`.

### B. Unified input

6. Place `locomo10.json` and `locomo_plus.json` in `data/`. Optionally run `build_conv.py` for stitched sessions.
7. From `data/`: `python unified_input.py` → writes unified JSON (e.g. `unified_input_samples_v2.json`; see `INSPECTION_JSON_NAME` in script). Set `SAVE_JSON_FOR_INSPECTION = False` to skip writing.

### C. Evaluation

8. In `evaluation_framework/scripts/env.sh`, set `DATA_FILE_PATH` to your unified JSON (e.g. `data/unified_input_samples.json` or `data/unified_input_samples_v2.json`).
9. Copy `evaluation_framework/scripts/env.local.sh.example` to `env.local.sh` in the same directory and set `OPENAI_BASE_URL` and `OPENAI_API_KEY` for `call_llm`. Do not commit `env.local.sh` (it is in `.gitignore`).
10. From repo root or `evaluation_framework`:

    ```bash
    ./evaluation_framework/scripts/evaluate.sh <model> [backend] [temperature] [concurrency]
    # e.g. ./evaluation_framework/scripts/evaluate.sh gpt-4o-mini call_llm 0.3 4
    ```

    Output: `evaluation_framework/output/unified_predictions.json` (or path from `OUT_DIR`).

11. LLM-as-judge:

    ```bash
    ./evaluation_framework/scripts/judge.sh [input_json] [out_json] [model] [concurrency] [summary_json]
    # e.g. ./evaluation_framework/scripts/judge.sh output/unified_predictions.json output/judged.json gpt-4o-mini 4 output/judge_summary.json
    ```

## Unified input format

Each sample has: `input_prompt`, `trigger`, `evidence`, `category`; optional `answer`, `time_gap`. See the docstring at the top of `data/unified_input.py`.

## Scoring

Judge labels: **correct** = 1, **partial** = 0.5, **wrong** = 0. Summary and category templates live in `evaluation_framework/task_eval/`.

## License and privacy

- Do not commit `env.local.sh` or any file containing real API keys or tokens.
- This repo does not ship credentials; configure locally via environment or `env.local.sh`.
