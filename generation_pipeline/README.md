# Generation pipeline

This folder implements the **automated** stages of the Locomo-Plus data generation pipeline. Steps 2 and 5 are manual and have no code here.

## Pipeline steps

| Step | Stage | Script | Output |
|------|--------|--------|--------|
| 1 | Cue dialogue generation | `cue_dialogue.py` | Cue dialogues (memory anchors) |
| 2 | Human filtering of cues | — | Filtered cue list (e.g. `selected_cue_query.json`) |
| 3 | Trigger query generation | `trigger_query.py` | Cue–query candidate pairs |
| 4 | Ranking and similarity filtering | `rank.py` | Ranked/filtered candidates |
| 5 | Final human validation | — | Final `locomo_plus.json` for `data/` |

### Step 1: Cue dialogue generation

- **Script**: `cue_dialogue.py`
- **Purpose**: Generate short cue dialogues that implicitly introduce user states (preferences, goals, values, constraints).
- **Config**: Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`) in the environment. No credentials in the repo.
- **Output**: `cue_dialogue_data.json` (in the script directory).

### Step 2: Human filtering (manual)

Annotators remove cues that are vague, unrealistic, or too explicit. Retained cues are saved (e.g. as `selected_cue_query.json`). Set `CUE_QUERY_INPUT` to that path if it is not the default `selected_cue_query.json` in this directory.

### Step 3: Trigger query generation

- **Script**: `trigger_query.py`
- **Input**: Filtered cues (default: `selected_cue_query.json`, or `CUE_QUERY_INPUT`).
- **Config**: Same as Step 1 (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`).
- **Output**: `complete_data_all_models.json`.

### Step 4: Ranking and filtering

- **Script**: `rank.py`
- **Input**: Default `complete_data_all_models.json` in this directory, or path in `RANK_INPUT`.
- **Optional**: `SENTENCE_TRANSFORMER_MPNET`, `SENTENCE_TRANSFORMER_BGE` for local embedding models (defaults: HuggingFace IDs).
- **Output**: `evaluated_similarity_results.json` (sorted by combined similarity; low similarity first).

### Step 5: Final human validation (manual)

Annotators check that the trigger query truly depends on the cue, that the dependency is implicit, and that the instance matches the intended evaluation design. The final set is written to `data/locomo_plus.json`.
