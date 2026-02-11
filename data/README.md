# Data

This directory holds data and scripts for building **unified input** (LoCoMo five categories + Cognitive).

## Scripts

- **build_conv.py**  
  Builds stitched dialogue contexts. For each sample in `locomo_plus.json` it:
  1. Randomly selects a LoCoMo conversation from `locomo10.json`
  2. Uses its session-level time axis
  3. Computes cue/query insertion times from `time_gap`
  4. Maps A/B in cue/query to the conversation speakers
  5. Inserts cue and query as new sessions
  6. Outputs a session-level dialogue context  

  You can post-process the stitched sessions for your final model input.

- **unified_input.py**  
  Merges LoCoMo (five categories) and Locomo-Plus (Cognitive) into one dataset.  
  - LoCoMo: conversation as-is + question (no insertion).  
  - Cognitive: same stitching logic as `build_conv` (map_speaker, time_gap, insert cue/query); no reference answer.  
  Writes a JSON of samples with `input_prompt`, `trigger`, `evidence`, `category`, optional `answer`/`time_gap`. Set `SAVE_JSON_FOR_INSPECTION` and `INSPECTION_JSON_NAME` as needed.

## Data files

- **locomo_plus.json**  
  Locomo-Plus (Cognitive) samples. Each entry: `cue_dialogue`, `trigger_query`, `time_gap`.

- **locomo10.json**  
  Original LoCoMo conversations: multiple sessions per conversation, each with a session timestamp and list of utterances; `speaker_a` and `speaker_b` identify speakers.

- **stitched_contexts.json** (optional)  
  Output of `build_conv.py`: session-level dialogue context including original sessions, inserted cue and query sessions, and metadata (e.g. cue_time, query_time, anchor utterances).
