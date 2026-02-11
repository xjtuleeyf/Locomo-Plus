"""
Evaluation framework: unified evaluation over six categories, outputs JSON for judge.

Flow:
  1. Load unified input JSON (each sample: input_prompt, evidence, category, etc.).
  2. For each sample: extract question from input_prompt, call call_test/call_llm/call_vllm for prediction.
  3. Build one record per sample: question_input, evidence, category, ground_truth, prediction, model.
  4. Write all records to --out-file for inspection or downstream judge.
  Default backend=call_test (placeholder with same I/O as real LLM) for pipeline validation.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import sys
# Allow "from task_eval.utils" when run from repo root or evaluation_framework
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from task_eval.utils import (
    load_unified_samples,
    call_model,
    build_output_record,
    extract_question_from_input_prompt,
)


def _process_one_sample(sample, args):
    """Process one sample and return one record (for concurrent or sequential use)."""
    input_prompt = sample.get("input_prompt", "")
    category = sample.get("category", "")
    if category == "Cognitive":
        question_input = (sample.get("trigger") or "").strip() or "Context dialogue (cue awareness)"
    else:
        question_input = extract_question_from_input_prompt(input_prompt)

    if not input_prompt:
        return build_output_record(
            sample,
            prediction="(no input_prompt)",
            model=args.model,
            question_input=question_input,
        )

    prediction = call_model(
        input_prompt,
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        category=category,
    )
    return build_output_record(
        sample, prediction, args.model, question_input=question_input
    )


def evaluate_dataset(args):
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_unified_samples(args.data_file)
    concurrency = max(1, int(args.concurrency))

    if concurrency <= 1:
        results = []
        for sample in tqdm(samples, desc=f"Evaluating {args.model}"):
            results.append(_process_one_sample(sample, args))
    else:
        results = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {executor.submit(_process_one_sample, s, args): i for i, s in enumerate(samples)}
            for future in tqdm(as_completed(future_to_idx), total=len(samples), desc=f"Evaluating {args.model}"):
                idx = future_to_idx[future]
                results[idx] = future.result()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model on unified input, write predictions JSON.")
    parser.add_argument("--data-file", required=True, type=str,
                        help="Path to unified input JSON (e.g. data/unified_input_samples.json)")
    parser.add_argument("--out-file", required=True, type=str,
                        help="Output JSON path (predictions only, no input_prompt)")
    parser.add_argument("--model", type=str, default="mock",
                        help="Model name (used in output)")
    parser.add_argument("--backend", type=str, default="call_test",
                        choices=["call_test", "call_llm", "call_vllm"],
                        help="Backend: call_test (same I/O as LLM, no API), call_llm, call_vllm")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature for call_llm (default: 0.3)")
    parser.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens",
                        help="Max tokens for call_llm (default: 1024)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent requests for call_llm/call_vllm (default: 1)")
    args = parser.parse_args()
    evaluate_dataset(args)
