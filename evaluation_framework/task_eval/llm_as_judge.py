"""
LLM-as-a-judge: reads prediction JSON from evaluate, scores via prompt templates (call_llm/call_vllm),
writes JSON with judge_label, judge_reason, judge_score and prints summary.
Scoring: correct=1, partial=0.5, wrong=0.
"""

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import sys
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from task_eval.prompt import PROMPT_TEMPLATES
from task_eval.utils import call_model

# Scoring: correct=1, partial=0.5, wrong=0
LABEL_TO_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def get_judge_prompt(category: str, evidence: str, pred: str, gold: str = "") -> str:
    """Get template from prompt.PROMPT_TEMPLATES by category and fill gold/pred/evidence. Cognitive has no gold."""
    template = PROMPT_TEMPLATES.get(category) or PROMPT_TEMPLATES["default"]
    return template.format(gold=gold or "", pred=pred or "", evidence=evidence or "")


def label_to_score(label: str) -> float:
    """Map judge label to score: correct=1, partial=0.5, wrong=0; unknown/empty=0."""
    return LABEL_TO_SCORE.get((label or "").strip().lower(), 0.0)


def _parse_judge_response(raw: str) -> tuple:
    """Parse label and reason from model output; return (label, reason)."""
    label, reason = "", ""
    raw = (raw or "").strip()
    try:
        # Try JSON block first
        m = re.search(r'\{[^{}]*"label"\s*:\s*["\']([^"\']+)["\'][^{}]*"reason"\s*:\s*["\']([^"\']*)["\']', raw, re.DOTALL)
        if m:
            label, reason = m.group(1).strip(), (m.group(2) or "").strip()
        else:
            obj = json.loads(raw)
            label = (obj.get("label") or "").strip()
            reason = (obj.get("reason") or "").strip()
    except Exception:
        if "correct" in raw.lower():
            label = "correct"
        elif "wrong" in raw.lower():
            label = "wrong"
        elif "partial" in raw.lower():
            label = "partial"
        reason = raw[:200] if raw else ""
    return label, reason


def _judge_one_record(record: dict, args) -> dict:
    """Run judge on one record; return record with judge_label and judge_reason."""
    r = dict(record)
    cat = r.get("category") or "default"
    evidence = r.get("evidence", "")
    pred = r.get("prediction", "")
    gold = r.get("ground_truth") or r.get("answer", "") or ""

    prompt = get_judge_prompt(cat, evidence, pred, gold)
    raw = call_model(
        prompt,
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    label, reason = _parse_judge_response(raw)
    score = label_to_score(label)
    r["judge_label"] = label
    r["judge_reason"] = reason
    r["judge_score"] = score
    return r


def _compute_summary(results: list) -> dict:
    """Aggregate scores by sample and category. Returns {total_score, total_samples, by_category: {...}}."""
    total_score = 0.0
    by_cat = defaultdict(lambda: {"score": 0.0, "count": 0})
    for r in results:
        s = float(r.get("judge_score", 0.0))
        total_score += s
        cat = r.get("category") or "default"
        by_cat[cat]["score"] += s
        by_cat[cat]["count"] += 1
    summary = {
        "total_score": round(total_score, 2),
        "total_samples": len(results),
        "max_possible": len(results),
        "overall_avg": round(total_score / len(results), 4) if results else 0.0,
        "by_category": {},
    }
    for cat, v in sorted(by_cat.items()):
        n = v["count"]
        summary["by_category"][cat] = {
            "score": round(v["score"], 2),
            "count": n,
            "avg": round(v["score"] / n, 4) if n else 0.0,
        }
    return summary


def _print_summary(summary: dict) -> None:
    """Print score summary to stdout."""
    print("\n" + "=" * 60)
    print("Judge score summary")
    print("=" * 60)
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Total score:   {summary['total_score']} / {summary['max_possible']}")
    print(f"  Average:       {summary['overall_avg']} (correct=1, partial=0.5, wrong=0)")
    print("-" * 60)
    print("  By category:")
    for cat, v in summary["by_category"].items():
        print(f"    {cat}: score {v['score']} / {v['count']} samples, avg {v['avg']}")
    print("=" * 60 + "\n")


def run_judge(args):
    with open(args.input_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concurrency = max(1, int(args.concurrency))

    if concurrency <= 1:
        results = []
        for r in tqdm(records, desc="Judge"):
            results.append(_judge_one_record(r, args))
    else:
        results = [None] * len(records)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {executor.submit(_judge_one_record, r, args): i for i, r in enumerate(records)}
            for future in tqdm(as_completed(future_to_idx), total=len(records), desc="Judge"):
                idx = future_to_idx[future]
                results[idx] = future.result()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} judged records to {out_path}")

    summary = _compute_summary(results)
    _print_summary(summary)
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge on prediction JSON (v3).")
    parser.add_argument("--input-file", required=True, help="Prediction JSON from evaluate_qa.py")
    parser.add_argument("--out-file", required=True, help="Output JSON with judge_label, judge_reason")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Judge model name")
    parser.add_argument("--backend", type=str, default="call_llm", choices=["call_test", "call_llm", "call_vllm"],
                        help="Backend for judge API")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=512, dest="max_tokens", help="Max tokens for judge")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent judge requests")
    parser.add_argument("--summary-file", type=str, default="",
                        help="Optional: write score summary JSON to this path")
    args = parser.parse_args()
    run_judge(args)
