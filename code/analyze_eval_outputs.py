#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ordered_counter_dict(counter: Counter) -> Dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def compute_winning_margin(label_logprobs: Dict[str, float], answer: str) -> Optional[float]:
    if answer not in label_logprobs:
        return None

    answer_score = label_logprobs[answer]
    other_scores = [score for label, score in label_logprobs.items() if label != answer]
    if not other_scores:
        return None

    return answer_score - max(other_scores)


def summarize_file(path: Path) -> Dict[str, Any]:
    payload = load_payload(path)
    results = payload.get("results", [])

    answer_counts: Counter = Counter()
    gold_counts: Counter = Counter()
    per_lang_total: Counter = Counter()
    per_lang_correct: Counter = Counter()
    label_logprob_buckets: Dict[str, List[float]] = defaultdict(list)
    winning_margins: List[float] = []

    scored_items = 0
    correctness_items = 0

    for item in results:
        answer = item.get("answer")
        if answer is not None:
            answer_counts[str(answer)] += 1

        gold = item.get("gold")
        if gold is not None:
            gold_counts[str(gold)] += 1

        lang = str(item.get("lang", "unknown"))
        if item.get("correct") is not None:
            correctness_items += 1
            per_lang_total[lang] += 1
            if item["correct"] is True:
                per_lang_correct[lang] += 1

        label_logprobs = item.get("label_logprobs")
        if isinstance(label_logprobs, dict) and label_logprobs:
            scored_items += 1
            for label, score in label_logprobs.items():
                label_logprob_buckets[str(label)].append(float(score))

            if answer is not None:
                margin = compute_winning_margin(label_logprobs, str(answer))
                if margin is not None:
                    winning_margins.append(margin)

    per_lang_accuracy = []
    for lang in sorted(per_lang_total):
        total = per_lang_total[lang]
        correct = per_lang_correct[lang]
        accuracy = (correct / total) if total else None
        per_lang_accuracy.append({
            "lang": lang,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
        })

    avg_label_logprobs = {
        label: round(avg, 6)
        for label, avg in (
            (label, mean(scores)) for label, scores in sorted(label_logprob_buckets.items())
        )
        if avg is not None
    }

    overall_correct = sum(per_lang_correct.values())
    overall_total = sum(per_lang_total.values())

    return {
        "file": str(path),
        "target_name": payload.get("target_name"),
        "dataset": payload.get("dataset"),
        "results_count": len(results),
        "answer_distribution": ordered_counter_dict(answer_counts),
        "gold_distribution": ordered_counter_dict(gold_counts),
        "correctness": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": round(overall_correct / overall_total, 4) if overall_total else None,
        },
        "per_language_accuracy": per_lang_accuracy,
        "average_label_logprobs": avg_label_logprobs,
        "average_winning_margin": round(mean(winning_margins), 6) if winning_margins else None,
        "scored_items": scored_items,
        "correctness_items": correctness_items,
    }


def iter_input_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    seen = set()

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(path.glob("*.json"))
        else:
            candidates = [path]

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)

    return files


def main() -> None:
    ap = argparse.ArgumentParser("Summarize evaluation output JSON files")
    ap.add_argument("inputs", nargs="+", help="Evaluation JSON files and/or directories")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = ap.parse_args()

    files = iter_input_files(args.inputs)
    if not files:
        raise SystemExit("No input files found.")

    summaries = [summarize_file(path) for path in files]
    if args.pretty:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(summaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
