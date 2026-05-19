#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_input_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    seen: Set[Path] = set()

    for raw in inputs:
        path = Path(raw)
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)
    return files


def parse_dataset_name(dataset_value: Optional[str]) -> str:
    if not dataset_value:
        return "unknown"
    return Path(dataset_value).stem


def parse_dataset_filter(dataset_arg: str) -> Optional[Set[str]]:
    if dataset_arg == "all":
        return None
    return {part.strip() for part in dataset_arg.split(",") if part.strip()}


def pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (part / total) * 100.0


def sorted_counter(counter: Counter) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))


def format_choice_distribution(counter: Counter, total: int) -> str:
    if total == 0 or not counter:
        return "-"
    parts = []
    for answer, count in sorted_counter(counter):
        parts.append(f"{answer}:{count} ({pct(count, total):.1f}%)")
    return ", ".join(parts)


def summarize_model_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_name = records[0]["target_name"]
    datasets = sorted({record["dataset_name"] for record in records})
    files = [record["file"] for record in records]

    answer_counts: Counter = Counter()
    gold_counts: Counter = Counter()
    per_lang_total: Counter = Counter()
    per_lang_correct: Counter = Counter()
    per_dataset_total: Counter = Counter()
    per_dataset_correct: Counter = Counter()

    total_results = 0
    correctness_items = 0
    scored_items = 0
    winning_margins: List[float] = []

    for record in records:
        dataset_name = record["dataset_name"]
        for item in record["results"]:
            total_results += 1

            answer = item.get("answer")
            if answer is not None:
                answer_counts[str(answer)] += 1

            gold = item.get("gold")
            if gold is not None:
                gold_counts[str(gold)] += 1

            correct = item.get("correct")
            if correct is not None:
                correctness_items += 1
                per_dataset_total[dataset_name] += 1
                lang = str(item.get("lang", "unknown"))
                per_lang_total[lang] += 1
                if correct is True:
                    per_dataset_correct[dataset_name] += 1
                    per_lang_correct[lang] += 1

            label_logprobs = item.get("label_logprobs")
            if isinstance(label_logprobs, dict) and label_logprobs:
                scored_items += 1
                if answer in label_logprobs:
                    answer_score = float(label_logprobs[answer])
                    other_scores = [
                        float(score)
                        for label, score in label_logprobs.items()
                        if label != answer
                    ]
                    if other_scores:
                        winning_margins.append(answer_score - max(other_scores))

    total_correct = sum(per_lang_correct.values())
    total_answered = sum(answer_counts.values())

    per_language_accuracy = [
        {
            "lang": lang,
            "correct": per_lang_correct[lang],
            "total": per_lang_total[lang],
            "accuracy": (per_lang_correct[lang] / per_lang_total[lang]) if per_lang_total[lang] else None,
        }
        for lang in sorted(per_lang_total)
    ]

    per_dataset_accuracy = [
        {
            "dataset": dataset_name,
            "correct": per_dataset_correct[dataset_name],
            "total": per_dataset_total[dataset_name],
            "accuracy": (
                per_dataset_correct[dataset_name] / per_dataset_total[dataset_name]
                if per_dataset_total[dataset_name]
                else None
            ),
        }
        for dataset_name in sorted(per_dataset_total)
    ]

    return {
        "target_name": target_name,
        "datasets": datasets,
        "files": files,
        "files_count": len(files),
        "results_count": total_results,
        "correctness_items": correctness_items,
        "correct": total_correct,
        "accuracy": (total_correct / correctness_items) if correctness_items else None,
        "answer_distribution": answer_counts,
        "gold_distribution": gold_counts,
        "unique_choices_used": len(answer_counts),
        "choice_coverage": (len(answer_counts) / 4.0) if answer_counts else 0.0,
        "dominant_choice": sorted_counter(answer_counts)[0][0] if answer_counts else None,
        "dominant_choice_share": (
            sorted_counter(answer_counts)[0][1] / total_answered if total_answered and answer_counts else None
        ),
        "per_language_accuracy": per_language_accuracy,
        "per_dataset_accuracy": per_dataset_accuracy,
        "average_winning_margin": (
            sum(winning_margins) / len(winning_margins) if winning_margins else None
        ),
        "scored_items": scored_items,
    }


def collect_records(files: List[Path], dataset_filter: Optional[Set[str]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in files:
        payload = load_payload(path)
        dataset_name = parse_dataset_name(payload.get("dataset"))
        if dataset_filter is not None and dataset_name not in dataset_filter:
            continue
        records.append({
            "file": str(path),
            "target_name": payload.get("target_name", path.stem),
            "dataset_name": dataset_name,
            "results": payload.get("results", []),
        })
    return records


def print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Rank':<4} {'Model':<36} {'Acc':>7} {'Correct':>8} {'Files':>5} "
        f"{'Datasets':>8} {'Choices':>7} {'TopChoice':>12} {'Margin':>8}"
    )
    print(header)
    print("-" * len(header))

    for idx, summary in enumerate(summaries, start=1):
        accuracy = f"{summary['accuracy'] * 100:6.2f}%" if summary["accuracy"] is not None else "   n/a "
        correct = f"{summary['correct']}/{summary['correctness_items']}"
        top_choice = summary["dominant_choice"] or "-"
        if summary["dominant_choice_share"] is not None:
            top_choice = f"{top_choice} {summary['dominant_choice_share'] * 100:4.1f}%"
        margin = f"{summary['average_winning_margin']:.3f}" if summary["average_winning_margin"] is not None else "-"
        print(
            f"{idx:<4} "
            f"{summary['target_name'][:36]:<36} "
            f"{accuracy:>7} "
            f"{correct:>8} "
            f"{summary['files_count']:>5} "
            f"{len(summary['datasets']):>8} "
            f"{summary['unique_choices_used']:>7} "
            f"{top_choice:>12} "
            f"{margin:>8}"
        )


def print_long_details(summaries: List[Dict[str, Any]]) -> None:
    for summary in summaries:
        print()
        print(summary["target_name"])
        print(f"  datasets: {', '.join(summary['datasets'])}")
        print(
            f"  accuracy: {summary['correct']}/{summary['correctness_items']}"
            + (
                f" ({summary['accuracy'] * 100:.2f}%)"
                if summary["accuracy"] is not None
                else ""
            )
        )
        print(
            f"  choices used: {summary['unique_choices_used']}/4"
            f" ({summary['choice_coverage'] * 100:.1f}%)"
        )
        print(
            "  answer distribution: "
            + format_choice_distribution(summary["answer_distribution"], sum(summary["answer_distribution"].values()))
        )
        print(
            "  gold distribution: "
            + format_choice_distribution(summary["gold_distribution"], sum(summary["gold_distribution"].values()))
        )
        if summary["average_winning_margin"] is not None:
            print(f"  average winning margin: {summary['average_winning_margin']:.6f}")

        if summary["per_dataset_accuracy"]:
            print("  per-dataset accuracy:")
            for row in summary["per_dataset_accuracy"]:
                acc = f"{row['accuracy'] * 100:.2f}%" if row["accuracy"] is not None else "n/a"
                print(f"    - {row['dataset']}: {row['correct']}/{row['total']} ({acc})")

        if summary["per_language_accuracy"]:
            print("  per-language accuracy:")
            for row in summary["per_language_accuracy"]:
                acc = f"{row['accuracy'] * 100:.2f}%" if row["accuracy"] is not None else "n/a"
                print(f"    - {row['lang']}: {row['correct']}/{row['total']} ({acc})")


def to_jsonable(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_name": summary["target_name"],
        "datasets": summary["datasets"],
        "files": summary["files"],
        "files_count": summary["files_count"],
        "results_count": summary["results_count"],
        "correctness_items": summary["correctness_items"],
        "correct": summary["correct"],
        "accuracy": round(summary["accuracy"], 6) if summary["accuracy"] is not None else None,
        "answer_distribution": dict(summary["answer_distribution"]),
        "gold_distribution": dict(summary["gold_distribution"]),
        "unique_choices_used": summary["unique_choices_used"],
        "choice_coverage": round(summary["choice_coverage"], 6),
        "dominant_choice": summary["dominant_choice"],
        "dominant_choice_share": round(summary["dominant_choice_share"], 6)
        if summary["dominant_choice_share"] is not None
        else None,
        "per_language_accuracy": [
            {
                **row,
                "accuracy": round(row["accuracy"], 6) if row["accuracy"] is not None else None,
            }
            for row in summary["per_language_accuracy"]
        ],
        "per_dataset_accuracy": [
            {
                **row,
                "accuracy": round(row["accuracy"], 6) if row["accuracy"] is not None else None,
            }
            for row in summary["per_dataset_accuracy"]
        ],
        "average_winning_margin": round(summary["average_winning_margin"], 6)
        if summary["average_winning_margin"] is not None
        else None,
        "scored_items": summary["scored_items"],
    }


def main() -> None:
    ap = argparse.ArgumentParser("Rank evaluation output files by model accuracy")
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Evaluation JSON files and/or directories",
    )
    ap.add_argument(
        "--dataset",
        default="all",
        help="Dataset filter: all, one dataset stem, or comma-separated stems",
    )
    ap.add_argument(
        "--long",
        action="store_true",
        help="Show detailed answer distribution percentages and per-dataset/per-language breakdowns",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text table",
    )
    args = ap.parse_args()

    dataset_filter = parse_dataset_filter(args.dataset)
    files = iter_input_files(args.input)
    if not files:
        raise SystemExit("No input files found.")

    records = collect_records(files, dataset_filter)
    if not records:
        raise SystemExit("No matching evaluation files after dataset filtering.")

    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[record["target_name"]].append(record)

    summaries = [summarize_model_records(model_records) for model_records in by_model.values()]
    summaries.sort(
        key=lambda row: (
            row["accuracy"] if row["accuracy"] is not None else -1.0,
            row["correct"],
            row["files_count"],
            row["target_name"],
        ),
        reverse=True,
    )

    if args.json:
        print(json.dumps([to_jsonable(summary) for summary in summaries], indent=2, ensure_ascii=False))
        return

    print_summary_table(summaries)
    if args.long:
        print_long_details(summaries)


if __name__ == "__main__":
    main()
