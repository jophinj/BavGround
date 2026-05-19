#!/usr/bin/env python3
"""Bootstrap confidence intervals for BiaBav result comparisons.

This script reads stored per-item JSON outputs from results_allmodels and
computes simple cluster/paired bootstrap intervals for the comparisons used in
the paper. It does not run model inference.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable


LANGS = {"en", "de", "bar"}
TOP_MODELS = [
    "EuroLLM-9B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B-Instruct",
]


def item_source_id(item_id: str) -> str:
    match = re.match(r"(.+)_(en|de|bar)_(\d+)$", item_id)
    if not match:
        return item_id
    return f"{match.group(1)}_{match.group(3)}"


def item_index(item_id: str) -> int:
    return int(item_id.rsplit("_", 1)[1])


def item_subset(item_id: str) -> str:
    return "GEN" if item_index(item_id) <= 10 else "GRD"


def load_results(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def letter_files(results_dir: Path) -> list[Path]:
    out = []
    for path in results_dir.glob("*.json"):
        name = path.name
        if "_letter" not in name:
            continue
        if "letter_shuffled" in name or "generate_letter" in name:
            continue
        out.append(path)
    return sorted(out)


def strategy_files(results_dir: Path, target_name: str, strategy: str) -> list[Path]:
    out = []
    for path in results_dir.glob(f"*_{strategy}.json"):
        payload = load_results(path)
        if payload.get("target_name") == target_name:
            out.append(path)
    return sorted(out)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("No bootstrap values")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def ci(values: list[float]) -> tuple[float, float]:
    return percentile(values, 0.025), percentile(values, 0.975)


def bootstrap_cluster(
    clusters: dict[str, list[dict]],
    statistic: Callable[[Iterable[dict]], float],
    n_boot: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    keys = sorted(clusters)
    observed = statistic(row for key in keys for row in clusters[key])
    reps = []
    for _ in range(n_boot):
        sampled = [rng.choice(keys) for _ in keys]
        reps.append(statistic(row for key in sampled for row in clusters[key]))
    lo, hi = ci(reps)
    return observed, lo, hi


def accuracy(rows: Iterable[dict]) -> float:
    vals = [1.0 if row["correct"] else 0.0 for row in rows]
    return mean(vals) if vals else float("nan")


def gap_by_filter(
    rows: Iterable[dict],
    left: Callable[[dict], bool],
    right: Callable[[dict], bool],
) -> float:
    rows = list(rows)
    return 100.0 * (accuracy(row for row in rows if left(row)) - accuracy(row for row in rows if right(row)))


def load_open_letter_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in letter_files(results_dir):
        payload = load_results(path)
        target = payload["target_name"]
        for item in payload["results"]:
            rows.append(
                {
                    "target_name": target,
                    "id": item["id"],
                    "source_id": item_source_id(item["id"]),
                    "lang": item["lang"],
                    "subset": item_subset(item["id"]),
                    "correct": bool(item["correct"]),
                }
            )
    return rows


def load_target_strategy_rows(results_dir: Path, target_name: str, strategy: str) -> dict[str, dict]:
    by_id = {}
    for path in strategy_files(results_dir, target_name, strategy):
        payload = load_results(path)
        for item in payload["results"]:
            by_id[item["id"]] = {
                "target_name": target_name,
                "strategy": strategy,
                "id": item["id"],
                "source_id": item_source_id(item["id"]),
                "lang": item["lang"],
                "subset": item_subset(item["id"]),
                "correct": bool(item["correct"]),
            }
    return by_id


def cluster_rows(rows: Iterable[dict], key_field: str = "source_id") -> dict[str, list[dict]]:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        clusters[row[key_field]].append(row)
    return dict(clusters)


def paired_model_gap_rows(rows: list[dict], model_a: str, model_b: str) -> list[dict]:
    by_pair: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row["target_name"] in {model_a, model_b}:
            by_pair[(row["source_id"], row["id"])][row["target_name"]] = row["correct"]
    paired = []
    for (source_id, item_id), vals in by_pair.items():
        if model_a in vals and model_b in vals:
            paired.append(
                {
                    "source_id": source_id,
                    "id": item_id,
                    "diff": (1.0 if vals[model_a] else 0.0) - (1.0 if vals[model_b] else 0.0),
                }
            )
    return paired


def paired_strategy_gap_rows(
    left_by_id: dict[str, dict],
    right_by_id: dict[str, dict],
) -> list[dict]:
    paired = []
    for item_id in sorted(set(left_by_id) & set(right_by_id)):
        left = left_by_id[item_id]
        right = right_by_id[item_id]
        paired.append(
            {
                "source_id": left["source_id"],
                "id": item_id,
                "diff": (1.0 if left["correct"] else 0.0) - (1.0 if right["correct"] else 0.0),
            }
        )
    return paired


def bootstrap_diff_rows(
    rows: list[dict],
    n_boot: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    clusters = cluster_rows(rows)

    def stat(sample_rows: Iterable[dict]) -> float:
        vals = [row["diff"] for row in sample_rows]
        return 100.0 * mean(vals)

    return bootstrap_cluster(clusters, stat, n_boot, rng)


def fmt_triplet(observed: float, lo: float, hi: float) -> str:
    return f"{observed:.1f} [{lo:.1f}, {hi:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-allmodels", type=Path, default=Path("results_allmodels"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    open_letter = load_open_letter_rows(args.results_allmodels)

    outputs = []

    # Aggregate GEN vs GRD, clustered by source question.
    observed, lo, hi = bootstrap_cluster(
        cluster_rows(open_letter),
        lambda rows: gap_by_filter(rows, lambda r: r["subset"] == "GEN", lambda r: r["subset"] == "GRD"),
        args.n_boot,
        rng,
    )
    outputs.append(("Open mean letter: GEN - GRD", observed, lo, hi))

    # Aggregate language gaps, clustered by source question.
    for lang in ["en", "de"]:
        observed, lo, hi = bootstrap_cluster(
            cluster_rows(open_letter),
            lambda rows, lang=lang: gap_by_filter(rows, lambda r: r["lang"] == lang, lambda r: r["lang"] == "bar"),
            args.n_boot,
            rng,
        )
        outputs.append((f"Open mean letter: {lang.upper()} - BAR", observed, lo, hi))

    # GENBA protocol gap.
    genba_letter = load_target_strategy_rows(args.results_allmodels, "genba-10b-it", "letter")
    genba_optavg = load_target_strategy_rows(args.results_allmodels, "genba-10b-it", "option_text_avg")
    observed, lo, hi = bootstrap_diff_rows(
        paired_strategy_gap_rows(genba_optavg, genba_letter),
        args.n_boot,
        rng,
    )
    outputs.append(("GENBA: option_text_avg - letter", observed, lo, hi))

    # Top-three open model letter differences.
    for i, model_a in enumerate(TOP_MODELS):
        for model_b in TOP_MODELS[i + 1 :]:
            observed, lo, hi = bootstrap_diff_rows(
                paired_model_gap_rows(open_letter, model_a, model_b),
                args.n_boot,
                rng,
            )
            short_a = model_a.replace("-Instruct", "").replace("-8B", "").replace("-7B", "")
            short_b = model_b.replace("-Instruct", "").replace("-8B", "").replace("-7B", "")
            outputs.append((f"Letter: {short_a} - {short_b}", observed, lo, hi))

    print("Comparison\tGap pp [95% bootstrap CI]")
    for label, observed, lo, hi in outputs:
        print(f"{label}\t{fmt_triplet(observed, lo, hi)}")


if __name__ == "__main__":
    main()
