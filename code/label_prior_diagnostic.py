#!/usr/bin/env python3
"""Answer-label prior diagnostics for BiaBav canonical letter scoring.

Reads the stored results/allmodels JSON files and computes, for each model,
the marginal predicted-label distribution under the canonical letter strategy.
No model inference is run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ("A", "B", "C", "D")


DISPLAY_NAMES = {
    "EuroLLM-9B-Instruct": "EuroLLM-9B",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "granite-3.0-8b-instruct": "Granite-3.0-8B",
    "occiglot-7b-eu5-instruct": "Occiglot-7B",
    "aya-expanse-8b": "Aya-8B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "OLMo-2-1124-7B-Instruct": "OLMo-2-7B",
    "salamandra-7b-instruct": "Salamandra-7B",
    "gemma-2-9b-it": "Gemma-2-9B",
    "leo-hessianai-7b-chat": "Leo-7B",
    "Teuken-7B-instruct-research-v0.4": "Teuken-7B",
    "Pharia-1-LLM-7B-control-hf": "Pharia-7B",
    "genba-10b-it": "GENBA-10B",
    "LLaMmlein_7B_chat": "LLaMmlein-7B",
}


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def is_canonical_letter(path: Path) -> bool:
    payload = load_json(path)
    config_strategy = payload.get("diagnostic_config", {}).get("mcq_strategy")
    if config_strategy:
        return config_strategy == "letter"
    results = payload.get("results") or []
    if not results:
        return False
    return results[0].get("diagnostics", {}).get("strategy") == "letter"


def load_letter_rows(results_dir: Path) -> dict[str, list[dict]]:
    rows_by_model: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        if not is_canonical_letter(path):
            continue
        payload = load_json(path)
        rows_by_model[payload["target_name"]].extend(payload["results"])
    return dict(rows_by_model)


def pct(x: float) -> float:
    return 100.0 * x


def label_prior_expected(pred_counts: Counter, gold_counts: Counter, n: int) -> float:
    return sum((pred_counts[label] / n) * (gold_counts[label] / n) for label in LABELS)


def kl_divergence(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(pi / qi, 2) for pi, qi in zip(p, q) if pi > 0 and qi > 0)


def js_distance(p: list[float], q: list[float]) -> float:
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return math.sqrt((kl_divergence(p, m) + kl_divergence(q, m)) / 2.0)


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[indexed[k][0]] = avg_rank
        i = j
    return out


def pearson(x: list[float], y: list[float]) -> float:
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return num / (den_x * den_y)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rank(x), rank(y))


def correlations(rows: list[dict]) -> list[tuple[str, float, float, float | None, float | None]]:
    actual = [row["accuracy"] for row in rows]
    predictors = [
        ("Predicted B rate", [row["pred_props"]["B"] for row in rows]),
        ("Predicted C rate", [row["pred_props"]["C"] for row in rows]),
        ("Predicted B+C rate", [row["pred_props"]["B"] + row["pred_props"]["C"] for row in rows]),
        ("Label-prior expected accuracy", [row["expected"] for row in rows]),
        ("JS distance to gold labels", [row["js_distance"] for row in rows]),
    ]
    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception:
        pearsonr = spearmanr = None

    out = []
    for name, pred in predictors:
        if pearsonr is None:
            out.append((name, pearson(pred, actual), spearman(pred, actual), None, None))
        else:
            pr = pearsonr(pred, actual)
            sr = spearmanr(pred, actual)
            out.append((name, float(pr.statistic), float(sr.statistic), float(pr.pvalue), float(sr.pvalue)))
    return out


def fmt_p(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-allmodels", type=Path, default=Path("results/allmodels"))
    args = parser.parse_args()

    rows_by_model = load_letter_rows(args.results_allmodels)
    summaries = []
    gold_counts: Counter | None = None

    for target_name, rows in rows_by_model.items():
        n = len(rows)
        if n != 618:
            raise ValueError(f"{target_name} has {n} canonical letter rows, expected 618")
        model_gold = Counter(row["gold"] for row in rows)
        if gold_counts is None:
            gold_counts = model_gold
        elif model_gold != gold_counts:
            raise ValueError(f"{target_name} has a different gold-label distribution")

        pred_counts = Counter(row["parsed_answer"] for row in rows)
        actual = sum(1 for row in rows if row["correct"]) / n
        expected = label_prior_expected(pred_counts, gold_counts, n)
        pred_props = {label: pred_counts[label] / n for label in LABELS}
        gold_props = {label: gold_counts[label] / n for label in LABELS}
        js = js_distance([pred_props[label] for label in LABELS], [gold_props[label] for label in LABELS])
        summaries.append(
            {
                "target_name": target_name,
                "display": DISPLAY_NAMES.get(target_name, target_name),
                "n": n,
                "accuracy": actual,
                "pred_counts": pred_counts,
                "pred_props": pred_props,
                "gold_props": gold_props,
                "expected": expected,
                "residual": actual - expected,
                "js_distance": js,
            }
        )

    if gold_counts is None:
        raise ValueError("No canonical letter files found")

    summaries.sort(key=lambda row: row["accuracy"], reverse=True)
    n = summaries[0]["n"]

    print("Trivial baselines")
    print(f"uniform random = 25.0")
    for label in LABELS:
        print(f"always-{label} = {pct(gold_counts[label] / n):.1f}")
    print(f"majority-label baseline = always-B = {pct(gold_counts['B'] / n):.1f}")
    print()

    print("Model\tAcc\tPredA\tPredB\tPredC\tPredD\tPredA%\tPredB%\tPredC%\tPredD%\tExpected\tResidual\tJS")
    print(
        "Gold\t--\t"
        + "\t".join(str(gold_counts[label]) for label in LABELS)
        + "\t"
        + "\t".join(f"{pct(gold_counts[label] / n):.1f}" for label in LABELS)
        + "\t--\t--\t--"
    )
    for row in summaries:
        print(
            f"{row['display']}\t{pct(row['accuracy']):.1f}\t"
            + "\t".join(str(row["pred_counts"][label]) for label in LABELS)
            + "\t"
            + "\t".join(f"{pct(row['pred_props'][label]):.1f}" for label in LABELS)
            + f"\t{pct(row['expected']):.1f}\t{pct(row['residual']):.1f}\t{row['js_distance']:.3f}"
        )
    print()

    print("Correlations with actual letter accuracy, n=15")
    print("Predictor\tPearson r\tPearson p\tSpearman rho\tSpearman p")
    for name, pr, sr, pp, sp in correlations(summaries):
        print(f"{name}\t{pr:.2f}\t{fmt_p(pp)}\t{sr:.2f}\t{fmt_p(sp)}")


if __name__ == "__main__":
    main()
