#!/usr/bin/env python3
"""Plot GENBA checkpoint trajectories for BiaBav.

This script generates the three figures used in the paper:

1. Accuracy by language under option_text_avg.
2. Accuracy by category under option_text_avg.
3. Faceted category plots showing all evaluation strategies.
4. Language-specific faceted category plots showing all evaluation strategies.

Example:
    python code/plot_genba_checkpoint_trajectories.py

The script expects checkpoint result JSON files named like:
    results/genba_checkpoints/answers_checkpoint41707_language_option_text_avg.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


STRATEGIES = [
    ("Letter", "letter"),
    ("Letter-shuf.", "letter_shuffled"),
    ("Option-text", "option_text"),
    ("Opt.-avg", "option_text_avg"),
    ("Gen-letter", "generate_letter"),
    ("Gen-option", "generate_option_text"),
    ("Gen-letter+txt", "generate_letter_and_text"),
    ("Sem.-gen", "semantic_embed_generated_answer"),
]

CATEGORIES = [
    ("Politics", "politics"),
    ("Arts & Identity", "arts_and_identity"),
    ("Traditions & Customs", "living_traditions_and_customs"),
    ("Culinary", "culinary"),
    ("Historical", "historical"),
    ("Landscape", "landscape"),
    ("Building & Sacred", "building_and_sacred_heritage"),
    ("Language", "language"),
]

LANGUAGES = [
    ("English", "en"),
    ("German", "de"),
    ("Bavarian", "bar"),
]

COLORS = {
    "letter": "#1f77b4",
    "letter_shuffled": "#d95f02",
    "option_text": "#4c8c2b",
    "option_text_avg": "#7b61a8",
    "generate_letter": "#b8860b",
    "generate_option_text": "#079992",
    "generate_letter_and_text": "#c44569",
    "semantic_embed_generated_answer": "#555555",
    "en": "#1f77b4",
    "de": "#d95f02",
    "bar": "#4c8c2b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/genba_checkpoints"),
        help="Directory containing GENBA checkpoint JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/CheckpointEMNLP/images"),
        help="Directory where figures are written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Output formats to write.",
    )
    return parser.parse_args()


def result_correct(row: dict) -> bool:
    """Return correctness across standard and semantic result schemas."""
    value = row.get("correct")
    if value is None:
        value = row.get("semantic_correct")
    return bool(value)


def checkpoint_sort_key(checkpoint: int) -> tuple[int, int]:
    """Keep checkpoint 41707 after 41500 without string sorting surprises."""
    return (checkpoint, checkpoint)


def read_results(results_dir: Path) -> list[dict]:
    strategy_names = sorted((name for _, name in STRATEGIES), key=len, reverse=True)
    strategy_pattern = "|".join(re.escape(name) for name in strategy_names)
    pattern = re.compile(
        rf"answers_checkpoint(?P<checkpoint>\d+)_(?P<category>.+)_(?P<strategy>{strategy_pattern})\.json$"
    )

    records: list[dict] = []
    for path in results_dir.glob("answers_checkpoint*.json"):
        match = pattern.match(path.name)
        if not match:
            continue

        checkpoint = int(match.group("checkpoint"))
        filename_category = match.group("category")
        strategy = match.group("strategy")

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        for row in data.get("results", []):
            records.append(
                {
                    "checkpoint": checkpoint,
                    "category": row.get("category", filename_category),
                    "strategy": strategy,
                    "lang": row.get("lang"),
                    "correct": result_correct(row),
                }
            )

    if not records:
        raise SystemExit(f"No checkpoint result records found in {results_dir}")
    return records


def build_accuracy_lookup(records: list[dict]) -> dict[tuple[int, str, str, str], float]:
    counts: defaultdict[tuple[int, str, str, str], list[int]] = defaultdict(lambda: [0, 0])

    for row in records:
        checkpoint = row["checkpoint"]
        strategy = row["strategy"]
        correct = int(row["correct"])

        for lang in (row.get("lang"), "*"):
            for category in (row.get("category"), "*"):
                bucket = counts[(checkpoint, lang or "*", category or "*", strategy)]
                bucket[0] += correct
                bucket[1] += 1

    return {
        key: 100.0 * correct / total
        for key, (correct, total) in counts.items()
        if total
    }


def lookup_series(
    accuracy_lookup: dict[tuple[int, str, str, str], float],
    checkpoints: list[int],
    *,
    strategy: str,
    lang: str = "*",
    category: str = "*",
) -> list[float]:
    return [
        accuracy_lookup.get((checkpoint, lang, category, strategy), math.nan)
        for checkpoint in checkpoints
    ]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight", dpi=300)


def style_axis(ax: plt.Axes, y_min: float, y_max: float) -> None:
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 42)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(False)


def plot_language_option_text_avg(
    accuracy_lookup: dict[tuple[int, str, str, str], float],
    checkpoints: list[int],
    out_dir: Path,
    formats: list[str],
) -> None:
    x = [checkpoint / 1000 for checkpoint in checkpoints]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for label, lang in LANGUAGES:
        y = lookup_series(
            accuracy_lookup,
            checkpoints,
            lang=lang,
            strategy="option_text_avg",
        )
        ax.plot(x, y, label=label, linewidth=2.2, color=COLORS[lang])

    style_axis(ax, 15, 65)
    ax.set_title("GENBA option-text-avg trajectory by language", pad=14)
    ax.set_xlabel("Checkpoint (thousands of steps)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold", labelpad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=True)
    fig.tight_layout()
    save_figure(fig, out_dir, "genba_checkpoint_language_option_text_avg", formats)
    plt.close(fig)


def plot_category_option_text_avg(
    accuracy_lookup: dict[tuple[int, str, str, str], float],
    checkpoints: list[int],
    out_dir: Path,
    formats: list[str],
) -> None:
    x = [checkpoint / 1000 for checkpoint in checkpoints]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for label, category in CATEGORIES:
        y = lookup_series(
            accuracy_lookup,
            checkpoints,
            category=category,
            strategy="option_text_avg",
        )
        ax.plot(x, y, label=label, linewidth=2.0)

    style_axis(ax, 0, 70)
    ax.set_title("GENBA option-text-avg trajectory by category", pad=14)
    ax.set_xlabel("Checkpoint (thousands of steps)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold", labelpad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=4, frameon=True)
    fig.tight_layout()
    save_figure(fig, out_dir, "genba_checkpoint_category_option_text_avg", formats)
    plt.close(fig)


def plot_strategy_trajectories(
    accuracy_lookup: dict[tuple[int, str, str, str], float],
    checkpoints: list[int],
    out_dir: Path,
    formats: list[str],
) -> None:
    x = [checkpoint / 1000 for checkpoint in checkpoints]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for strategy_label, strategy in STRATEGIES:
        y = lookup_series(accuracy_lookup, checkpoints, strategy=strategy)
        ax.plot(
            x,
            y,
            label=strategy_label,
            linewidth=1.8,
            color=COLORS[strategy],
        )

    style_axis(ax, 0, 70)
    ax.set_title("GENBA checkpoint trajectories by evaluation strategy", pad=14)
    ax.set_xlabel("Checkpoint (thousands of steps)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold", labelpad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=4, frameon=True)
    fig.tight_layout()
    save_figure(fig, out_dir, "genba_checkpoint_strategy_trajectories", formats)
    plt.close(fig)


def plot_category_strategy_facets(
    accuracy_lookup: dict[tuple[int, str, str, str], float],
    checkpoints: list[int],
    out_dir: Path,
    formats: list[str],
) -> None:
    x = [checkpoint / 1000 for checkpoint in checkpoints]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 8.6), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (category_label, category) in zip(axes, CATEGORIES):
        for strategy_label, strategy in STRATEGIES:
            y = lookup_series(
                accuracy_lookup,
                checkpoints,
                category=category,
                strategy=strategy,
            )
            ax.plot(
                x,
                y,
                label=strategy_label,
                linewidth=1.4,
                color=COLORS[strategy],
            )
        style_axis(ax, 0, 85)
        ax.set_title(category_label, loc="left", fontsize=10, pad=6)

    fig.suptitle("GENBA checkpoint trajectories by category and strategy", y=0.995)
    fig.supxlabel("Checkpoint (thousands of steps)", y=0.035, fontweight="bold")
    fig.supylabel("Accuracy (%)", x=0.012, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(top=0.82, left=0.07, right=0.99, bottom=0.09, hspace=0.32, wspace=0.18)
    save_figure(fig, out_dir, "genba_checkpoint_category_strategy_facets", formats)
    plt.close(fig)


def plot_language_category_strategy_facets(
    accuracy_lookup: dict[tuple[int, str, str, str], float],
    checkpoints: list[int],
    out_dir: Path,
    formats: list[str],
) -> None:
    x = [checkpoint / 1000 for checkpoint in checkpoints]

    for language_label, language in LANGUAGES:
        fig, axes = plt.subplots(2, 4, figsize=(13.5, 8.6), sharex=True, sharey=True)
        axes = axes.ravel()

        for ax, (category_label, category) in zip(axes, CATEGORIES):
            for strategy_label, strategy in STRATEGIES:
                y = lookup_series(
                    accuracy_lookup,
                    checkpoints,
                    lang=language,
                    category=category,
                    strategy=strategy,
                )
                ax.plot(
                    x,
                    y,
                    label=strategy_label,
                    linewidth=1.4,
                    color=COLORS[strategy],
                )
            style_axis(ax, 0, 100)
            ax.set_title(category_label, loc="left", fontsize=10, pad=6)

        fig.suptitle(
            f"GENBA checkpoint trajectories by category and strategy ({language_label})",
            y=0.995,
        )
        fig.supxlabel("Checkpoint (thousands of steps)", y=0.035, fontweight="bold")
        fig.supylabel("Accuracy (%)", x=0.012, fontweight="bold")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=4,
            frameon=False,
        )
        fig.subplots_adjust(
            top=0.82,
            left=0.07,
            right=0.99,
            bottom=0.09,
            hspace=0.32,
            wspace=0.18,
        )
        save_figure(
            fig,
            out_dir,
            f"genba_checkpoint_{language}_category_strategy_facets",
            formats,
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    records = read_results(args.results_dir)
    accuracy_lookup = build_accuracy_lookup(records)
    checkpoints = sorted({row["checkpoint"] for row in records}, key=checkpoint_sort_key)

    plot_language_option_text_avg(accuracy_lookup, checkpoints, args.out_dir, args.formats)
    plot_category_option_text_avg(accuracy_lookup, checkpoints, args.out_dir, args.formats)
    plot_strategy_trajectories(accuracy_lookup, checkpoints, args.out_dir, args.formats)
    plot_category_strategy_facets(accuracy_lookup, checkpoints, args.out_dir, args.formats)
    plot_language_category_strategy_facets(accuracy_lookup, checkpoints, args.out_dir, args.formats)

    print(f"Wrote checkpoint trajectory figures to {args.out_dir}")


if __name__ == "__main__":
    main()
