#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


DATASETS = [
    "arts_and_identity",
    "building_and_sacred_heritage",
    "culinary",
    "historical",
    "landscape",
    "language",
    "living_traditions_and_customs",
    "politics",
]

STRATEGIES = [
    "generate_letter_and_text",
    "generate_option_text",
    "generate_letter",
    "letter_shuffled",
    "option_text_avg",
    "option_text",
    "embed_contextual_option",
    "embed_isolated_option",
    "letter",
]

DATASETS = sorted(DATASETS, key=len, reverse=True)
STRATEGIES = sorted(STRATEGIES, key=len, reverse=True)

FILENAME_RE = re.compile(
    r"^answers_(?P<body>.+)_(?P<timestamp>\d{8}_\d{6})\.json$"
)


def parse_file(path: Path):
    """
    Parses:
      answers_<model>_<dataset>_<strategy>_<timestamp>.json

    Handles underscores in both model names and dataset names.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        return None

    body = m.group("body")
    timestamp = m.group("timestamp")

    strategy = None
    before_strategy = None

    for s in STRATEGIES:
        suffix = "_" + s
        if body.endswith(suffix):
            strategy = s
            before_strategy = body[: -len(suffix)]
            break

    if strategy is None:
        return None

    dataset = None
    model = None

    for d in DATASETS:
        suffix = "_" + d
        if before_strategy.endswith(suffix):
            dataset = d
            model = before_strategy[: -len(suffix)]
            break

    if dataset is None or model is None or not model:
        return None

    return {
        "model": model,
        "dataset": dataset,
        "strategy": strategy,
        "timestamp": timestamp,
        "path": path,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge all-model answer JSON files from multiple output folders. "
            "For duplicate model+dataset+strategy files, keeps only the latest timestamp. "
            "Copied files have timestamps removed from filenames."
        )
    )

    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="Input folders.",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        default="merged_allmodels_answers",
        help="Output directory. Default: merged_allmodels_answers",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done; do not copy files.",
    )

    args = parser.parse_args()

    input_dirs = [Path(d) for d in args.input_dirs]
    outdir = Path(args.outdir)

    for d in input_dirs:
        if not d.is_dir():
            raise SystemExit(f"[ERROR] Input directory does not exist: {d}")

    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    latest = {}
    skipped = []

    for input_dir in input_dirs:
        for path in input_dir.glob("answers_*.json"):
            parsed = parse_file(path)

            if parsed is None:
                skipped.append(path)
                continue

            key = (
                parsed["model"],
                parsed["dataset"],
                parsed["strategy"],
            )

            old = latest.get(key)

            if old is None or parsed["timestamp"] > old["timestamp"]:
                latest[key] = parsed

    copied = 0

    for key, item in sorted(latest.items()):
        model, dataset, strategy = key

        new_name = f"answers_{model}_{dataset}_{strategy}.json"
        dst = outdir / new_name

        if dst.exists() and not args.overwrite and not args.dry_run:
            raise SystemExit(
                f"[ERROR] Output file already exists: {dst}\n"
                f"Use --overwrite if you want to replace existing files."
            )

        if args.dry_run:
            print(f"[DRY-RUN] {item['path']} -> {dst}")
        else:
            shutil.copy2(item["path"], dst)

        copied += 1

    by_model = defaultdict(list)
    by_model_dataset = defaultdict(set)
    by_model_strategy = defaultdict(set)
    by_dataset = defaultdict(set)
    by_strategy = defaultdict(set)

    for model, dataset, strategy in latest:
        by_model[model].append((dataset, strategy))
        by_model_dataset[model].add(dataset)
        by_model_strategy[model].add(strategy)
        by_dataset[dataset].add(model)
        by_strategy[strategy].add(model)

    print()
    print("============================================================")
    print("MERGE SUMMARY")
    print("============================================================")
    print(f"Input dirs              : {', '.join(str(d) for d in input_dirs)}")
    print(f"Output dir              : {outdir}")
    print(f"Dry run                 : {args.dry_run}")
    print(f"Files selected/copied   : {copied}")
    print(f"Unique models           : {len(by_model)}")
    print(f"Unique datasets         : {len(by_dataset)}")
    print(f"Unique strategies       : {len(by_strategy)}")
    print(f"Skipped unmatched files : {len(skipped)}")
    print()

    print("Datasets:")
    for dataset in sorted(by_dataset):
        print(f"  {dataset}")

    print()
    print("Strategies:")
    for strategy in sorted(by_strategy):
        print(f"  {strategy}")

    print()
    print("Per-model summary:")
    print("model\tfiles\tdatasets\tstrategies\tstrategy_names")

    for model in sorted(by_model):
        files = len(by_model[model])
        datasets = sorted(by_model_dataset[model])
        strategies = sorted(by_model_strategy[model])

        print(
            f"{model}\t"
            f"{files}\t"
            f"{len(datasets)}\t"
            f"{len(strategies)}\t"
            f"{','.join(strategies)}"
        )

    if skipped:
        print()
        print("Skipped files that did not match expected pattern:")
        for path in skipped[:50]:
            print(f"  {path}")
        if len(skipped) > 50:
            print(f"  ... and {len(skipped) - 50} more")


if __name__ == "__main__":
    main()
