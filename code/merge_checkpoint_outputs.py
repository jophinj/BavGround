#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path


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

# longest first, so option_text_avg is matched before option_text
STRATEGIES = sorted(STRATEGIES, key=len, reverse=True)

FILENAME_RE = re.compile(
    r"^answers_checkpoint(?P<checkpoint>\d+)_(?P<middle>.+)_(?P<timestamp>\d{8}_\d{6})\.json$"
)


def parse_file(path: Path):
    """
    Parses:
      answers_checkpoint41707_politics_option_text_avg_20260511_082248.json

    Returns:
      checkpoint = 41707
      category   = politics
      strategy   = option_text_avg
      timestamp  = 20260511_082248
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        return None

    checkpoint = int(m.group("checkpoint"))
    middle = m.group("middle")
    timestamp = m.group("timestamp")

    for strategy in STRATEGIES:
        suffix = "_" + strategy
        if middle.endswith(suffix):
            category = middle[: -len(suffix)]
            return {
                "checkpoint": checkpoint,
                "category": category,
                "strategy": strategy,
                "timestamp": timestamp,
                "path": path,
            }

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge checkpoint answer JSON files from multiple output folders."
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="Input folders, e.g. out_ckpts_multi_strategy_v2_genba_checkpoints out_ckpts_multi_strategy_v2_genba_checkpoints_run2",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="merged_ckpt_answers",
        help="Output directory. Default: merged_ckpt_answers",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files in the output directory.",
    )
    args = parser.parse_args()

    input_dirs = [Path(d) for d in args.input_dirs]
    outdir = Path(args.outdir)

    for d in input_dirs:
        if not d.is_dir():
            raise SystemExit(f"[ERROR] Input directory does not exist: {d}")

    outdir.mkdir(parents=True, exist_ok=True)

    # key = checkpoint, category, strategy
    latest = {}
    skipped = []

    for d in input_dirs:
        for path in d.glob("answers_checkpoint*.json"):
            parsed = parse_file(path)
            if parsed is None:
                skipped.append(path)
                continue

            key = (
                parsed["checkpoint"],
                parsed["category"],
                parsed["strategy"],
            )

            old = latest.get(key)
            if old is None or parsed["timestamp"] > old["timestamp"]:
                latest[key] = parsed

    copied = 0

    for key, item in sorted(latest.items()):
        checkpoint, category, strategy = key

        # timestamp removed here
        new_name = f"answers_checkpoint{checkpoint}_{category}_{strategy}.json"
        dst = outdir / new_name

        if dst.exists() and not args.overwrite:
            raise SystemExit(
                f"[ERROR] Output file already exists: {dst}\n"
                f"Use --overwrite if you want to replace existing files."
            )

        shutil.copy2(item["path"], dst)
        copied += 1

    # Summary
    by_checkpoint = defaultdict(list)
    by_checkpoint_strategy = defaultdict(set)
    by_checkpoint_category = defaultdict(set)

    for checkpoint, category, strategy in latest:
        by_checkpoint[checkpoint].append((category, strategy))
        by_checkpoint_strategy[checkpoint].add(strategy)
        by_checkpoint_category[checkpoint].add(category)

    print()
    print("============================================================")
    print("MERGE SUMMARY")
    print("============================================================")
    print(f"Input dirs              : {', '.join(str(d) for d in input_dirs)}")
    print(f"Output dir              : {outdir}")
    print(f"Copied files            : {copied}")
    print(f"Unique checkpoints      : {len(by_checkpoint)}")
    print(f"Skipped unmatched files : {len(skipped)}")
    print()

    print("Per-checkpoint summary:")
    print("checkpoint\tfiles\tcategories\tstrategies\tstrategy_names")

    for checkpoint in sorted(by_checkpoint):
        files = len(by_checkpoint[checkpoint])
        categories = len(by_checkpoint_category[checkpoint])
        strategies = sorted(by_checkpoint_strategy[checkpoint])

        print(
            f"{checkpoint}\t"
            f"{files}\t"
            f"{categories}\t"
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
