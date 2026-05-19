#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


# ------------------------------------------------------------------------------
# Prompts / generation config
# ------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful, precise assistant.\n"
    "Follow the user instruction.\n"
    "Do not add meta-commentary.\n"
    "Do not ask follow-up questions.\n"
    "Finish the answer and stop."
)

GENERATE_CFG = {
    "max_new_tokens": 128,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 3,
}

# For logprob scoring (MCQ/classify): no sampling, no repetition penalty.
# We don't use generate() at all there, but keep this for record/debug.
SCORE_CFG = {
    "do_sample": False,
}

MAX_INPUT_LEN = 1024


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


def build_prompt(user_instr: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n### User:\n{user_instr}\n\n### Assistant:\n"


def clean_response(text: str) -> str:
    for stop in ["\n###", "\nUser:", "\nAssistant:"]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    if not isinstance(items, list):
        raise ValueError("Dataset must be a list OR an object with an 'items' list.")
    return items


def resolve_dataset_paths(dataset_arg: str) -> List[Path]:
    """
    Backward compatible dataset resolver.

    Supported inputs:
    - single JSON file path
    - comma-separated JSON file paths
    - directory containing .json files
    """
    raw_parts = [part.strip() for part in dataset_arg.split(",") if part.strip()]
    if not raw_parts:
        raise SystemExit("No dataset path provided.")

    resolved: List[Path] = []
    seen = set()

    for raw in raw_parts:
        path = Path(raw)

        if path.is_dir():
            for child in sorted(path.glob("*.json")):
                child_resolved = child.resolve()
                if child_resolved not in seen:
                    seen.add(child_resolved)
                    resolved.append(child)
            continue

        if not path.exists():
            raise SystemExit(f"Dataset path does not exist: {path}")
        if not path.is_file():
            raise SystemExit(f"Dataset path is not a file or directory: {path}")

        path_resolved = path.resolve()
        if path_resolved not in seen:
            seen.add(path_resolved)
            resolved.append(path)

    if not resolved:
        raise SystemExit(f"No dataset JSON files found for: {dataset_arg}")

    return resolved


def item_category(item: Dict[str, Any]) -> str:
    if "category" in item:
        return str(item["category"])
    return str(item.get("meta", {}).get("category", ""))


def item_bias(item: Dict[str, Any]) -> str:
    if "bias" in item:
        return str(item["bias"])
    return str(item.get("meta", {}).get("bias", ""))


def filter_items(
    items: List[Dict[str, Any]],
    types: Optional[List[str]] = None,
    langs: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    biases: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    def norm_list(x):
        return [s.strip() for s in x if s.strip()] if x else None

    types = norm_list(types)
    langs = norm_list(langs)
    categories = norm_list(categories)
    biases = norm_list(biases)

    out = []
    for it in items:
        if types and it.get("type") not in types:
            continue
        if langs and it.get("lang") not in langs:
            continue
        cat = item_category(it)
        if categories and cat not in categories:
            continue
        b = item_bias(it)
        if biases and b not in biases:
            continue
        out.append(it)
    return out


def format_instruction(item: Dict[str, Any]) -> str:
    """
    Convert dataset items into one instruction string.

    IMPORTANT:
    - Dataset format is unchanged.
    - We add a stable "Answer:" anchor for MCQ/classify to improve scoring alignment.
    """
    t = item["type"]
    prompt = item["prompt"]

    if t == "mcq":
        opts = item.get("options", {})
        opt_lines = "\n".join([f"{k}: {v}" for k, v in opts.items()])
        return (
            f"{prompt}\n\nOptions:\n{opt_lines}\n\n"
            f"Reply with exactly one character: A, B, C, or D.\n"
            f"Answer:"
        )

    if t == "classify":
        allowed = item.get("allowed", [])
        allowed_str = ", ".join([f"'{x}'" for x in allowed]) if allowed else "a valid label"
        return (
            f"{prompt}\n\n"
            f"Reply with exactly one of: {allowed_str}. No explanation.\n"
            f"Answer:"
        )

    if t == "generate":
        return prompt

    raise ValueError(f"Unknown item type: {t}")


# ------------------------------------------------------------------------------
# Checkpoint / model selection
# ------------------------------------------------------------------------------

def parse_range_spec(spec: str) -> Tuple[int, int, int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError("range spec must be start:end:step e.g. 0:2000:500")
    start, end, step = map(int, parts)
    if step <= 0:
        raise ValueError("step must be > 0")
    return start, end, step


def discover_checkpoints(base: Path, name_regex: str, range_spec: Optional[str]) -> List[Path]:
    rx = re.compile(name_regex)
    found: List[Tuple[int, Path]] = []

    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = rx.match(p.name)
        if not m:
            continue
        step_num = int(m.group(1))
        found.append((step_num, p))

    found.sort(key=lambda x: x[0])

    if range_spec:
        start, end, step = parse_range_spec(range_spec)
        wanted = set(range(start, end + 1, step))
        found = [(n, p) for (n, p) in found if n in wanted]

    return [p for _, p in found]


def resolve_hf_dir(checkpoint_dir: Path) -> Path:
    candidates = list(checkpoint_dir.glob("checkpoint_*_to_hf"))
    if not candidates:
        raise FileNotFoundError(f"No HF export dir found under {checkpoint_dir}")
    return sorted(candidates)[0]


def split_csv_arg(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    vals = [x.strip() for x in text.split(",")]
    vals = [x for x in vals if x]
    return vals if vals else None


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_dir}")


def build_eval_targets(args) -> List[Dict[str, Any]]:
    """
    Supported modes:

    1) Existing checkpoint mode (backward compatible):
       --base ... [--ckpt-regex ...] [--range ...]

    2) Direct single-model mode:
       --model-dir /path/to/model

    3) Multi-model names under base dir:
       --models model1,model2 --models-base /path/to/base
       or:
       --models model1,model2 --base /path/to/base
       (allowed for convenience)
    """
    targets: List[Dict[str, Any]] = []

    # Mode 2: direct explicit model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
        validate_model_dir(model_dir)
        targets.append({
            "name": model_dir.name,
            "kind": "model_dir",
            "hf_dir": model_dir,
            "source": str(model_dir),
        })
        return targets

    # Mode 3: named models under a base directory
    model_names = split_csv_arg(args.models)
    if model_names:
        models_base = args.models_base or args.base
        if not models_base:
            raise SystemExit("When using --models, provide --models-base or --base.")
        base_dir = Path(models_base)

        if not base_dir.exists():
            raise SystemExit(f"Models base directory does not exist: {base_dir}")
        if not base_dir.is_dir():
            raise SystemExit(f"Models base path is not a directory: {base_dir}")

        for model_name in model_names:
            model_dir = base_dir / model_name
            validate_model_dir(model_dir)
            targets.append({
                "name": model_name,
                "kind": "named_model",
                "hf_dir": model_dir,
                "source": str(model_dir),
            })
        return targets

    # Mode 1: checkpoint scanning (existing behavior)
    if not args.base:
        raise SystemExit("Checkpoint mode requires --base.")

    base = Path(args.base)
    if not base.exists():
        raise SystemExit(f"--base does not exist: {base}")
    if not base.is_dir():
        raise SystemExit(f"--base is not a directory: {base}")

    ckpts = discover_checkpoints(base, args.ckpt_regex, args.range_spec)
    log(f"Checkpoints selected: {len(ckpts)}")
    if not ckpts:
        raise SystemExit("No checkpoints matched; check --base, --ckpt-regex, --range.")

    for ckpt_dir in ckpts:
        step_match = re.match(args.ckpt_regex, ckpt_dir.name)
        step = int(step_match.group(1)) if step_match else ckpt_dir.name
        hf_dir = resolve_hf_dir(ckpt_dir)

        targets.append({
            "name": f"checkpoint{step}",
            "kind": "checkpoint",
            "hf_dir": hf_dir,
            "source": str(ckpt_dir),
        })

    return targets


# ------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------

def load_model(model_dir: Path):
    log(f"Loading model from: {model_dir}")

    config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Ensure padding is defined
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    ).eval()

    device = next(model.parameters()).device
    log(
        f"Model loaded on {device} | "
        f"eos={tokenizer.eos_token}({tokenizer.eos_token_id}) "
        f"pad_id={tokenizer.pad_token_id}"
    )
    return model, tokenizer, device


# ------------------------------------------------------------------------------
# Logprob scoring (MCQ + classify)
# ------------------------------------------------------------------------------

@torch.no_grad()
def score_candidate_logprob(
    model,
    tokenizer,
    device,
    prompt_text: str,
    candidate_text: str,
) -> float:
    """
    Compute log P(candidate_text | prompt_text) using teacher forcing.
    Works for both single-token labels ("A") and multi-token labels ("Neutral", "Nein", ...).
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    cand_ids = tokenizer(candidate_text, add_special_tokens=False).input_ids

    if len(cand_ids) == 0:
        return float("-inf")

    # Need room for cand_ids + 1 position for predicting first candidate token
    max_prompt = max(1, MAX_INPUT_LEN - len(cand_ids) - 1)
    prompt_ids = prompt_ids[-max_prompt:]

    full_ids = prompt_ids + cand_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    out = model(input_ids=input_ids)
    logits = out.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_ids)
    total = 0.0
    for i, tok_id in enumerate(cand_ids):
        pos = prompt_len + i - 1
        if pos < 0:
            continue
        total += float(log_probs[0, pos, tok_id].item())

    return total


def normalize_label_for_scoring(label: str) -> List[str]:
    """
    Some tokenizers strongly prefer a leading space before the first word.
    Score both variants and take the better one.
    """
    label = label.strip()
    variants = [label, " " + label]

    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@torch.no_grad()
def pick_best_label_by_logprob(
    model,
    tokenizer,
    device,
    prompt_text: str,
    labels: List[str],
) -> Tuple[str, Dict[str, float]]:
    """
    Returns best_label and a dict of label -> score.
    Scores are max over variants (with/without leading space).
    """
    scores: Dict[str, float] = {}
    for lab in labels:
        best = float("-inf")
        for variant in normalize_label_for_scoring(lab):
            s = score_candidate_logprob(model, tokenizer, device, prompt_text, variant)
            if s > best:
                best = s
        scores[lab] = best

    best_label = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_label, scores


def parse_mcq_gold(item: Dict[str, Any]) -> Optional[str]:
    """
    Backward-compatible:
    - prefer 'gold'
    - fall back to 'answer'
    """
    g = item.get("gold", item.get("answer", None))
    if g is None:
        return None
    g = str(g).strip().upper()
    return g if g in {"A", "B", "C", "D"} else None


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def build_output_path(
    target_name: str,
    outdir: Path,
    run_timestamp: str,
    dataset_tag: Optional[str] = None,
) -> Path:
    if dataset_tag:
        out_name = f"answers_{safe_name(target_name)}_{safe_name(dataset_tag)}_{run_timestamp}.json"
    else:
        out_name = f"answers_{safe_name(target_name)}_{run_timestamp}.json"
    return outdir / out_name

def evaluate_one_target(
    target: Dict[str, Any],
    items: List[Dict[str, Any]],
    args,
    dataset_path: Path,
    model,
    tokenizer,
    device,
    outdir: Path,
    run_timestamp: str,
    dataset_tag: Optional[str] = None,
) -> Path:
    target_name = target["name"]
    hf_dir = Path(target["hf_dir"])
    out_path = build_output_path(target_name, outdir, run_timestamp, dataset_tag)
    if out_path.exists():
        log(f"Skipping {target_name} (already exists): {out_path}")
        return out_path

    results: List[Dict[str, Any]] = []
    t_start = time.time()

    for it in items:
        instr = format_instruction(it)
        full_prompt = build_prompt(instr)
        t = it["type"]

        if t == "generate":
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_LEN,
            ).to(device)

            gen_cfg = dict(GENERATE_CFG)
            gen_cfg["eos_token_id"] = tokenizer.eos_token_id
            gen_cfg["pad_token_id"] = tokenizer.pad_token_id

            with torch.no_grad():
                output = model.generate(**inputs, **gen_cfg)

            new_tokens = output[0][inputs.input_ids.shape[-1]:]
            answer = clean_response(tokenizer.decode(new_tokens, skip_special_tokens=True))

            results.append({
                "id": it["id"],
                "lang": it.get("lang"),
                "type": t,
                "category": it.get("category", item_category(it)),
                "bias": it.get("bias", item_bias(it)),
                "prompt": instr,
                "answer": answer,
                "parsed_answer": None,
                "gold": it.get("gold", it.get("answer", None)),
            })
            continue

        if t == "mcq":
            labels = ["A", "B", "C", "D"]
            best, scores = pick_best_label_by_logprob(model, tokenizer, device, full_prompt, labels)
            gold = parse_mcq_gold(it)
            correct = (gold == best) if gold else None

            results.append({
                "id": it["id"],
                "lang": it.get("lang"),
                "type": t,
                "category": it.get("category", item_category(it)),
                "bias": it.get("bias", item_bias(it)),
                "prompt": instr,
                "answer": best,
                "parsed_answer": best,
                "gold": gold,
                "correct": correct,
                "label_logprobs": scores,
            })
            continue

        if t == "classify":
            allowed = it.get("allowed", None)
            if not allowed or not isinstance(allowed, list):
                raise ValueError(f"classify item missing allowed list: {it.get('id')}")

            allowed = [str(x) for x in allowed]
            best, scores = pick_best_label_by_logprob(model, tokenizer, device, full_prompt, allowed)

            results.append({
                "id": it["id"],
                "lang": it.get("lang"),
                "type": t,
                "category": it.get("category", item_category(it)),
                "bias": it.get("bias", item_bias(it)),
                "prompt": instr,
                "answer": best,
                "parsed_answer": best,
                "gold": it.get("gold", it.get("answer", None)),
                "label_logprobs": scores,
            })
            continue

        raise ValueError(f"Unknown type: {t}")

    elapsed = round(time.time() - t_start, 2)

    payload = {
        "run_timestamp": run_timestamp,
        "target_name": target_name,
        "target_kind": target["kind"],
        "checkpoint": target_name if target["kind"] == "checkpoint" else None,
        "hf_dir": str(hf_dir),
        "source": target.get("source"),
        "dataset": str(dataset_path),
        "filters": {
            "type": args.types,
            "lang": args.langs,
            "category": args.categories,
            "bias": args.biases,
        },
        "generation_config": {
            "generate_items": GENERATE_CFG,
            "scoring_items": SCORE_CFG,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "seconds": elapsed,
        "results": results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log(f"Done {target_name}: {out_path} ({elapsed}s)")
    return out_path


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        "Evaluate checkpoints and/or standalone HF models on a mixed-type dataset"
    )

    # Existing checkpoint mode args (kept backward compatible)
    ap.add_argument("--base", default=None,
                    help="Base directory containing checkpoint*/ dirs (checkpoint mode)")

    ap.add_argument("--dataset", required=True,
                    help="Dataset input: JSON file, comma-separated JSON files, or a directory of JSON files")

    ap.add_argument("--outdir", required=True,
                    help="Where to write outputs")

    ap.add_argument("--ckpt-regex", default=r"^checkpoint(\d+)$",
                    help="Regex for checkpoint dir names, must capture step number as group(1)")

    ap.add_argument("--range", dest="range_spec", default=None,
                    help="Optional: start:end:step, e.g. 0:2000:500")

    # New standalone model modes
    ap.add_argument("--model-dir", default=None,
                    help="Direct HF model directory to evaluate")

    ap.add_argument("--models", default=None,
                    help="Comma-separated model names under --models-base (or --base)")

    ap.add_argument("--models-base", default=None,
                    help="Base directory containing standalone model dirs for --models")

    # Dataset filters
    ap.add_argument("--type", dest="types", default=None,
                    help="Comma-separated: mcq,classify,generate")

    ap.add_argument("--lang", dest="langs", default=None,
                    help="Comma-separated: en,de,bar")

    ap.add_argument("--category", dest="categories", default=None,
                    help="Comma-separated categories")

    ap.add_argument("--bias", dest="biases", default=None,
                    help="Comma-separated: cultural,political,gender")

    # Generation knobs (only used for generate items)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    GENERATE_CFG["max_new_tokens"] = args.max_new
    GENERATE_CFG["temperature"] = args.temperature
    GENERATE_CFG["top_p"] = args.top_p

    targets = build_eval_targets(args)
    log(f"Evaluation targets selected: {len(targets)}")
    for t in targets:
        log(f"  - {t['name']} [{t['kind']}] -> {t['hf_dir']}")

    dataset_paths = resolve_dataset_paths(args.dataset)
    log(f"Dataset files selected: {len(dataset_paths)}")
    for ds in dataset_paths:
        log(f"  - {ds}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    multiple_datasets = len(dataset_paths) > 1

    prepared_datasets: List[Tuple[Path, List[Dict[str, Any]], Optional[str]]] = []
    for dataset_path in dataset_paths:
        items = load_dataset(dataset_path)
        items = filter_items(
            items,
            types=args.types.split(",") if args.types else None,
            langs=args.langs.split(",") if args.langs else None,
            categories=args.categories.split(",") if args.categories else None,
            biases=args.biases.split(",") if args.biases else None,
        )
        log(f"Dataset items after filtering [{dataset_path}]: {len(items)}")
        if not items:
            log(f"Skipping dataset with no remaining items after filtering: {dataset_path}")
            continue

        dataset_tag = dataset_path.stem if multiple_datasets else None
        prepared_datasets.append((dataset_path, items, dataset_tag))

    if not prepared_datasets:
        raise SystemExit("No dataset items left after filtering across all selected datasets.")

    for target in targets:
        hf_dir = Path(target["hf_dir"])
        model, tokenizer, device = load_model(hf_dir)
        try:
            for dataset_path, items, dataset_tag in prepared_datasets:
                out_path = build_output_path(target["name"], outdir, run_timestamp, dataset_tag)
                if out_path.exists():
                    log(f"Skipping {target['name']} (already exists): {out_path}")
                    continue

                evaluate_one_target(
                    target=target,
                    items=items,
                    args=args,
                    dataset_path=dataset_path,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    outdir=outdir,
                    run_timestamp=run_timestamp,
                    dataset_tag=dataset_tag,
                )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log("All selected targets finished.")


if __name__ == "__main__":
    main()
