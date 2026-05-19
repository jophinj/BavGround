#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

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

def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)

def build_prompt(user_instr: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n### User:\n{user_instr}\n\n### Assistant:\n"

def clean_response(text: str) -> str:
    for stop in ["\n###", "\nUser:", "\nAssistant:"]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()

# ──────────────────────────────────────────────────────────────────────────────
# Dataset handling (simple schema)
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    if not isinstance(items, list):
        raise ValueError("Dataset must be a list OR an object with an 'items' list.")
    return items

def item_category(item: Dict[str, Any]) -> str:
    # supports both styles: item["category"] or item["meta"]["category"]
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
        return [s.strip() for s in x] if x else None

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
    Convert your simple items into a single instruction string for the model.
    """
    t = item["type"]
    prompt = item["prompt"]

    if t == "mcq":
        opts = item.get("options", {})
        # keep it deterministic and easy to parse: ask for letter only
        opt_lines = "\n".join([f"{k}: {v}" for k, v in opts.items()])
        return f"{prompt}\n\nOptions:\n{opt_lines}\n\nAnswer with only A, B, C, or D."
    elif t == "classify":
        allowed = item.get("allowed", [])
        allowed_str = ", ".join([f"'{x}'" for x in allowed]) if allowed else "a valid label"
        return f"{prompt}\n\nAnswer with one of: {allowed_str}. No explanation."
    elif t == "generate":
        return prompt
    else:
        raise ValueError(f"Unknown item type: {t}")

# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint selection
# ──────────────────────────────────────────────────────────────────────────────

def parse_range_spec(spec: str) -> tuple[int, int, int]:
    """
    Format: start:end:step   e.g. 0:2000:500
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError("range spec must be start:end:step e.g. 0:2000:500")
    start, end, step = map(int, parts)
    if step <= 0:
        raise ValueError("step must be > 0")
    return start, end, step

def discover_checkpoints(
    base: Path,
    name_regex: str,
    range_spec: Optional[str],
) -> List[Path]:
    """
    Finds directories like: base/checkpoint500/, base/checkpoint1000/, ...
    If range_spec is provided, only keep checkpoints within [start, end] and matching step.
    """
    rx = re.compile(name_regex)

    found: List[tuple[int, Path]] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = rx.match(p.name)
        if not m:
            continue
        # first capturing group should be the numeric step
        step_num = int(m.group(1))
        found.append((step_num, p))

    found.sort(key=lambda x: x[0])

    if range_spec:
        start, end, step = parse_range_spec(range_spec)
        wanted = set(range(start, end + 1, step))
        found = [(n, p) for (n, p) in found if n in wanted]

    return [p for _, p in found]

def resolve_hf_dir(checkpoint_dir: Path) -> Path:
    """
    Your structure: training/checkpoint19500/checkpoint_19500_to_hf/
    We'll pick the first child dir that matches 'checkpoint_*_to_hf'
    """
    candidates = list(checkpoint_dir.glob("checkpoint_*_to_hf"))
    if not candidates:
        raise FileNotFoundError(f"No HF export dir found under {checkpoint_dir}")
    # if multiple, pick the first sorted
    return sorted(candidates)[0]

# ──────────────────────────────────────────────────────────────────────────────
# Main eval loop
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path):
    log(f"Loading model from: {model_dir}")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=True, local_files_only=True)

    GENERATE_CFG["eos_token_id"] = tokenizer.eos_token_id
    GENERATE_CFG["pad_token_id"] = tokenizer.eos_token_id

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
    log(f"Model loaded on {device}")
    return model, tokenizer, device

def main():
    ap = argparse.ArgumentParser("Evaluate many checkpoints on a simple mixed-type dataset")
    ap.add_argument("--base", required=True, help="Base directory containing checkpoint*/ dirs")
    ap.add_argument("--dataset", required=True, help="Path to dataset JSON")
    ap.add_argument("--outdir", required=True, help="Where to write outputs")

    # checkpoint selection
    ap.add_argument("--ckpt-regex", default=r"^checkpoint(\d+)$",
                    help="Regex for checkpoint dir names, must capture step number as group(1)")
    ap.add_argument("--range", dest="range_spec", default=None,
                    help="Optional: start:end:step, e.g. 0:2000:500")

    # dataset filters
    ap.add_argument("--type", dest="types", default=None,
                    help="Comma-separated: mcq,classify,generate")
    ap.add_argument("--lang", dest="langs", default=None,
                    help="Comma-separated: en,de,bar")
    ap.add_argument("--category", dest="categories", default=None,
                    help="Comma-separated categories (from item.category or item.meta.category)")
    ap.add_argument("--bias", dest="biases", default=None,
                    help="Comma-separated: cultural,political,gender (from item.bias or item.meta.bias)")

    # generation knobs (optional)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)

    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # set generation knobs
    GENERATE_CFG["max_new_tokens"] = args.max_new
    GENERATE_CFG["temperature"] = args.temperature
    GENERATE_CFG["top_p"] = args.top_p

    # load + filter dataset
    items = load_dataset(Path(args.dataset))
    items = filter_items(
        items,
        types=args.types.split(",") if args.types else None,
        langs=args.langs.split(",") if args.langs else None,
        categories=args.categories.split(",") if args.categories else None,
        biases=args.biases.split(",") if args.biases else None,
    )
    log(f"Dataset items after filtering: {len(items)}")
    if not items:
        raise SystemExit("No items left after filtering; adjust --type/--lang/--category/--bias.")

    # pick checkpoints
    ckpts = discover_checkpoints(base, args.ckpt_regex, args.range_spec)
    log(f"Checkpoints selected: {len(ckpts)}")
    if not ckpts:
        raise SystemExit("No checkpoints matched; check --base, --ckpt-regex, --range.")

    # eval each checkpoint
    for ckpt_dir in ckpts:
        step_match = re.match(args.ckpt_regex, ckpt_dir.name)
        step = int(step_match.group(1)) if step_match else ckpt_dir.name

        hf_dir = resolve_hf_dir(ckpt_dir)
        out_path = outdir / f"answers_checkpoint{step}.json"

        # resume support (skip if already exists)
        if out_path.exists():
            log(f"Skipping checkpoint{step} (already exists): {out_path}")
            continue

        model, tokenizer, device = load_model(hf_dir)

        results = []
        t_start = time.time()

        for it in items:
            instr = format_instruction(it)
            prompt = build_prompt(instr)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

            with torch.no_grad():
                output = model.generate(**inputs, **GENERATE_CFG)

            new_tokens = output[0][inputs.input_ids.shape[-1]:]
            answer = clean_response(tokenizer.decode(new_tokens, skip_special_tokens=True))

            results.append({
                "id": it["id"],
                "lang": it.get("lang"),
                "type": it.get("type"),
                "category": it.get("category", item_category(it)),
                "bias": it.get("bias", item_bias(it)),
                "prompt": instr,
                "answer": answer,
            })

        elapsed = round(time.time() - t_start, 2)

        payload = {
            "checkpoint": str(ckpt_dir.name),
            "hf_dir": str(hf_dir),
            "dataset": str(args.dataset),
            "filters": {
                "type": args.types,
                "lang": args.langs,
                "category": args.categories,
                "bias": args.biases,
            },
            "generation_config": GENERATE_CFG,
            "seconds": elapsed,
            "results": results,
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # free memory between checkpoints
        del model
        torch.cuda.empty_cache()

        log(f"Done checkpoint{step}: {out_path} ({elapsed}s)")

    log("All selected checkpoints finished.")

if __name__ == "__main__":
    main()

