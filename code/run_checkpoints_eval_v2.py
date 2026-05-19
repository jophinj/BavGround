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

# ──────────────────────────────────────────────────────────────────────────────
# Generation configs
# ──────────────────────────────────────────────────────────────────────────────

# Used for "generate" items; controlled by CLI: --max-new, --temperature, --top-p
GENERATE_CFG_DEFAULT = {
    "max_new_tokens": 128,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 3,
}

# Used for "mcq" and "classify" items (strict, deterministic)
GENERATE_CFG_STRICT = {
    "max_new_tokens": 5,
    "do_sample": False,
    "temperature": 1.0,   # ignored when do_sample=False
    "top_p": 1.0,         # ignored when do_sample=False
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}

def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)

# Plain prompt works better than Vicuna-style wrappers for many base LLaMA exports
def build_prompt(user_instr: str) -> str:
    return f"{user_instr}\n"

def clean_response(text: str) -> str:
    # Keep first chunk if the model still rambles
    for stop in ["\n###", "\nUser:", "\nAssistant:"]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()

def gen_cfg_for(item_type: str, eos_id: int) -> Dict[str, Any]:
    cfg = dict(GENERATE_CFG_DEFAULT if item_type == "generate" else GENERATE_CFG_STRICT)
    cfg["eos_token_id"] = eos_id
    cfg["pad_token_id"] = eos_id
    return cfg

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
    t = item["type"]
    prompt = item["prompt"]

    if t == "mcq":
        opts = item.get("options", {})
        opt_lines = "\n".join([f"{k}: {v}" for k, v in opts.items()])
        return (
            f"{prompt}\n\nOptions:\n{opt_lines}\n\n"
            "Reply with exactly one character: A, B, C, or D."
        )
    elif t == "classify":
        allowed = item.get("allowed", [])
        allowed_str = ", ".join([f"'{x}'" for x in allowed]) if allowed else "a valid label"
        return (
            f"{prompt}\n\n"
            f"Reply with exactly one of: {allowed_str}. No explanation."
        )
    elif t == "generate":
        return prompt
    else:
        raise ValueError(f"Unknown item type: {t}")

# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers (makes results directly usable)
# ──────────────────────────────────────────────────────────────────────────────

def parse_mcq(answer: str) -> Optional[str]:
    m = re.search(r"\b([ABCD])\b", answer)
    return m.group(1) if m else None

def parse_classify(answer: str, allowed: List[str]) -> Optional[str]:
    for lab in allowed:
        if re.search(rf"\b{re.escape(lab)}\b", answer):
            return lab
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint selection
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path):
    log(f"Loading model from: {model_dir}")
    model_dir_s = str(model_dir)

    config = AutoConfig.from_pretrained(
        model_dir_s,
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir_s,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"Tokenizer is not fast for {model_dir_s}. "
            "Fast tokenizer.json likely missing/corrupt, or HF is falling back to SentencePiece."
        )

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError(f"No eos_token_id found in tokenizer for {model_dir_s}")

    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_dir_s,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    ).eval()

    device = next(model.parameters()).device
    log(f"Model loaded on {device} | eos_token_id={eos_id}")
    return model, tokenizer, device, eos_id

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("Evaluate many checkpoints on a simple mixed-type dataset (v2)")
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

    # KEEP OLD ARGUMENTS (compat with your job script)
    # These only apply to type=generate items.
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)

    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Apply old knobs to GENERATE only
    GENERATE_CFG_DEFAULT["max_new_tokens"] = args.max_new
    GENERATE_CFG_DEFAULT["temperature"] = args.temperature
    GENERATE_CFG_DEFAULT["top_p"] = args.top_p

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

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ckpt_dir in ckpts:
        step_match = re.match(args.ckpt_regex, ckpt_dir.name)
        step = int(step_match.group(1)) if step_match else ckpt_dir.name

        hf_dir = resolve_hf_dir(ckpt_dir)

        # timestamped output so multiple experiments can coexist
        out_path = outdir / f"answers_checkpoint{step}_{run_stamp}.json"

        model, tokenizer, device, eos_id = load_model(hf_dir)

        results = []
        t_start = time.time()

        for it in items:
            instr = format_instruction(it)
            prompt = build_prompt(instr)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

            cfg = gen_cfg_for(it["type"], eos_id)

            with torch.no_grad():
                output = model.generate(**inputs, **cfg)

            new_tokens = output[0][inputs.input_ids.shape[-1]:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer = clean_response(raw)

            parsed = None
            if it["type"] == "mcq":
                parsed = parse_mcq(answer)
            elif it["type"] == "classify":
                parsed = parse_classify(answer, it.get("allowed", []))

            results.append({
                "id": it["id"],
                "lang": it.get("lang"),
                "type": it.get("type"),
                "category": it.get("category", item_category(it)),
                "bias": it.get("bias", item_bias(it)),
                "prompt": instr,
                "answer": answer,
                "parsed_answer": parsed,
                "gold": it.get("answer") if it.get("type") == "mcq" else None,
            })

        elapsed = round(time.time() - t_start, 2)

        payload = {
            "run_timestamp": run_stamp,
            "checkpoint": str(ckpt_dir.name),
            "hf_dir": str(hf_dir),
            "dataset": str(args.dataset),
            "filters": {
                "type": args.types,
                "lang": args.langs,
                "category": args.categories,
                "bias": args.biases,
            },
            "generation_config": {
                "generate_items": GENERATE_CFG_DEFAULT,
                "mcq_classify_items": GENERATE_CFG_STRICT,
                "eos_token_id": eos_id,
                "pad_token_id": eos_id,
            },
            "seconds": elapsed,
            "results": results,
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        del model
        torch.cuda.empty_cache()

        log(f"Done checkpoint{step}: {out_path} ({elapsed}s)")

    log("All selected checkpoints finished.")

if __name__ == "__main__":
    main()

