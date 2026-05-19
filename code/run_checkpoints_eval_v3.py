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
    # keep your existing format
    return f"{SYSTEM_PROMPT}\n\n### User:\n{user_instr}\n\n### Assistant:\n"


def clean_response(text: str) -> str:
    for stop in ["\n###", "\nUser:", "\nAssistant:"]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


# ------------------------------------------------------------------------------
# Dataset (unchanged schema support)
# ------------------------------------------------------------------------------

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
    """
    Convert dataset items into one instruction string.

    IMPORTANT: Dataset format unchanged.
    We *only* add a stable "Answer:" anchor for MCQ/classify to improve scoring alignment.
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
# Checkpoint selection
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


# ------------------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------------------

def load_model(model_dir: Path):
    log(f"Loading model from: {model_dir}")

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ensure padding is defined
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
    log(f"Model loaded on {device} | eos={tokenizer.eos_token}({tokenizer.eos_token_id}) pad_id={tokenizer.pad_token_id}")
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
    Compute log P(candidate_text | prompt_text) using teacher-forcing.
    Works for both single-token labels ("A") and multi-token ("Neutral", "Nein", ...).
    """

    # Tokenize prompt and candidate *separately* to control special tokens.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    cand_ids = tokenizer(candidate_text, add_special_tokens=False).input_ids

    if len(cand_ids) == 0:
        return float("-inf")

    # Truncate prompt if needed to fit
    # Need room for cand_ids + 1 position for predicting first cand token.
    max_prompt = max(1, MAX_INPUT_LEN - len(cand_ids) - 1)
    prompt_ids = prompt_ids[-max_prompt:]

    full_ids = prompt_ids + cand_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    out = model(input_ids=input_ids)
    logits = out.logits  # [1, seq_len, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    # For each cand token at position (prompt_len + i), probability is from logits at (prompt_len + i - 1)
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
    Some tokenizers strongly prefer a leading space before first word.
    We score both variants and take the better one.

    For single-letter labels ("A"), the leading space variant may tokenize differently too.
    """
    label = label.strip()
    variants = [label, " " + label]
    # de-dup while preserving order
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
    Returns best_label and a dict of label->score.
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
    g = item.get("gold", None)
    if g is None:
        return None
    g = str(g).strip()
    return g if g in {"A", "B", "C", "D"} else None


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Evaluate many checkpoints on a simple mixed-type dataset (v3: logprob for mcq/classify)")
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
                    help="Comma-separated categories")
    ap.add_argument("--bias", dest="biases", default=None,
                    help="Comma-separated: cultural,political,gender")

    # generation knobs (only used for generate items)
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

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # eval each checkpoint
    for ckpt_dir in ckpts:
        step_match = re.match(args.ckpt_regex, ckpt_dir.name)
        step = int(step_match.group(1)) if step_match else ckpt_dir.name

        hf_dir = resolve_hf_dir(ckpt_dir)

        out_path = outdir / f"answers_checkpoint{step}_{run_timestamp}.json"
        if out_path.exists():
            log(f"Skipping checkpoint{step} (already exists): {out_path}")
            continue

        model, tokenizer, device = load_model(hf_dir)

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

                # ensure eos/pad are set for generate
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
                    "gold": it.get("gold", None),
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
                    "label_logprobs": scores,  # useful for calibration/analysis
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
                    "gold": it.get("gold", None),
                    "label_logprobs": scores,
                })
                continue

            raise ValueError(f"Unknown type: {t}")

        elapsed = round(time.time() - t_start, 2)

        payload = {
            "run_timestamp": run_timestamp,
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

        # free memory between checkpoints
        del model
        torch.cuda.empty_cache()

        log(f"Done checkpoint{step}: {out_path} ({elapsed}s)")

    log("All selected checkpoints finished.")


if __name__ == "__main__":
    main()

