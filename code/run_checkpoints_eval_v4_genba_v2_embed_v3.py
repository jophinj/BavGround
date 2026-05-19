#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from run_checkpoints_eval_v4 import (
    GENERATE_CFG,
    MAX_INPUT_LEN,
    SCORE_CFG,
    build_eval_targets,
    build_prompt,
    clean_response,
    filter_items,
    item_bias,
    item_category,
    load_dataset,
    load_model,
    log,
    parse_mcq_gold,
    pick_best_label_by_logprob,
    resolve_dataset_paths,
    safe_name,
    score_candidate_logprob,
)


MCQ_STRATEGY_DESCRIPTIONS = {
    "letter": "Score answer letters A/B/C/D directly with a language-localized prompt.",
    "option_text": "Score full option texts instead of letters using total logprob.",
    "option_text_avg": "Score full option texts by average logprob per token.",
    "letter_shuffled": "Deterministically shuffle displayed answer positions and map back to original labels.",
    "generate_letter": "Generate with a language-localized prompt that asks for the answer letter only.",
    "generate_option_text": "Generate with a language-localized prompt that asks for the exact option text only.",
    "generate_letter_and_text": "Generate with a language-localized prompt that asks for letter first, then option text.",
    "embed_contextual_option": "Use the prompt final hidden-state anchor and compare it to prompt-conditioned option embeddings with cosine similarity.",
    "embed_isolated_option": "Use the prompt final hidden-state anchor and compare it to isolated option-text embeddings with cosine similarity.",
}

MCQ_GENERATE_MAX_NEW_TOKENS = 24
EMBED_CONTEXT_TEMPLATE = "Answer: {option_text}"


def ensure_generation_cache_compat() -> None:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if hasattr(DynamicCache, "get_max_length"):
        return

    def get_max_length(self):
        if hasattr(self, "get_max_cache_shape"):
            return self.get_max_cache_shape()
        return None

    DynamicCache.get_max_length = get_max_length


def split_csv_arg(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def build_output_path(
    target_name: str,
    outdir: Path,
    run_timestamp: str,
    strategy_name: str,
    dataset_tag: Optional[str] = None,
) -> Path:
    suffix = safe_name(strategy_name)
    if dataset_tag:
        name = f"answers_{safe_name(target_name)}_{safe_name(dataset_tag)}_{suffix}_{run_timestamp}.json"
    else:
        name = f"answers_{safe_name(target_name)}_{suffix}_{run_timestamp}.json"
    return outdir / name


def deterministic_shuffle(labels: List[str], item_id: str) -> List[str]:
    seed = int(hashlib.md5(item_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    out = list(labels)
    rng.shuffle(out)
    return out


def mcq_prompt_instructions(lang: Optional[str]) -> Dict[str, str]:
    if lang == "de":
        return {
            "letter_only": "Antworte nur mit genau einem Buchstaben: A, B, C oder D.",
            "option_text": "Antworte nur mit dem exakten Optionstext. Keine Erklärung.",
            "letter_and_text": "Beginne mit dem richtigen Buchstaben A, B, C oder D. Danach darfst du optional den exakten Optionstext nennen. Beispiele: 'B' oder 'B: Lenbachhaus'.",
        }
    if lang == "bar":
        return {
            "letter_only": "Antwort bloß mit genau oam Buchstaben: A, B, C oda D.",
            "option_text": "Antwort bloß mit dem exakten Optionstext. Koane Erklärung.",
            "letter_and_text": "Fang mit dem richtigen Buchstaben A, B, C oda D o. Danach derfst optional den exakten Optionstext dazuschreibn. Beispui: 'B' oder 'B: Lenbachhaus'.",
        }
    return {
        "letter_only": "Reply with exactly one character: A, B, C, or D.",
        "option_text": "Reply with the exact option text only. No explanation.",
        "letter_and_text": "Reply by starting with the correct option letter: A, B, C, or D. You may optionally continue with the exact option text. Examples: 'B' or 'B: Lenbachhaus'.",
    }


def format_standard_mcq(item: Dict[str, Any]) -> str:
    opts = item.get("options", {})
    opt_lines = "\n".join([f"{k}: {v}" for k, v in opts.items()])
    instr = mcq_prompt_instructions(item.get("lang"))
    return f"{item['prompt']}\n\nOptions:\n{opt_lines}\n\n{instr['letter_only']}\nAnswer:"


def format_option_text_mcq(item: Dict[str, Any]) -> str:
    opts = item.get("options", {})
    opt_lines = "\n".join([f"{k}: {v}" for k, v in opts.items()])
    instr = mcq_prompt_instructions(item.get("lang"))
    return f"{item['prompt']}\n\nOptions:\n{opt_lines}\n\n{instr['option_text']}\nAnswer:"


def format_generate_letter_and_text_mcq(item: Dict[str, Any]) -> str:
    opts = item.get("options", {})
    opt_lines = "\n".join([f"{k}: {v}" for k, v in opts.items()])
    instr = mcq_prompt_instructions(item.get("lang"))
    return f"{item['prompt']}\n\nOptions:\n{opt_lines}\n\n{instr['letter_and_text']}\nAnswer:"


def format_generate_option_text_mcq(item: Dict[str, Any]) -> str:
    opts = item.get("options", {})
    opt_lines = "\n".join([f"{k}: {v}" for k, v in opts.items()])
    instr = mcq_prompt_instructions(item.get("lang"))
    return (
        f"{item['prompt']}\n\nOptions:\n{opt_lines}\n\n"
        f"{instr['option_text']}\n"
        "Do not repeat the question.\n"
        "Answer:"
    )


def normalize_text_for_match(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[“”\"'`´]", "", text)
    text = re.sub(r"[^a-z0-9äöüß\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_generated_letter(text: str) -> Optional[str]:
    upper = text.upper()
    match = re.search(r"\b(?:ANSWER|ANTWORT)?\s*[:\-]?\s*([ABCD])\b", upper)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([ABCD])(?:[\s\)\]\.\:\-\,]|$)", upper)
    if match:
        return match.group(1)
    return None


def token_overlap_score(text_words: List[str], option_words: List[str]) -> float:
    if not text_words or not option_words:
        return 0.0
    overlap = len(set(text_words) & set(option_words))
    return overlap / max(1, len(set(option_words)))


def parse_generated_option_label(
    text: str,
    options: Dict[str, str],
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    normalized_text = normalize_text_for_match(text)
    if not normalized_text:
        return None, None, None

    normalized_options = {
        label: normalize_text_for_match(option_text)
        for label, option_text in options.items()
    }

    for label, option_text in normalized_options.items():
        if option_text and normalized_text == option_text:
            return label, "option_exact", 1.0

    for label, option_text in normalized_options.items():
        if option_text and option_text in normalized_text:
            return label, "option_contains", 1.0

    for label, option_text in normalized_options.items():
        if option_text and normalized_text.startswith(option_text):
            return label, "option_prefix", 1.0

    text_words = normalized_text.split()
    best_label = None
    best_score = 0.0
    for label, option_text in normalized_options.items():
        opt_words = option_text.split()
        if len(opt_words) >= 2:
            prefix_len = min(len(text_words), len(opt_words), 3)
            if prefix_len >= 2 and text_words[:prefix_len] == opt_words[:prefix_len]:
                return label, "option_word_prefix", 1.0
        score = token_overlap_score(text_words, opt_words)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is not None and best_score >= 0.5:
        return best_label, "option_token_overlap", best_score
    return None, None, best_score if best_label is not None else None


def parse_generated_mcq_answer(
    text: str,
    options: Dict[str, str],
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    letter = parse_generated_letter(text)
    if letter:
        return letter, "letter", 1.0
    return parse_generated_option_label(text, options)


def candidate_text_variants(text: str) -> List[str]:
    text = text.strip()
    variants = [text, " " + text]
    seen = set()
    out = []
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def nonempty_mcq_options(item: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    options = item.get("options", {})
    valid: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    for label, option_text in options.items():
        text = "" if option_text is None else str(option_text)
        if text.strip():
            valid[label] = text
        else:
            skipped[label] = text

    if not valid:
        raise ValueError(f"MCQ item has no non-empty options: {item.get('id')}")
    return valid, skipped


def mean_pool_hidden(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.mean(dim=0)


def model_forward_inputs(inputs: Any) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in inputs.items() if k != "token_type_ids"}


def encode_hidden_states(
    model,
    tokenizer,
    device,
    text: str,
) -> Tuple[torch.Tensor, List[int]]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
        add_special_tokens=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**model_forward_inputs(inputs), output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1][0]
    token_ids = inputs.input_ids[0].tolist()
    return hidden, token_ids


def prompt_answer_anchor(
    model,
    tokenizer,
    device,
    prompt_text: str,
) -> torch.Tensor:
    hidden, _ = encode_hidden_states(model, tokenizer, device, prompt_text)
    return hidden[-1].detach()


def contextual_option_embedding(
    model,
    tokenizer,
    device,
    prompt_text: str,
    option_text: str,
) -> Dict[str, Any]:
    best_score = float("-inf")
    best_variant = option_text
    best_embedding: Optional[torch.Tensor] = None
    best_token_count = 0

    for variant in candidate_text_variants(option_text):
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        ).input_ids
        full_text = prompt_text + variant
        hidden, token_ids = encode_hidden_states(model, tokenizer, device, full_text)
        option_token_count = max(0, len(token_ids) - len(prompt_ids))
        if option_token_count <= 0:
            continue
        option_hidden = hidden[-option_token_count:]
        pooled = mean_pool_hidden(option_hidden).detach()
        score_proxy = pooled.norm().item()
        if score_proxy > best_score:
            best_score = score_proxy
            best_variant = variant
            best_embedding = pooled
            best_token_count = option_token_count

    if best_embedding is None:
        raise ValueError(f"Could not derive contextual embedding for option: {option_text!r}")

    return {
        "embedding": best_embedding,
        "token_count": best_token_count,
        "best_variant": best_variant,
    }


def isolated_option_embedding(
    model,
    tokenizer,
    device,
    option_text: str,
) -> Dict[str, Any]:
    best_score = float("-inf")
    best_variant = option_text
    best_embedding: Optional[torch.Tensor] = None
    best_token_count = 0
    best_extraction_method = "prefix_delta"

    for variant in candidate_text_variants(option_text):
        clean_variant = variant.strip()
        context_text = EMBED_CONTEXT_TEMPLATE.format(option_text=clean_variant)
        hidden, token_ids = encode_hidden_states(model, tokenizer, device, context_text)
        prefix_ids = tokenizer(
            EMBED_CONTEXT_TEMPLATE.format(option_text=""),
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        ).input_ids
        option_token_count = max(0, len(token_ids) - len(prefix_ids))
        extraction_method = "prefix_delta"

        if option_token_count <= 0:
            standalone_ids = tokenizer(
                clean_variant,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_INPUT_LEN,
            ).input_ids
            option_token_count = len(standalone_ids)
            extraction_method = "standalone_token_count_fallback"

        if option_token_count <= 0:
            continue
        option_token_count = min(option_token_count, hidden.shape[0])
        option_hidden = hidden[-option_token_count:]
        pooled = mean_pool_hidden(option_hidden).detach()
        score_proxy = pooled.norm().item()
        if score_proxy > best_score:
            best_score = score_proxy
            best_variant = variant
            best_embedding = pooled
            best_token_count = option_token_count
            best_extraction_method = extraction_method

    if best_embedding is None:
        raise ValueError(f"Could not derive isolated embedding for option: {option_text!r}")

    return {
        "embedding": best_embedding,
        "token_count": best_token_count,
        "best_variant": best_variant,
        "extraction_method": best_extraction_method,
    }


def cosine_similarity_score(anchor: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(F.cosine_similarity(anchor.unsqueeze(0), candidate.unsqueeze(0)).item())


def score_candidate_details(
    model,
    tokenizer,
    device,
    prompt_text: str,
    candidate_text: str,
) -> Dict[str, Any]:
    best_total = float("-inf")
    best_variant = candidate_text
    best_token_count = 0

    for variant in candidate_text_variants(candidate_text):
        cand_ids = tokenizer(variant, add_special_tokens=False).input_ids
        total = score_candidate_logprob(model, tokenizer, device, prompt_text, variant)
        token_count = len(cand_ids)
        if total > best_total:
            best_total = total
            best_variant = variant
            best_token_count = token_count

    avg = best_total / best_token_count if best_token_count > 0 else float("-inf")
    return {
        "total_logprob": best_total,
        "token_count": best_token_count,
        "avg_logprob_per_token": avg,
        "best_variant": best_variant,
    }


def evaluate_mcq_letter(item: Dict[str, Any], model, tokenizer, device) -> Dict[str, Any]:
    instr = format_standard_mcq(item)
    full_prompt = build_prompt(instr)
    labels = ["A", "B", "C", "D"]
    best, scores = pick_best_label_by_logprob(model, tokenizer, device, full_prompt, labels)
    gold = parse_mcq_gold(item)
    return {
        "prompt": instr,
        "answer": best,
        "parsed_answer": best,
        "gold": gold,
        "correct": (gold == best) if gold else None,
        "label_logprobs": scores,
        "diagnostics": {
            "strategy": "letter",
            "strategy_description": MCQ_STRATEGY_DESCRIPTIONS["letter"],
        },
    }


def evaluate_mcq_option_text(
    item: Dict[str, Any],
    model,
    tokenizer,
    device,
    use_avg_logprob: bool = False,
) -> Dict[str, Any]:
    instr = format_option_text_mcq(item)
    full_prompt = build_prompt(instr)
    options, skipped_options = nonempty_mcq_options(item)
    scores: Dict[str, float] = {}
    candidate_details: Dict[str, Dict[str, Any]] = {}
    for label, option_text in options.items():
        details = score_candidate_details(model, tokenizer, device, full_prompt, option_text)
        candidate_details[label] = details
        scores[label] = details["avg_logprob_per_token"] if use_avg_logprob else details["total_logprob"]

    best = max(scores.items(), key=lambda kv: kv[1])[0]
    gold = parse_mcq_gold(item)
    strategy_name = "option_text_avg" if use_avg_logprob else "option_text"
    return {
        "prompt": instr,
        "answer": best,
        "parsed_answer": best,
        "gold": gold,
        "correct": (gold == best) if gold else None,
        "label_logprobs": scores,
        "diagnostics": {
            "strategy": strategy_name,
            "strategy_description": MCQ_STRATEGY_DESCRIPTIONS[strategy_name],
            "scored_candidates": options,
            "skipped_empty_options": skipped_options,
            "score_mode": "avg_logprob_per_token" if use_avg_logprob else "total_logprob",
            "candidate_score_details": candidate_details,
        },
    }


def evaluate_mcq_embedding(
    item: Dict[str, Any],
    model,
    tokenizer,
    device,
    use_contextual_option: bool,
) -> Dict[str, Any]:
    instr = format_option_text_mcq(item)
    full_prompt = build_prompt(instr)
    anchor = prompt_answer_anchor(model, tokenizer, device, full_prompt)
    options, skipped_options = nonempty_mcq_options(item)
    scores: Dict[str, float] = {}
    candidate_details: Dict[str, Dict[str, Any]] = {}

    for label, option_text in options.items():
        if use_contextual_option:
            emb_info = contextual_option_embedding(model, tokenizer, device, full_prompt, option_text)
            score_mode = "prompt_anchor_to_prompt_conditioned_option_cosine"
            strategy_name = "embed_contextual_option"
        else:
            emb_info = isolated_option_embedding(model, tokenizer, device, option_text)
            score_mode = "prompt_anchor_to_isolated_option_cosine"
            strategy_name = "embed_isolated_option"

        cosine = cosine_similarity_score(anchor, emb_info["embedding"])
        scores[label] = cosine
        candidate_details[label] = {
            "cosine_similarity": cosine,
            "token_count": emb_info["token_count"],
            "best_variant": emb_info["best_variant"],
            "extraction_method": emb_info.get("extraction_method", "prefix_delta"),
        }

    best = max(scores.items(), key=lambda kv: kv[1])[0]
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else None
    gold = parse_mcq_gold(item)
    return {
        "prompt": instr,
        "answer": best,
        "parsed_answer": best,
        "gold": gold,
        "correct": (gold == best) if gold else None,
        "label_logprobs": scores,
        "diagnostics": {
            "strategy": strategy_name,
            "strategy_description": MCQ_STRATEGY_DESCRIPTIONS[strategy_name],
            "score_mode": score_mode,
            "candidate_embedding_details": candidate_details,
            "skipped_empty_options": skipped_options,
            "winning_margin": margin,
            "embedding_note": (
                "Scores are cosine similarities between the prompt answer-anchor "
                "and pooled candidate embeddings, not token probabilities."
            ),
        },
    }


def evaluate_mcq_letter_shuffled(item: Dict[str, Any], model, tokenizer, device) -> Dict[str, Any]:
    original_options = item.get("options", {})
    original_labels = ["A", "B", "C", "D"]
    display_to_original: Dict[str, str] = {}
    original_to_display: Dict[str, str] = {}

    shuffled_originals = deterministic_shuffle(original_labels, str(item["id"]))
    displayed_lines = []
    for display_label, original_label in zip(original_labels, shuffled_originals):
        displayed_lines.append(f"{display_label}: {original_options[original_label]}")
        display_to_original[display_label] = original_label
        original_to_display[original_label] = display_label

    instr = (
        f"{item['prompt']}\n\nOptions:\n"
        + "\n".join(displayed_lines)
        + "\n\n"
        + mcq_prompt_instructions(item.get("lang"))["letter_only"]
        + "\nAnswer:"
    )
    full_prompt = build_prompt(instr)
    displayed_labels = ["A", "B", "C", "D"]
    displayed_best, displayed_scores = pick_best_label_by_logprob(
        model,
        tokenizer,
        device,
        full_prompt,
        displayed_labels,
    )

    original_scores = {
        original_label: displayed_scores[display_label]
        for display_label, original_label in display_to_original.items()
    }
    best = display_to_original[displayed_best]
    gold = parse_mcq_gold(item)
    return {
        "prompt": instr,
        "answer": best,
        "parsed_answer": best,
        "gold": gold,
        "correct": (gold == best) if gold else None,
        "label_logprobs": original_scores,
        "diagnostics": {
            "strategy": "letter_shuffled",
            "strategy_description": MCQ_STRATEGY_DESCRIPTIONS["letter_shuffled"],
            "display_to_original": display_to_original,
            "original_to_display": original_to_display,
            "displayed_answer": displayed_best,
            "displayed_label_logprobs": displayed_scores,
        },
    }


def generate_mcq_raw(
    prompt_text: str,
    model,
    tokenizer,
    device,
) -> str:
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LEN,
    ).to(device)

    gen_cfg = dict(GENERATE_CFG)
    gen_cfg["do_sample"] = False
    gen_cfg["max_new_tokens"] = MCQ_GENERATE_MAX_NEW_TOKENS
    gen_cfg["temperature"] = 1.0
    gen_cfg["top_p"] = 1.0
    gen_cfg["repetition_penalty"] = 1.0
    gen_cfg["no_repeat_ngram_size"] = 0
    gen_cfg["eos_token_id"] = tokenizer.eos_token_id
    gen_cfg["pad_token_id"] = tokenizer.pad_token_id
    gen_cfg["use_cache"] = False

    ensure_generation_cache_compat()
    with torch.no_grad():
        output = model.generate(**model_forward_inputs(inputs), **gen_cfg)

    new_tokens = output[0][inputs.input_ids.shape[-1]:]
    return clean_response(tokenizer.decode(new_tokens, skip_special_tokens=True))


def evaluate_mcq_generate_variant(
    item: Dict[str, Any],
    model,
    tokenizer,
    device,
    variant: str,
) -> Dict[str, Any]:
    if variant == "generate_letter":
        instr = format_standard_mcq(item)
    elif variant == "generate_option_text":
        instr = format_generate_option_text_mcq(item)
    elif variant == "generate_letter_and_text":
        instr = format_generate_letter_and_text_mcq(item)
    else:
        raise ValueError(f"Unknown generate variant: {variant}")

    full_prompt = build_prompt(instr)
    raw_answer = generate_mcq_raw(full_prompt, model, tokenizer, device)
    options = item.get("options", {})
    parsed, parse_method, parse_confidence = parse_generated_mcq_answer(raw_answer, options)
    gold = parse_mcq_gold(item)

    return {
        "prompt": instr,
        "answer": parsed if parsed else raw_answer,
        "parsed_answer": parsed,
        "gold": gold,
        "correct": (gold == parsed) if gold and parsed else None,
        "label_logprobs": None,
        "diagnostics": {
            "strategy": variant,
            "strategy_description": MCQ_STRATEGY_DESCRIPTIONS[variant],
            "raw_generation": raw_answer,
            "parse_method": parse_method,
            "parse_confidence": parse_confidence,
            "max_new_tokens": MCQ_GENERATE_MAX_NEW_TOKENS,
        },
    }


def evaluate_classify(item: Dict[str, Any], model, tokenizer, device) -> Dict[str, Any]:
    allowed = item.get("allowed", None)
    if not allowed or not isinstance(allowed, list):
        raise ValueError(f"classify item missing allowed list: {item.get('id')}")

    allowed = [str(x) for x in allowed]
    if item.get("lang") == "de":
        instruction = f"Antworte genau mit einem von: {', '.join([repr(x) for x in allowed])}. Keine Erklärung."
    elif item.get("lang") == "bar":
        instruction = f"Antwort genau mit oam von: {', '.join([repr(x) for x in allowed])}. Koane Erklärung."
    else:
        instruction = f"Reply with exactly one of: {', '.join([repr(x) for x in allowed])}. No explanation."

    instr = f"{item['prompt']}\n\n{instruction}\nAnswer:"
    full_prompt = build_prompt(instr)
    best, scores = pick_best_label_by_logprob(model, tokenizer, device, full_prompt, allowed)
    return {
        "prompt": instr,
        "answer": best,
        "parsed_answer": best,
        "gold": item.get("gold", item.get("answer", None)),
        "label_logprobs": scores,
        "diagnostics": {
            "strategy": "classify_logprob",
            "strategy_description": "Standard classify scoring retained with language-localized instructions.",
        },
    }


def evaluate_generate(item: Dict[str, Any], model, tokenizer, device) -> Dict[str, Any]:
    instr = item["prompt"]
    full_prompt = build_prompt(instr)
    raw_answer = generate_mcq_raw(full_prompt, model, tokenizer, device)
    return {
        "prompt": instr,
        "answer": raw_answer,
        "parsed_answer": None,
        "gold": item.get("gold", item.get("answer", None)),
        "diagnostics": {
            "strategy": "generate_freeform",
            "strategy_description": "Standard free-form generation retained for compatibility.",
        },
    }


def evaluate_item_with_strategy(
    item: Dict[str, Any],
    strategy: str,
    model,
    tokenizer,
    device,
) -> Dict[str, Any]:
    item_type = item["type"]

    if item_type == "mcq":
        if strategy == "letter":
            core = evaluate_mcq_letter(item, model, tokenizer, device)
        elif strategy == "option_text":
            core = evaluate_mcq_option_text(item, model, tokenizer, device, use_avg_logprob=False)
        elif strategy == "option_text_avg":
            core = evaluate_mcq_option_text(item, model, tokenizer, device, use_avg_logprob=True)
        elif strategy == "embed_contextual_option":
            core = evaluate_mcq_embedding(item, model, tokenizer, device, use_contextual_option=True)
        elif strategy == "embed_isolated_option":
            core = evaluate_mcq_embedding(item, model, tokenizer, device, use_contextual_option=False)
        elif strategy == "letter_shuffled":
            core = evaluate_mcq_letter_shuffled(item, model, tokenizer, device)
        elif strategy in {"generate_letter", "generate_option_text", "generate_letter_and_text"}:
            core = evaluate_mcq_generate_variant(item, model, tokenizer, device, strategy)
        else:
            raise ValueError(f"Unknown MCQ strategy: {strategy}")
    elif item_type == "classify":
        core = evaluate_classify(item, model, tokenizer, device)
    elif item_type == "generate":
        core = evaluate_generate(item, model, tokenizer, device)
    else:
        raise ValueError(f"Unknown type: {item_type}")

    return {
        "id": item["id"],
        "lang": item.get("lang"),
        "type": item_type,
        "category": item.get("category", item_category(item)),
        "bias": item.get("bias", item_bias(item)),
        **core,
    }


def evaluate_one_target_strategy(
    target: Dict[str, Any],
    items: List[Dict[str, Any]],
    args,
    dataset_path: Path,
    model,
    tokenizer,
    device,
    outdir: Path,
    run_timestamp: str,
    strategy_name: str,
    dataset_tag: Optional[str] = None,
) -> Path:
    target_name = target["name"]
    hf_dir = Path(target["hf_dir"])
    out_path = build_output_path(target_name, outdir, run_timestamp, strategy_name, dataset_tag)
    if out_path.exists():
        log(f"Skipping {target_name} [{strategy_name}] (already exists): {out_path}")
        return out_path

    results: List[Dict[str, Any]] = []
    t_start = time.time()
    for item in items:
        results.append(evaluate_item_with_strategy(item, strategy_name, model, tokenizer, device))

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
        "diagnostic_config": {
            "mcq_strategy": strategy_name,
            "mcq_strategy_description": MCQ_STRATEGY_DESCRIPTIONS.get(strategy_name),
            "available_mcq_strategies": MCQ_STRATEGY_DESCRIPTIONS,
            "mcq_generate_max_new_tokens": MCQ_GENERATE_MAX_NEW_TOKENS,
            "analysis_goal": (
                "Diagnose whether poor MCQ accuracy is driven by answer-letter bias, "
                "position bias, tokenizer length effects, generation parsing issues, "
                "or hidden-state semantic alignment under embedding-based scoring."
            ),
            "metric_taxonomy": (
                "This run mixes probability-based metrics (logprob scoring), "
                "embedding-based metrics (cosine similarity over hidden-state embeddings), "
                "and generated-text-based metrics (generate_* strategies)."
            ),
        },
        "seconds": elapsed,
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log(f"Done {target_name} [{strategy_name}]: {out_path} ({elapsed}s)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        "GENBA diagnostic evaluator embed_v3 with language-localized prompts, embedding-based MCQ metrics, and expanded generate strategies"
    )
    ap.add_argument("--base", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ckpt-regex", default=r"^checkpoint(\d+)$")
    ap.add_argument("--range", dest="range_spec", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--models-base", default=None)
    ap.add_argument("--type", dest="types", default=None)
    ap.add_argument("--lang", dest="langs", default=None)
    ap.add_argument("--category", dest="categories", default=None)
    ap.add_argument("--bias", dest="biases", default=None)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument(
        "--mcq-strategies",
        default="letter,option_text,option_text_avg,embed_contextual_option,embed_isolated_option,letter_shuffled,generate_letter,generate_option_text,generate_letter_and_text",
        help=(
            "Comma-separated MCQ diagnostic strategies. Useful options: "
            "letter, option_text, option_text_avg, embed_contextual_option, "
            "embed_isolated_option, letter_shuffled, "
            "generate_letter, generate_option_text, generate_letter_and_text"
        ),
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    GENERATE_CFG["max_new_tokens"] = args.max_new
    GENERATE_CFG["temperature"] = args.temperature
    GENERATE_CFG["top_p"] = args.top_p

    strategies = split_csv_arg(args.mcq_strategies)
    if not strategies:
        raise SystemExit("No MCQ strategies selected.")
    for strategy in strategies:
        if strategy not in MCQ_STRATEGY_DESCRIPTIONS:
            raise SystemExit(f"Unknown MCQ strategy: {strategy}")

    targets = build_eval_targets(args)
    log(f"Evaluation targets selected: {len(targets)}")
    for target in targets:
        log(f"  - {target['name']} [{target['kind']}] -> {target['hf_dir']}")

    dataset_paths = resolve_dataset_paths(args.dataset)
    log(f"Dataset files selected: {len(dataset_paths)}")
    for ds in dataset_paths:
        log(f"  - {ds}")
    log(f"MCQ diagnostic strategies selected: {', '.join(strategies)}")

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
                for strategy_name in strategies:
                    out_path = build_output_path(
                        target["name"],
                        outdir,
                        run_timestamp,
                        strategy_name,
                        dataset_tag,
                    )
                    if out_path.exists():
                        log(f"Skipping {target['name']} [{strategy_name}] (already exists): {out_path}")
                        continue
                    evaluate_one_target_strategy(
                        target=target,
                        items=items,
                        args=args,
                        dataset_path=dataset_path,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        outdir=outdir,
                        run_timestamp=run_timestamp,
                        strategy_name=strategy_name,
                        dataset_tag=dataset_tag,
                    )
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log("All selected targets finished.")


if __name__ == "__main__":
    main()
