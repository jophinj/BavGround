#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SYSTEM_PROMPT = (
    "You are a helpful, precise assistant.\n"
    "Follow the user instruction.\n"
    "Do not add meta-commentary.\n"
    "Do not ask follow-up questions.\n"
    "Finish the answer and stop."
)

# gpt-4.1-mini is the default because it is inexpensive, strong enough for this
# benchmark-style closed-model baseline, and supports chat logprobs for the
# letter strategy. Override with --model for stronger or newer baselines.
DEFAULT_MODEL = "gpt-4.1-mini"

MCQ_STRATEGY_DESCRIPTIONS = {
    "letter": "Score answer letters A/B/C/D directly from first-token API logprobs.",
    "generate_letter_and_text": (
        "Generate with a language-localized prompt that asks for letter first, "
        "then option text."
    ),
}

MCQ_GENERATE_MAX_OUTPUT_TOKENS = 24
LETTER_TOP_LOGPROBS = 5
GPT5_MIN_LETTER_COMPLETION_TOKENS = 64
GPT5_MIN_GENERATE_COMPLETION_TOKENS = 128


def log(msg: str) -> None:
    print(f"[closed-eval] {msg}", flush=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def split_csv_arg(text: Optional[str]) -> Optional[List[str]]:
    if not text:
        return None
    vals = [part.strip() for part in text.split(",") if part.strip()]
    return vals or None


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    if not isinstance(items, list):
        raise ValueError("Dataset must be a list OR an object with an 'items' list.")
    return items


def resolve_dataset_paths(dataset_arg: str) -> List[Path]:
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
    out = []
    for item in items:
        if types and item.get("type") not in types:
            continue
        if langs and item.get("lang") not in langs:
            continue
        if categories and item_category(item) not in categories:
            continue
        if biases and item_bias(item) not in biases:
            continue
        out.append(item)
    return out


def parse_mcq_gold(item: Dict[str, Any]) -> Optional[str]:
    gold = item.get("gold", item.get("answer", None))
    if gold is None:
        return None
    gold = str(gold).strip().upper()
    return gold if gold in {"A", "B", "C", "D"} else None


def mcq_prompt_instructions(lang: Optional[str]) -> Dict[str, str]:
    if lang == "de":
        return {
            "letter_only": "Antworte nur mit genau einem Buchstaben: A, B, C oder D.",
            "letter_and_text": "Beginne mit dem richtigen Buchstaben A, B, C oder D. Danach darfst du optional den exakten Optionstext nennen. Beispiele: 'B' oder 'B: Lenbachhaus'.",
        }
    if lang == "bar":
        return {
            "letter_only": "Antwort bloß mit genau oam Buchstaben: A, B, C oda D.",
            "letter_and_text": "Fang mit dem richtigen Buchstaben A, B, C oda D o. Danach derfst optional den exakten Optionstext dazuschreibn. Beispui: 'B' oder 'B: Lenbachhaus'.",
        }
    return {
        "letter_only": "Reply with exactly one character: A, B, C, or D.",
        "letter_and_text": "Reply by starting with the correct option letter: A, B, C, or D. You may optionally continue with the exact option text. Examples: 'B' or 'B: Lenbachhaus'.",
    }


def format_standard_mcq(item: Dict[str, Any]) -> str:
    opts = item.get("options", {})
    opt_lines = "\n".join([f"{k}: {v}" for k, v in opts.items()])
    instr = mcq_prompt_instructions(item.get("lang"))
    return f"{item['prompt']}\n\nOptions:\n{opt_lines}\n\n{instr['letter_only']}\nAnswer:"


def format_generate_letter_and_text_mcq(item: Dict[str, Any]) -> str:
    opts = item.get("options", {})
    opt_lines = "\n".join([f"{k}: {v}" for k, v in opts.items()])
    instr = mcq_prompt_instructions(item.get("lang"))
    return f"{item['prompt']}\n\nOptions:\n{opt_lines}\n\n{instr['letter_and_text']}\nAnswer:"


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


def clean_response(text: str) -> str:
    for stop in ["\n###", "\nUser:", "\nAssistant:"]:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


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


def model_uses_default_temperature(model: str) -> bool:
    return model.startswith("gpt-5")


def maybe_add_temperature(kwargs: Dict[str, Any], args) -> Dict[str, Any]:
    if args.temperature is None:
        return kwargs
    if model_uses_default_temperature(args.model):
        return kwargs
    return {**kwargs, "temperature": args.temperature}


def effective_letter_max_completion_tokens(args) -> int:
    if args.letter_max_output_tokens is not None:
        return args.letter_max_output_tokens
    if model_uses_default_temperature(args.model):
        return GPT5_MIN_LETTER_COMPLETION_TOKENS
    return 1


def effective_generate_max_completion_tokens(args) -> int:
    if model_uses_default_temperature(args.model):
        return max(args.max_output_tokens, GPT5_MIN_GENERATE_COMPLETION_TOKENS)
    return args.max_output_tokens


def completion_with_retries(client: Any, args, messages: List[Dict[str, str]], **kwargs):
    last_error: Optional[Exception] = None
    for attempt in range(args.retries + 1):
        try:
            return client.chat.completions.create(
                model=args.model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:  # API exceptions vary by SDK version.
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            if status_code == 400:
                raise
            if attempt >= args.retries:
                break
            sleep_for = args.retry_base_seconds * (2 ** attempt)
            log(f"API call failed ({exc}); retrying in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise last_error if last_error else RuntimeError("API call failed without an exception")


def is_bad_request(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 400


def chat_messages(instruction: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]


def extract_text(response: Any) -> str:
    choice = response.choices[0]
    content = choice.message.content
    if isinstance(content, str):
        return clean_response(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
        return clean_response("".join(parts))
    return ""


def label_from_logprob_token(token: str) -> Optional[str]:
    normalized = token.strip().upper()
    return normalized if normalized in {"A", "B", "C", "D"} else None


def extract_letter_logprobs(response: Any) -> Dict[str, float]:
    scores = {label: float("-inf") for label in ["A", "B", "C", "D"]}
    choice = response.choices[0]
    logprob_content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not logprob_content:
        return scores

    first_token = logprob_content[0]
    own_label = label_from_logprob_token(getattr(first_token, "token", ""))
    own_logprob = getattr(first_token, "logprob", None)
    if own_label and own_logprob is not None:
        scores[own_label] = max(scores[own_label], float(own_logprob))

    for candidate in getattr(first_token, "top_logprobs", []) or []:
        label = label_from_logprob_token(getattr(candidate, "token", ""))
        logprob = getattr(candidate, "logprob", None)
        if label and logprob is not None:
            scores[label] = max(scores[label], float(logprob))
    return scores


def usage_to_dict(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    out = {}
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out


def add_usage(total: Dict[str, int], usage: Dict[str, Any]) -> None:
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def evaluate_mcq_letter(
    item: Dict[str, Any],
    client: Any,
    args,
    usage_total: Dict[str, int],
) -> Dict[str, Any]:
    instr = format_standard_mcq(item)
    used_logprobs = True
    fallback_reason = None
    try:
        response = completion_with_retries(
            client,
            args,
            chat_messages(instr),
            **maybe_add_temperature(
                {
                    "max_completion_tokens": effective_letter_max_completion_tokens(args),
                    "logprobs": True,
                    "top_logprobs": LETTER_TOP_LOGPROBS,
                },
                args,
            ),
        )
    except Exception as exc:
        if not is_bad_request(exc):
            raise
        used_logprobs = False
        fallback_reason = str(exc)
        log(f"Letter logprob call rejected; falling back to generated letter parsing: {exc}")
        response = completion_with_retries(
            client,
            args,
            chat_messages(instr),
            **maybe_add_temperature(
                {"max_completion_tokens": effective_letter_max_completion_tokens(args)},
                args,
            ),
        )
    usage = usage_to_dict(response)
    add_usage(usage_total, usage)

    raw_answer = extract_text(response)
    scores = extract_letter_logprobs(response) if used_logprobs else None
    if scores:
        best = max(scores.items(), key=lambda kv: kv[1])[0]
    else:
        best = raw_answer
    if scores and not math.isfinite(scores[best]):
        parsed = parse_generated_letter(raw_answer)
        best = parsed if parsed else raw_answer
    if not scores:
        parsed = parse_generated_letter(raw_answer)
        best = parsed if parsed else raw_answer
    gold = parse_mcq_gold(item)

    return {
        "prompt": instr,
        "answer": best,
        "parsed_answer": best if best in {"A", "B", "C", "D"} else None,
        "gold": gold,
        "correct": (gold == best) if gold and best in {"A", "B", "C", "D"} else None,
        "label_logprobs": scores,
        "diagnostics": {
            "strategy": "letter",
            "strategy_description": MCQ_STRATEGY_DESCRIPTIONS["letter"],
            "raw_generation": raw_answer,
            "api_usage": usage,
            "top_logprobs": LETTER_TOP_LOGPROBS,
            "max_completion_tokens": effective_letter_max_completion_tokens(args),
            "used_logprobs": used_logprobs,
            "fallback_reason": fallback_reason,
        },
    }


def evaluate_mcq_generate_letter_and_text(
    item: Dict[str, Any],
    client: Any,
    args,
    usage_total: Dict[str, int],
) -> Dict[str, Any]:
    instr = format_generate_letter_and_text_mcq(item)
    response = completion_with_retries(
        client,
        args,
        chat_messages(instr),
        **maybe_add_temperature(
            {"max_completion_tokens": effective_generate_max_completion_tokens(args)},
            args,
        ),
    )
    usage = usage_to_dict(response)
    add_usage(usage_total, usage)

    raw_answer = extract_text(response)
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
            "strategy": "generate_letter_and_text",
            "strategy_description": MCQ_STRATEGY_DESCRIPTIONS["generate_letter_and_text"],
            "raw_generation": raw_answer,
            "parse_method": parse_method,
            "parse_confidence": parse_confidence,
            "max_output_tokens": effective_generate_max_completion_tokens(args),
            "api_usage": usage,
        },
    }


def evaluate_item_with_strategy(
    item: Dict[str, Any],
    strategy: str,
    client: Any,
    args,
    usage_total: Dict[str, int],
) -> Dict[str, Any]:
    if item["type"] != "mcq":
        raise ValueError(f"Closed-model runner currently supports only mcq items: {item.get('id')}")

    if strategy == "letter":
        core = evaluate_mcq_letter(item, client, args, usage_total)
    elif strategy == "generate_letter_and_text":
        core = evaluate_mcq_generate_letter_and_text(item, client, args, usage_total)
    else:
        raise ValueError(f"Unknown MCQ strategy: {strategy}")

    return {
        "id": item["id"],
        "lang": item.get("lang"),
        "type": item["type"],
        "category": item.get("category", item_category(item)),
        "bias": item.get("bias", item_bias(item)),
        **core,
    }


def evaluate_one_dataset_strategy(
    client: Any,
    items: List[Dict[str, Any]],
    args,
    dataset_path: Path,
    outdir: Path,
    run_timestamp: str,
    strategy_name: str,
    dataset_tag: Optional[str],
) -> Path:
    target_name = args.model
    out_path = build_output_path(target_name, outdir, run_timestamp, strategy_name, dataset_tag)
    if out_path.exists():
        log(f"Skipping {target_name} [{strategy_name}] (already exists): {out_path}")
        return out_path

    usage_total: Dict[str, int] = {}
    results: List[Dict[str, Any]] = []
    t_start = time.time()
    for index, item in enumerate(items, start=1):
        results.append(evaluate_item_with_strategy(item, strategy_name, client, args, usage_total))
        if args.progress_every and index % args.progress_every == 0:
            log(f"{dataset_path.stem} [{strategy_name}]: {index}/{len(items)} items")

    elapsed = round(time.time() - t_start, 2)
    payload = {
        "run_timestamp": run_timestamp,
        "target_name": target_name,
        "target_kind": "closed_model",
        "checkpoint": None,
        "hf_dir": None,
        "source": "openai_api",
        "dataset": str(dataset_path),
        "filters": {
            "type": args.types,
            "lang": args.langs,
            "category": args.categories,
            "bias": args.biases,
        },
        "generation_config": {
            "provider": "openai",
            "model": args.model,
            "temperature": args.temperature,
            "temperature_sent_to_api": None if model_uses_default_temperature(args.model) else args.temperature,
            "letter_max_completion_tokens": effective_letter_max_completion_tokens(args),
            "generate_max_completion_tokens": effective_generate_max_completion_tokens(args),
        },
        "diagnostic_config": {
            "mcq_strategy": strategy_name,
            "mcq_strategy_description": MCQ_STRATEGY_DESCRIPTIONS.get(strategy_name),
            "available_mcq_strategies": MCQ_STRATEGY_DESCRIPTIONS,
            "mcq_generate_max_new_tokens": args.max_output_tokens,
            "analysis_goal": (
                "Closed-model diagnostic baseline for comparing strict letter "
                "selection against generated letter-and-option-text answers."
            ),
        },
        "api_usage_total": usage_total,
        "seconds": elapsed,
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log(f"Done {target_name} [{strategy_name}]: {out_path} ({elapsed}s)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        "OpenAI closed-model evaluator for BiaBav MCQ letter and generate_letter_and_text strategies"
    )
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--outdir", default="results_closedmodels")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--mcq-strategies",
        default="letter,generate_letter_and_text",
        help="Comma-separated strategies: letter, generate_letter_and_text",
    )
    parser.add_argument("--type", dest="types", default="mcq")
    parser.add_argument("--lang", dest="langs", default=None)
    parser.add_argument("--category", dest="categories", default=None)
    parser.add_argument("--bias", dest="biases", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=MCQ_GENERATE_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--letter-max-output-tokens",
        type=int,
        default=None,
        help="Completion token budget for letter scoring. Defaults to 64 for GPT-5 models and 1 otherwise.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature. Omitted automatically for GPT-5-family models.",
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset item limit for smoke tests.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set your API key first, e.g. `export OPENAI_API_KEY=...`.")

    strategies = split_csv_arg(args.mcq_strategies) or []
    if not strategies:
        raise SystemExit("No MCQ strategies selected.")
    for strategy in strategies:
        if strategy not in MCQ_STRATEGY_DESCRIPTIONS:
            raise SystemExit(f"Unknown MCQ strategy: {strategy}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_paths = resolve_dataset_paths(args.dataset)
    log(f"Model selected: {args.model}")
    log(f"Dataset files selected: {len(dataset_paths)}")
    for dataset_path in dataset_paths:
        log(f"  - {dataset_path}")
    log(f"MCQ diagnostic strategies selected: {', '.join(strategies)}")

    prepared_datasets: List[Tuple[Path, List[Dict[str, Any]], Optional[str]]] = []
    multiple_datasets = len(dataset_paths) > 1
    for dataset_path in dataset_paths:
        items = load_dataset(dataset_path)
        items = filter_items(
            items,
            types=split_csv_arg(args.types),
            langs=split_csv_arg(args.langs),
            categories=split_csv_arg(args.categories),
            biases=split_csv_arg(args.biases),
        )
        if args.limit is not None:
            items = items[: args.limit]
        log(f"Dataset items after filtering [{dataset_path}]: {len(items)}")
        if not items:
            continue
        dataset_tag = dataset_path.stem if multiple_datasets else None
        prepared_datasets.append((dataset_path, items, dataset_tag))

    if not prepared_datasets:
        raise SystemExit("No dataset items left after filtering across all selected datasets.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install the OpenAI Python SDK with `pip install openai`."
        ) from exc

    client = OpenAI()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for dataset_path, items, dataset_tag in prepared_datasets:
        for strategy in strategies:
            evaluate_one_dataset_strategy(
                client=client,
                items=items,
                args=args,
                dataset_path=dataset_path,
                outdir=outdir,
                run_timestamp=run_timestamp,
                strategy_name=strategy,
                dataset_tag=dataset_tag,
            )

    log("All selected closed-model evaluations finished.")


if __name__ == "__main__":
    main()
