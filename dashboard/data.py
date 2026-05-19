from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


DEFAULT_INPUTS = ["results_allmodels"]
PRESET_INPUTS = {
    "Open-weight model results": ["results_allmodels"],
    "Closed-model reference": ["results_closedmodels"],
    "GENBA checkpoint results": ["results_genba_checkpoints"],
    "All paper result folders": [
        "results_allmodels",
        "results_closedmodels",
        "results_genba_checkpoints",
    ],
}


def resolve_inputs(inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    seen: Set[Path] = set()

    for raw in inputs:
        text = raw.strip()
        if not text:
            continue

        path = Path(text)
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.exists() or candidate.suffix.lower() != ".json":
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)

    return sorted(files)


def parse_dataset_name(dataset_value: Optional[str]) -> str:
    if not dataset_value:
        return "unknown"
    return Path(dataset_value).stem


def parse_checkpoint_step(target_name: Optional[str], source: Optional[str]) -> Optional[int]:
    for text in (target_name or "", source or ""):
        match = re.search(r"checkpoint(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def parse_question_number(question_id: Optional[str]) -> Optional[int]:
    if not question_id:
        return None

    match = re.search(r"_(\d+)$", str(question_id))
    if not match:
        return None

    return int(match.group(1))


def infer_question_subset(question_id: Optional[str]) -> str:
    question_number = parse_question_number(question_id)
    if question_number is None:
        return "unknown"
    if 1 <= question_number <= 10:
        return "General (GEN)"
    if 11 <= question_number <= 25:
        return "Grounded (GRD)"
    return "unknown"


def infer_target_name(payload: Dict[str, Any], source_name: str) -> str:
    return (
        payload.get("target_name")
        or payload.get("checkpoint")
        or Path(source_name).stem
    )


def infer_run_timestamp(payload: Dict[str, Any], source_name: str) -> str:
    if payload.get("run_timestamp"):
        return str(payload["run_timestamp"])

    match = re.search(r"_(\d{8}_\d{6})(?:\.json)?$", Path(source_name).name)
    if match:
        return match.group(1)

    return "unknown"


def make_target_run_name(target_name: str, run_timestamp: str) -> str:
    return f"{target_name} @ {run_timestamp}" if run_timestamp != "unknown" else target_name


def infer_strategy(payload: Dict[str, Any], source_name: str) -> Tuple[str, Optional[str]]:
    diagnostic_config = payload.get("diagnostic_config") or {}
    strategy = diagnostic_config.get("mcq_strategy")
    description = diagnostic_config.get("mcq_strategy_description")

    if not strategy:
        for item in payload.get("results", []):
            diagnostics = item.get("diagnostics") or {}
            if diagnostics.get("strategy"):
                strategy = diagnostics.get("strategy")
                description = diagnostics.get("strategy_description")
                break

    if not strategy:
        name = Path(source_name).stem
        match = re.search(
            r"_(letter_shuffled|option_text_avg|option_text|letter|generate)(?:_\d{8}_\d{6})?$",
            name,
        )
        if match:
            strategy = match.group(1)

    return str(strategy or "standard"), description


def make_target_strategy_name(target_name: str, strategy: str) -> str:
    return target_name if strategy == "standard" else f"{target_name} / {strategy}"


def make_target_run_strategy_name(target_name: str, run_timestamp: str, strategy: str) -> str:
    name = make_target_run_name(target_name, run_timestamp)
    return name if strategy == "standard" else f"{name} / {strategy}"


def compute_label_metrics(
    label_logprobs: Any,
    answer: Optional[str],
    gold: Optional[str],
) -> Dict[str, Any]:
    if not isinstance(label_logprobs, dict) or not label_logprobs:
        return {
            "best_label": None,
            "best_score": None,
            "runner_up_label": None,
            "runner_up_score": None,
            "winning_margin": None,
            "answer_score": None,
            "gold_score": None,
            "answer_gold_gap": None,
            "gold_rank": None,
            "choices_used": 0,
        }

    answer_scores = {str(label): float(score) for label, score in label_logprobs.items()}
    scored = sorted(
        answer_scores.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    best_label, best_score = scored[0]
    runner_up_label = None
    runner_up_score = None
    winning_margin = None

    if len(scored) > 1:
        runner_up_label, runner_up_score = scored[1]
        winning_margin = best_score - runner_up_score

    if answer is not None:
        answer = str(answer)
        if answer in answer_scores:
            best_score = answer_scores[answer]
            other_scores = [score for label, score in answer_scores.items() if label != answer]
            if other_scores:
                winning_margin = best_score - max(other_scores)
            ranked = sorted(answer_scores.items(), key=lambda kv: kv[1], reverse=True)
            for label, score in ranked:
                if label != answer:
                    runner_up_label = label
                    runner_up_score = score
                    break
            best_label = answer

    gold = str(gold) if gold is not None else None
    answer_score = answer_scores.get(answer) if answer is not None else None
    gold_score = answer_scores.get(gold) if gold is not None else None
    answer_gold_gap = None
    if answer_score is not None and gold_score is not None:
        answer_gold_gap = answer_score - gold_score

    gold_rank = None
    if gold is not None:
        for index, (label, _) in enumerate(scored, start=1):
            if label == gold:
                gold_rank = index
                break

    return {
        "best_label": best_label,
        "best_score": best_score,
        "runner_up_label": runner_up_label,
        "runner_up_score": runner_up_score,
        "winning_margin": winning_margin,
        "answer_score": answer_score,
        "gold_score": gold_score,
        "answer_gold_gap": answer_gold_gap,
        "gold_rank": gold_rank,
        "choices_used": len(label_logprobs),
    }


def load_records(files: Iterable[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    payloads: List[Tuple[str, Dict[str, Any]]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            payloads.append((str(path), json.load(f)))

    return load_payloads(payloads)


def load_payloads(payloads: Iterable[Tuple[str, Dict[str, Any]]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    payload_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []

    for source_name, payload in payloads:
        if not isinstance(payload, dict):
            continue

        dataset_name = parse_dataset_name(payload.get("dataset"))
        target_name = infer_target_name(payload, source_name)
        source = payload.get("source") or payload.get("hf_dir")
        checkpoint_step = parse_checkpoint_step(target_name, source)
        run_timestamp = infer_run_timestamp(payload, source_name)
        target_run_name = make_target_run_name(target_name, run_timestamp)
        strategy, strategy_description = infer_strategy(payload, source_name)
        target_strategy_name = make_target_strategy_name(target_name, strategy)
        target_run_strategy_name = make_target_run_strategy_name(
            target_name,
            run_timestamp,
            strategy,
        )
        filters = payload.get("filters") or {}

        payload_rows.append(
            {
                "file": source_name,
                "target_name": target_name,
                "target_run_name": target_run_name,
                "target_strategy_name": target_strategy_name,
                "target_run_strategy_name": target_run_strategy_name,
                "strategy": strategy,
                "strategy_description": strategy_description,
                "target_kind": payload.get("target_kind"),
                "dataset": payload.get("dataset"),
                "dataset_name": dataset_name,
                "run_timestamp": run_timestamp,
                "seconds": payload.get("seconds"),
                "source": source,
                "checkpoint": payload.get("checkpoint"),
                "checkpoint_step": checkpoint_step,
                "filter_type": filters.get("type"),
                "filter_lang": filters.get("lang"),
                "filter_category": filters.get("category"),
                "filter_bias": filters.get("bias"),
                "results_count": len(payload.get("results", [])),
            }
        )

        for item in payload.get("results", []):
            question_id = item.get("id")
            question_number = parse_question_number(question_id)
            label_metrics = compute_label_metrics(
                item.get("label_logprobs"),
                item.get("answer"),
                item.get("gold"),
            )
            result_rows.append(
                {
                    "file": source_name,
                    "target_name": target_name,
                    "target_run_name": target_run_name,
                    "target_strategy_name": target_strategy_name,
                    "target_run_strategy_name": target_run_strategy_name,
                    "strategy": strategy,
                    "strategy_description": strategy_description,
                    "target_kind": payload.get("target_kind"),
                    "dataset": payload.get("dataset"),
                    "dataset_name": dataset_name,
                    "run_timestamp": run_timestamp,
                    "checkpoint": payload.get("checkpoint"),
                    "checkpoint_step": checkpoint_step,
                    "id": question_id,
                    "question_number": question_number,
                    "question_subset": infer_question_subset(question_id),
                    "lang": item.get("lang"),
                    "type": item.get("type"),
                    "category": item.get("category"),
                    "bias": item.get("bias"),
                    "prompt": item.get("prompt"),
                    "answer": item.get("answer"),
                    "parsed_answer": item.get("parsed_answer"),
                    "gold": item.get("gold"),
                    "correct": item.get("correct"),
                    "label_logprobs": item.get("label_logprobs"),
                    "best_label": label_metrics["best_label"],
                    "best_score": label_metrics["best_score"],
                    "runner_up_label": label_metrics["runner_up_label"],
                    "runner_up_score": label_metrics["runner_up_score"],
                    "winning_margin": label_metrics["winning_margin"],
                    "answer_score": label_metrics["answer_score"],
                    "gold_score": label_metrics["gold_score"],
                    "answer_gold_gap": label_metrics["answer_gold_gap"],
                    "gold_rank": label_metrics["gold_rank"],
                    "choices_used": label_metrics["choices_used"],
                }
            )

    payload_df = pd.DataFrame(payload_rows)
    result_df = pd.DataFrame(result_rows)

    if not result_df.empty:
        result_df["is_correct"] = result_df["correct"].fillna(False).astype(bool)

    return payload_df, result_df
