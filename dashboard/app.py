from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import DEFAULT_INPUTS, PRESET_INPUTS, load_payloads, load_records, resolve_inputs


st.set_page_config(page_title="Checkpoint Evaluation Dashboard", layout="wide")


DISCOVERY_SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "ckpt_venv",
    "dashboard",
    "dataset",
    "paper",
}


@st.cache_data(show_spinner=False)
def load_data(
    input_text: str,
    uploaded_payloads: Tuple[Tuple[str, str], ...],
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    raw_inputs = [part.strip() for part in input_text.split(",") if part.strip()]
    files = resolve_inputs(raw_inputs)

    file_payload_df, file_result_df = load_records(files)

    upload_payload_rows: List[Tuple[str, dict[str, Any]]] = []
    for filename, content in uploaded_payloads:
        payload = json.loads(content)
        upload_payload_rows.append((f"uploaded:{filename}", payload))

    upload_payload_df, upload_result_df = load_payloads(upload_payload_rows)

    payload_df = pd.concat([file_payload_df, upload_payload_df], ignore_index=True)
    result_df = pd.concat([file_result_df, upload_result_df], ignore_index=True)
    loaded_files = [str(path) for path in files] + [source for source, _ in upload_payload_rows]

    return payload_df, result_df, loaded_files


@st.cache_data(show_spinner=False)
def discover_json_folders(root_text: str = ".") -> List[tuple[str, int]]:
    root = Path(root_text)
    if not root.exists():
        return []

    discovered: List[tuple[str, int]] = []
    seen: set[Path] = set()

    def add_if_json_folder(path: Path) -> None:
        try:
            json_count = sum(
                1
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() == ".json"
            )
        except OSError:
            return
        if json_count == 0:
            return

        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        discovered.append((path.relative_to(root).as_posix(), json_count))

    try:
        top_level_dirs = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir()
                and path.name not in DISCOVERY_SKIP_DIRS
                and not path.name.startswith(".")
            ),
            key=lambda path: path.name.lower(),
        )
    except OSError:
        return []

    for path in top_level_dirs:
        add_if_json_folder(path)

        try:
            child_dirs = sorted(
                (
                    child
                    for child in path.iterdir()
                    if child.is_dir()
                    and child.name not in DISCOVERY_SKIP_DIRS
                    and not child.name.startswith(".")
                ),
                key=lambda child: child.name.lower(),
            )
        except OSError:
            continue

        for child in child_dirs:
            add_if_json_folder(child)

    return discovered


def apply_filters(
    df: pd.DataFrame,
    targets: List[str],
    runs: List[str],
    strategies: List[str],
    datasets: List[str],
    langs: List[str],
    types: List[str],
    question_subsets: List[str],
    regex_text: str,
) -> pd.DataFrame:
    out = df.copy()

    if targets:
        out = out[out["target_name"].isin(targets)]
    if runs:
        out = out[out["run_timestamp"].isin(runs)]
    if strategies:
        out = out[out["strategy"].isin(strategies)]
    if datasets:
        out = out[out["dataset_name"].isin(datasets)]
    if langs:
        out = out[out["lang"].isin(langs)]
    if types:
        out = out[out["type"].isin(types)]
    if question_subsets:
        out = out[out["question_subset"].isin(question_subsets)]
    if regex_text.strip():
        try:
            rx = re.compile(regex_text.strip())
            out = out[out["target_name"].fillna("").map(lambda x: bool(rx.search(x)))]
        except re.error as exc:
            st.sidebar.error(f"Invalid regex: {exc}")

    return out


def display_comparison_label(
    df: pd.DataFrame,
    split_runs: bool,
    split_strategies: bool,
) -> pd.DataFrame:
    if not split_runs and not split_strategies:
        return df

    out = df.copy()
    out["base_target_name"] = out["target_name"]

    if split_runs and split_strategies and "target_run_strategy_name" in out:
        out["target_name"] = out["target_run_strategy_name"]
    elif split_runs and "target_run_name" in out:
        out["target_name"] = out["target_run_name"]
    elif split_strategies and "target_strategy_name" in out:
        out["target_name"] = out["target_strategy_name"]

    return out


def target_sort_key(target_name: str, checkpoint_step: Any = None) -> tuple[int, int, str]:
    if pd.notna(checkpoint_step):
        return (0, int(checkpoint_step), str(target_name))

    match = re.search(r"checkpoint(\d+)", str(target_name))
    if match:
        return (0, int(match.group(1)), str(target_name))

    return (1, 0, str(target_name).lower())


def sorted_targets(df: pd.DataFrame) -> List[str]:
    target_steps = (
        df[["target_name", "checkpoint_step"]]
        .dropna(subset=["target_name"])
        .drop_duplicates()
        .groupby("target_name", dropna=True)["checkpoint_step"]
        .min()
        .to_dict()
    )
    return sorted(target_steps, key=lambda target: target_sort_key(target, target_steps[target]))


def sort_by_question_and_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_target_sort"] = out.apply(
        lambda row: target_sort_key(row["target_name"], row.get("checkpoint_step")),
        axis=1,
    )
    return out.sort_values(["id", "_target_sort"]).drop(columns=["_target_sort"])


def wrong_answer_label(row: pd.Series) -> str:
    if bool(row.get("is_correct")):
        return ""

    answer = row.get("answer")
    gold = row.get("gold")
    label = f"{answer}/{gold}"

    gap = row.get("answer_gold_gap")
    if pd.notna(gap):
        label = f"{label} {float(gap):+.2f}"

    return label


def question_accuracy_chart(df: pd.DataFrame) -> None:
    chart_df = df[df["correct"].notna()].copy()
    if chart_df.empty:
        return

    chart_df["wrong_label"] = chart_df.apply(wrong_answer_label, axis=1)
    chart_df["correct_value"] = chart_df["is_correct"].astype(int)

    question_ids = sorted(chart_df["id"].dropna().unique().tolist())
    target_names = sorted_targets(chart_df)
    if not question_ids or not target_names:
        return

    z = (
        chart_df.pivot_table(
            index="target_name",
            columns="id",
            values="correct_value",
            aggfunc="max",
        )
        .reindex(index=target_names, columns=question_ids)
    )
    text = (
        chart_df.pivot_table(
            index="target_name",
            columns="id",
            values="wrong_label",
            aggfunc="first",
        )
        .reindex(index=target_names, columns=question_ids)
        .fillna("")
    )
    hover = (
        chart_df.assign(
            hover_text=chart_df.apply(
                lambda row: (
                    f"Target: {row['target_name']}<br>"
                    f"Question: {row['id']}<br>"
                    f"Answer: {row.get('answer')}<br>"
                    f"Gold: {row.get('gold')}<br>"
                    f"Correct: {bool(row.get('is_correct'))}<br>"
                    f"Answer score: {row.get('answer_score')}<br>"
                    f"Gold score: {row.get('gold_score')}<br>"
                    f"Answer - gold score: {row.get('answer_gold_gap')}<br>"
                    f"Gold rank: {row.get('gold_rank')}"
                ),
                axis=1,
            )
        )
        .pivot_table(index="target_name", columns="id", values="hover_text", aggfunc="first")
        .reindex(index=target_names, columns=question_ids)
        .fillna("")
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z.to_numpy(),
            x=question_ids,
            y=target_names,
            text=text.to_numpy(),
            customdata=hover.to_numpy(),
            texttemplate="%{text}",
            hovertemplate="%{customdata}<extra></extra>",
            colorscale=[
                [0.0, "#ff4b4b"],
                [0.5, "#ff4b4b"],
                [0.5, "#49c285"],
                [1.0, "#49c285"],
            ],
            colorbar=dict(
                title="Correct",
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Wrong", "Correct"],
            ),
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(
        title="Correctness by Question and Target",
        xaxis_title="Question ID",
        yaxis_title="Target",
        xaxis_tickangle=-45,
        height=max(360, 70 + 35 * len(target_names)),
        margin=dict(l=20, r=20, t=60, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)


def metric_row(df: pd.DataFrame) -> None:
    total_models = df["target_name"].nunique()
    total_datasets = df["dataset_name"].nunique()
    total_questions = df["id"].nunique()
    scored = df["correct"].notna().sum()
    correct = df["is_correct"].sum() if "is_correct" in df else 0
    accuracy = (correct / scored) * 100.0 if scored else 0.0

    cols = st.columns(5)
    cols[0].metric("Targets", total_models)
    cols[1].metric("Datasets", total_datasets)
    cols[2].metric("Questions", total_questions)
    cols[3].metric("Scored Items", scored)
    cols[4].metric("Accuracy", f"{accuracy:.2f}%")


def overview_tab(df: pd.DataFrame) -> None:
    metric_row(df)

    scored_df = df[df["correct"].notna()].copy()
    if scored_df.empty:
        st.info("No scored rows in the current selection.")
        return

    model_acc = (
        scored_df.groupby("target_name", as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
        .sort_values("accuracy", ascending=False)
    )
    model_acc["accuracy_pct"] = model_acc["accuracy"] * 100.0

    dataset_order_df = (
        scored_df.groupby("dataset_name", as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
        .sort_values("accuracy", ascending=False)
    )
    dataset_order = dataset_order_df["dataset_name"].tolist()

    dataset_acc = (
        scored_df.groupby(["dataset_name", "target_name"], as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
    )
    dataset_acc["accuracy_pct"] = dataset_acc["accuracy"] * 100.0

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            model_acc,
            x="target_name",
            y="accuracy_pct",
            title="Accuracy by Target",
            labels={"target_name": "Target", "accuracy_pct": "Accuracy (%)"},
        )
        fig.update_layout(xaxis_tickangle=-40, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            dataset_acc,
            x="dataset_name",
            y="accuracy_pct",
            color="target_name",
            barmode="group",
            category_orders={
                "dataset_name": dataset_order,
                "target_name": sorted_targets(scored_df),
            },
            title="Accuracy by Dataset and Target",
            labels={
                "dataset_name": "Dataset",
                "target_name": "Target",
                "accuracy_pct": "Accuracy (%)",
            },
        )
        fig.update_layout(xaxis_tickangle=-30, height=420)
        st.plotly_chart(fig, use_container_width=True)

    choice_df = (
        df[df["answer"].notna()]
        .groupby(["target_name", "answer"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if not choice_df.empty:
        fig = px.bar(
            choice_df,
            x="target_name",
            y="count",
            color="answer",
            title="Choice Usage by Target",
            labels={"target_name": "Target", "count": "Answers"},
        )
        fig.update_layout(xaxis_tickangle=-40, height=460)
        st.plotly_chart(fig, use_container_width=True)


def compare_tab(df: pd.DataFrame) -> None:
    scored_df = df[df["correct"].notna()].copy()
    if scored_df.empty:
        st.info("No scored rows available for comparison.")
        return

    subset_order = ["General (GEN)", "Grounded (GRD)"]
    subset_df = scored_df[scored_df["question_subset"].isin(subset_order)].copy()
    if not subset_df.empty:
        st.subheader("General vs Grounded Questions")
        subset_view = st.radio(
            "Question-type comparison",
            ["By model", "By model and language", "Appendix table"],
            horizontal=True,
        )

        if subset_view == "By model":
            question_type_df = (
                subset_df.groupby(
                    ["target_name", "question_subset"],
                    as_index=False,
                )["is_correct"]
                .mean()
                .rename(columns={"is_correct": "accuracy"})
            )
            question_type_df["accuracy_pct"] = question_type_df["accuracy"] * 100.0
            fig = px.bar(
                question_type_df,
                x="target_name",
                y="accuracy_pct",
                color="question_subset",
                barmode="group",
                category_orders={
                    "target_name": sorted_targets(subset_df),
                    "question_subset": subset_order,
                },
                title="Accuracy on General vs Grounded Questions",
                labels={
                    "target_name": "Target",
                    "accuracy_pct": "Accuracy (%)",
                    "question_subset": "Question subset",
                },
            )
            fig.update_layout(xaxis_tickangle=-40, height=460)
            st.plotly_chart(fig, use_container_width=True)

        elif subset_view == "By model and language":
            question_type_lang_df = (
                subset_df.groupby(
                    ["target_name", "lang", "question_subset"],
                    as_index=False,
                )["is_correct"]
                .mean()
                .rename(columns={"is_correct": "accuracy"})
            )
            question_type_lang_df["accuracy_pct"] = question_type_lang_df["accuracy"] * 100.0
            fig = px.bar(
                question_type_lang_df,
                x="target_name",
                y="accuracy_pct",
                color="question_subset",
                facet_col="lang",
                barmode="group",
                category_orders={
                    "target_name": sorted_targets(subset_df),
                    "question_subset": subset_order,
                },
                title="Accuracy on General vs Grounded Questions by Language",
                labels={
                    "target_name": "Target",
                    "accuracy_pct": "Accuracy (%)",
                    "question_subset": "Question subset",
                    "lang": "Language",
                },
            )
            fig.update_layout(xaxis_tickangle=-40, height=520)
            st.plotly_chart(fig, use_container_width=True)

        else:
            appendix_df = (
                subset_df.groupby(
                    ["target_name", "dataset_name", "lang", "question_subset"],
                    as_index=False,
                )
                .agg(
                    accuracy=("is_correct", "mean"),
                    correct=("is_correct", "sum"),
                    total=("is_correct", "size"),
                )
                .sort_values(["target_name", "dataset_name", "lang", "question_subset"])
            )
            appendix_df["accuracy_pct"] = appendix_df["accuracy"] * 100.0
            st.dataframe(
                appendix_df[
                    [
                        "target_name",
                        "dataset_name",
                        "lang",
                        "question_subset",
                        "correct",
                        "total",
                        "accuracy_pct",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    heatmap_df = (
        scored_df.groupby(["target_name", "dataset_name"], as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
    )
    if heatmap_df.empty:
        st.info("Not enough data for comparison.")
        return

    pivot = heatmap_df.pivot(index="target_name", columns="dataset_name", values="accuracy")
    ordered_targets = [target for target in sorted_targets(scored_df) if target in pivot.index]
    pivot = pivot.reindex(ordered_targets)
    fig = px.imshow(
        pivot * 100.0,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels={"x": "Dataset", "y": "Target", "color": "Accuracy (%)"},
        title="Accuracy Heatmap",
    )
    fig.update_layout(height=max(420, 30 * len(pivot.index) + 180))
    st.plotly_chart(fig, use_container_width=True)

    lang_df = (
        scored_df.groupby(["target_name", "lang"], as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
    )
    lang_df["accuracy_pct"] = lang_df["accuracy"] * 100.0
    fig = px.bar(
        lang_df,
        x="target_name",
        y="accuracy_pct",
        color="lang",
        barmode="group",
        category_orders={"target_name": sorted_targets(scored_df)},
        title="Per-Language Accuracy by Target",
        labels={"target_name": "Target", "accuracy_pct": "Accuracy (%)"},
    )
    fig.update_layout(xaxis_tickangle=-40, height=460)
    st.plotly_chart(fig, use_container_width=True)

    if "strategy" in scored_df and scored_df["strategy"].nunique() > 1:
        strategy_df = (
            scored_df.groupby(["target_name", "dataset_name", "strategy"], as_index=False)["is_correct"]
            .mean()
            .rename(columns={"is_correct": "accuracy"})
        )
        strategy_df["accuracy_pct"] = strategy_df["accuracy"] * 100.0
        dataset_order = (
            scored_df.groupby("dataset_name")["is_correct"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        strategy_order = sorted(strategy_df["strategy"].dropna().unique().tolist())
        target_order = sorted_targets(scored_df)

        facet_args = {}
        if len(target_order) <= 4:
            facet_args = {"facet_col": "target_name", "facet_col_wrap": 2}

        fig = px.bar(
            strategy_df,
            x="dataset_name",
            y="accuracy_pct",
            color="strategy",
            barmode="group",
            category_orders={
                "dataset_name": dataset_order,
                "strategy": strategy_order,
                "target_name": target_order,
            },
            title="Strategy Accuracy by Dataset",
            labels={
                "dataset_name": "Dataset",
                "accuracy_pct": "Accuracy (%)",
                "strategy": "Strategy",
                "target_name": "Target",
            },
            hover_data=["target_name"],
            **facet_args,
        )
        fig.update_layout(xaxis_tickangle=-35, height=520 if facet_args else 460)
        st.plotly_chart(fig, use_container_width=True)


def checkpoint_tab(df: pd.DataFrame) -> None:
    scored_df = df[df["correct"].notna()].copy()
    ckpt_df = scored_df[scored_df["checkpoint_step"].notna()].copy()
    if ckpt_df.empty:
        st.info("No checkpoint-style targets in the current selection.")
        return

    ckpt_df["checkpoint_step"] = ckpt_df["checkpoint_step"].astype(int)
    agg = (
        ckpt_df.groupby(["target_name", "checkpoint_step", "dataset_name"], as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
    )
    agg["accuracy_pct"] = agg["accuracy"] * 100.0

    fig = px.line(
        agg,
        x="checkpoint_step",
        y="accuracy_pct",
        color="target_name",
        line_dash="dataset_name",
        markers=True,
        title="Checkpoint Accuracy Trends",
        labels={"checkpoint_step": "Checkpoint Step", "accuracy_pct": "Accuracy (%)"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def question_tab(df: pd.DataFrame) -> None:
    available_datasets = sorted(df["dataset_name"].dropna().unique().tolist())
    if not available_datasets:
        st.info("No dataset rows available.")
        return

    dataset_name = st.selectbox("Dataset", available_datasets, key="question_dataset")
    dataset_df = df[df["dataset_name"] == dataset_name].copy()

    available_langs = ["all"] + sorted(dataset_df["lang"].dropna().unique().tolist())
    lang_choice = st.selectbox("Language", available_langs, index=0, key="question_lang")
    if lang_choice != "all":
        dataset_df = dataset_df[dataset_df["lang"] == lang_choice]

    question_mode = st.radio("Question Scope", ["All questions", "Single question"], horizontal=True)

    if question_mode == "Single question":
        question_df = (
            dataset_df[["id", "prompt", "lang", "type", "category", "bias"]]
            .drop_duplicates()
            .sort_values(["lang", "id"])
        )
        labels = {
            row["id"]: f"{row['id']} [{row['lang']}] {str(row['prompt']).splitlines()[0][:120]}"
            for _, row in question_df.iterrows()
        }
        selected_id = st.selectbox("Question", list(labels.keys()), format_func=lambda x: labels[x])
        dataset_df = dataset_df[dataset_df["id"] == selected_id]

    if question_mode == "All questions":
        question_accuracy_chart(dataset_df)

    table_df = dataset_df[
        [
            "target_name",
            "checkpoint_step",
            "lang",
            "id",
            "answer",
            "gold",
            "correct",
            "best_label",
            "best_score",
            "answer_score",
            "gold_score",
            "answer_gold_gap",
            "gold_rank",
            "runner_up_label",
            "runner_up_score",
            "winning_margin",
        ]
    ]
    table_df = sort_by_question_and_target(table_df)
    table_df = table_df.drop(columns=["checkpoint_step"])
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    if question_mode == "Single question":
        target_options = sorted_targets(dataset_df)
        if target_options:
            default_targets = target_options if len(target_options) <= 5 else target_options[:5]
            selected_targets = st.multiselect(
                "Inspect Target Scores",
                target_options,
                default=default_targets,
            )
            item_rows = dataset_df[dataset_df["target_name"].isin(selected_targets)]
            if not item_rows.empty:
                st.markdown(f"**Prompt**: {item_rows.iloc[0]['prompt']}")

                score_rows = []
                for _, row in item_rows.iterrows():
                    label_logprobs = row.get("label_logprobs")
                    if not isinstance(label_logprobs, dict) or not label_logprobs:
                        continue
                    for label, score in label_logprobs.items():
                        score_rows.append(
                            {
                                "target_name": row["target_name"],
                                "label": str(label),
                                "score": float(score),
                                "answer": row.get("answer"),
                                "gold": row.get("gold"),
                                "correct": row.get("correct"),
                            }
                        )

                if score_rows:
                    score_df = pd.DataFrame(score_rows)
                    fig = px.bar(
                        score_df,
                        x="label",
                        y="score",
                        color="target_name",
                        barmode="group",
                        category_orders={"target_name": selected_targets},
                        title="Label Scores by Target",
                        labels={
                            "label": "Label",
                            "score": "Logprob Score",
                            "target_name": "Target",
                        },
                        hover_data=["answer", "gold", "correct"],
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No label_logprobs stored for the selected target(s).")


def raw_tab(df: pd.DataFrame) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True)


def data_source_controls() -> tuple[str, Tuple[Tuple[str, str], ...]]:
    st.sidebar.header("Data Sources")

    discovered_folders = discover_json_folders()
    discovered_paths = [folder for folder, _ in discovered_folders]
    if "results/allmodels" in discovered_paths:
        default_folders = ["results/allmodels"]
    else:
        default_folders = [
            folder
            for folder in discovered_paths
            if Path(folder).name.startswith("results")
        ][:1]

    selected_folders = st.sidebar.multiselect(
        "Result folders",
        discovered_paths,
        default=default_folders,
        format_func=lambda folder: (
            f"{folder} ({dict(discovered_folders).get(folder, 0)} JSON files)"
        ),
        help="Discovered folders near the top of this repository. Click folders to add them to the comparison.",
    )

    with st.sidebar.expander("Legacy quick folders", expanded=False):
        preset_labels = list(PRESET_INPUTS.keys())
        selected_presets = st.multiselect(
            "Add preset folders",
            preset_labels,
            default=[],
            help="Older named presets kept for compatibility.",
        )

    preset_paths: List[str] = []
    for label in selected_presets:
        preset_paths.extend(PRESET_INPUTS[label])

    extra_paths = st.sidebar.text_area(
        "Additional folders or JSON files",
        value="",
        placeholder="logs, /path/to/results, one_file.json",
        help="Comma-separated paths. Folders are scanned for *.json files.",
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload JSON result files",
        type="json",
        accept_multiple_files=True,
        help="Uploaded files are combined with the selected folders for this session.",
    )

    uploaded_payloads: List[Tuple[str, str]] = []
    for uploaded_file in uploaded_files:
        uploaded_payloads.append(
            (uploaded_file.name, uploaded_file.getvalue().decode("utf-8"))
        )

    path_parts = (
        selected_folders
        + preset_paths
        + [part.strip() for part in extra_paths.split(",") if part.strip()]
    )
    input_text = ",".join(path_parts or DEFAULT_INPUTS)
    return input_text, tuple(uploaded_payloads)


def render_load_summary(payload_df: pd.DataFrame, loaded_files: List[str]) -> None:
    st.sidebar.success(f"Loaded {len(loaded_files)} JSON files")

    if payload_df.empty:
        return

    total_items = int(payload_df["results_count"].fillna(0).sum())
    st.sidebar.caption(
        f"{payload_df['target_name'].nunique()} targets, "
        f"{payload_df['dataset_name'].nunique()} datasets, "
        f"{payload_df['strategy'].nunique()} strategies, "
        f"{total_items:,} result rows"
    )

    def source_folder(value: str) -> str:
        if value.startswith("uploaded:"):
            return "uploaded files"
        return str(Path(value).parent)

    source_summary = (
        payload_df.assign(source_folder=payload_df["file"].map(source_folder))
        .groupby("source_folder", as_index=False)
        .agg(
            files=("file", "nunique"),
            targets=("target_name", "nunique"),
            strategies=("strategy", "nunique"),
            rows=("results_count", "sum"),
        )
        .sort_values(["files", "source_folder"], ascending=[False, True])
    )
    with st.sidebar.expander("Loaded source summary", expanded=False):
        st.dataframe(source_summary, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Evaluation Results Dashboard")
    st.caption(
        "Explore checkpoint and model outputs across datasets, languages, questions, and answer-score behavior."
    )

    input_text, uploaded_payloads = data_source_controls()

    try:
        payload_df, result_df, loaded_files = load_data(input_text, uploaded_payloads)
    except json.JSONDecodeError as exc:
        st.error(f"Could not read one of the selected JSON files: {exc}")
        return

    if result_df.empty:
        st.warning("No evaluation JSON files were loaded from the selected inputs.")
        return

    render_load_summary(payload_df, loaded_files)

    available_targets = sorted_targets(result_df)
    available_runs = sorted(result_df["run_timestamp"].dropna().unique().tolist(), reverse=True)
    available_strategies = sorted(result_df["strategy"].dropna().unique().tolist())
    available_datasets = sorted(result_df["dataset_name"].dropna().unique().tolist())
    available_langs = sorted(result_df["lang"].dropna().unique().tolist())
    available_types = sorted(result_df["type"].dropna().unique().tolist())
    available_question_subsets = sorted(result_df["question_subset"].dropna().unique().tolist())

    st.sidebar.header("Filters")
    with st.sidebar.expander("Target filters", expanded=True):
        regex_text = st.text_input(
            "Target regex",
            value="",
            help="Useful for checkpoints like checkpoint(\\d+).",
        )
        selected_targets = st.multiselect("Target(s)", available_targets)
        selected_runs = st.multiselect(
            "Run timestamp(s)",
            available_runs,
            help="Select one run, multiple runs, or leave empty to include all runs.",
        )
        selected_strategies = st.multiselect(
            "Strategy(s)",
            available_strategies,
            help="Select evaluation strategies such as letter, option_text, letter_shuffled, or generate.",
        )
        split_runs = st.toggle(
            "Compare selected runs separately",
            value=True,
            help="When enabled, the same target from different timestamps is shown as separate series.",
        )
        split_strategies = st.toggle(
            "Compare selected strategies separately",
            value=any(strategy != "standard" for strategy in available_strategies),
            help="When enabled, the same target with different evaluation strategies is shown as separate series.",
        )

    with st.sidebar.expander("Dataset filters", expanded=True):
        selected_datasets = st.multiselect("Dataset(s)", available_datasets)
        selected_langs = st.multiselect("Language(s)", available_langs)
        selected_types = st.multiselect("Item type(s)", available_types)
        selected_question_subsets = st.multiselect(
            "Question subset(s)",
            available_question_subsets,
            help="GEN covers question numbers 1-10 in each category/language; GRD covers 11-25.",
        )

    filtered_df = apply_filters(
        result_df,
        selected_targets,
        selected_runs,
        selected_strategies,
        selected_datasets,
        selected_langs,
        selected_types,
        selected_question_subsets,
        regex_text,
    )

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    display_df = display_comparison_label(filtered_df, split_runs, split_strategies)

    with st.expander("Loaded Files", expanded=False):
        st.write(loaded_files)

    tabs = st.tabs(
        [
            "Overview",
            "Compare",
            "Checkpoint Trends",
            "Question Explorer",
            "Raw Data",
        ]
    )

    with tabs[0]:
        overview_tab(display_df)
    with tabs[1]:
        compare_tab(display_df)
    with tabs[2]:
        checkpoint_tab(display_df)
    with tabs[3]:
        question_tab(display_df)
    with tabs[4]:
        raw_tab(display_df)


if __name__ == "__main__":
    main()
