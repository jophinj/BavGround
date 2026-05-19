# Dashboard README

This folder contains a Streamlit dashboard for inspecting BavGround result
files.

The dashboard reads existing JSON outputs. It does not run model inference.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface. |
| `data.py` | JSON loading, normalization, and metric helpers. |
| `requirements.txt` | Python dependencies for the dashboard. |

## Run

From the repository root:

```bash
python -m venv .venv-bavground
source .venv-bavground/bin/activate
python -m pip install --upgrade pip
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Then open the URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Recommended Inputs

In the sidebar, enter one or more comma-separated folders:

```text
results/allmodels,results/closedmodels,results/genba_checkpoints
```

You can also upload individual JSON files through the browser.

## Dashboard Views

The dashboard supports:

- filtering by model or checkpoint
- filtering by run timestamp
- filtering by evaluation strategy
- filtering by dataset/category
- filtering by language
- separating GEN and GRD subsets
- aggregate accuracy tables
- category and language breakdowns
- checkpoint trajectories
- answer-choice distributions
- per-item prompts, predictions, gold labels, and diagnostics

## Result Format

The app expects JSON files with top-level metadata and a `results` list, as
produced by the evaluation scripts in `code/`.

Important fields:

- top-level `target_name`
- top-level `run_timestamp`
- top-level `dataset`
- top-level `diagnostic_config.mcq_strategy`
- per-item `id`
- per-item `lang`
- per-item `answer`
- per-item `gold`
- per-item `correct`
