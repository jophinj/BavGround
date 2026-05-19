# Results README

This folder contains the item-level evaluation outputs used in the BavGround
paper. Results are grouped by experiment type so reviewers can navigate the
artifact without scanning thousands of JSON files at the repository root.

## Folder Layout

```text
results/
  allmodels/
    Open-weight model outputs.

  closedmodels/
    Closed-model reference outputs.

  genba_checkpoints/
    GENBA-10B checkpoint trajectory outputs.
```

Each subfolder has its own README with filename patterns, schema notes, and
paper usage.

## Summary

| Folder | Files | Rows | Targets | Strategies | Paper use |
| --- | ---: | ---: | ---: | ---: | --- |
| `allmodels/` | 1,440 | 111,240 | 15 models | 12 | Main open-weight comparison and protocol analysis |
| `closedmodels/` | 16 | 1,236 | 1 model | 2 | Closed-model reference |
| `genba_checkpoints/` | 8,160 | 630,360 | 85 checkpoints | 12 | Continued-pretraining checkpoint trajectories |

All result files are JSON files with top-level run metadata and a `results`
list containing item-level predictions and diagnostics.

## Common JSON Fields

Top-level fields generally include:

- `run_timestamp`: anonymized run label, such as `run_open_models_1`
- `target_name`: model or checkpoint name
- `target_kind`: `named_model`, `closed_model`, or `checkpoint`
- `dataset`: source dataset file
- `filters`: dataset filters applied for the run
- `generation_config`: decoding and scoring configuration
- `diagnostic_config`: evaluation protocol metadata
- `results`: item-level outputs

Common item-level fields include:

- `id`
- `lang`
- `type`
- `category`
- `prompt`
- `answer`
- `parsed_answer`
- `gold`
- `correct`
- `label_logprobs`
- `diagnostics`

## Protocols

The open-weight and checkpoint folders contain:

- `letter`
- `letter_shuffled`
- `option_text`
- `option_text_avg`
- `generate_letter`
- `generate_option_text`
- `generate_letter_and_text`
- `semantic_embed_generated_answer`
- `semantic_embed_option`
- `semantic_embed_contextual_option`
- `embed_contextual_option`
- `embed_isolated_option`

The closed-model folder contains:

- `letter`
- `generate_letter_and_text`

## Useful Commands

Bootstrap confidence intervals:

```bash
python code/bootstrap_biabav_ci.py --results-allmodels results/allmodels
```

Label-prior diagnostics:

```bash
python code/label_prior_diagnostic.py --results-allmodels results/allmodels
```

Checkpoint plots:

```bash
python code/plot_genba_checkpoint_trajectories.py \
  --results-dir results/genba_checkpoints \
  --out-dir images
```

Dashboard:

```bash
python -m venv .venv-bavground
source .venv-bavground/bin/activate
python -m pip install --upgrade pip
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Recommended dashboard input:

```text
results/allmodels,results/closedmodels,results/genba_checkpoints
```
