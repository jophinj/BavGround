# Code README

This folder contains the evaluation and analysis scripts used for the BavGround
paper artifact.

## Main Runners

| Script | Purpose |
| --- | --- |
| `run_checkpoints_eval_v4.py` | Shared evaluation utilities: dataset loading, target resolution, prompting, model loading, and log-probability scoring helpers. |
| `run_checkpoints_eval_v4_genba_v2.py` | Main output-level evaluator for open-weight models and checkpoints. Supports letter scoring, shuffled letters, option-text scoring, and generated-answer protocols. |
| `run_checkpoints_eval_v4_genba_v2_embed_v3.py` | Evaluator for hidden-state alignment diagnostics: `embed_contextual_option` and `embed_isolated_option`. |
| `run_closed_model_eval_openai.py` | Closed-model API evaluator for `letter` and `generate_letter_and_text`. |

## Analysis Scripts

| Script | Purpose |
| --- | --- |
| `bootstrap_biabav_ci.py` | Computes paired nonparametric bootstrap intervals used in the paper. |
| `label_prior_diagnostic.py` | Computes answer-label prior diagnostics for canonical `letter` scoring. |
| `plot_genba_checkpoint_trajectories.py` | Produces checkpoint trajectory figures from `results/genba_checkpoints/`. |

## Job Scripts

| Script | Run type |
| --- | --- |
| `job_checkpoints_eval_v4_genba_v2.sh` | Open-weight model output-level protocols. |
| `job_checkpoints_eval_v4_genba_checkpoints_v2.sh` | GENBA checkpoint output-level protocols. |
| `job_checkpoints_eval_v4_genba_v2_embed_v3.sh` | Open-weight hidden-state embedding diagnostics. |
| `job_checkpoints_eval_v4_genba_v2_checkpoint_embed_v3.sh` | GENBA checkpoint hidden-state embedding diagnostics. |
| `job_checkpoints_eval_v4_genba_v2_semantic_embed.sh` | Open-weight external semantic-embedding diagnostics. |
| `job_checkpoints_eval_v4_genba_v2_semantic_embed_checkpoints.sh` | GENBA checkpoint external semantic-embedding diagnostics. |

The two semantic job scripts call
`run_checkpoints_eval_v4_genba_v2_semantic_embed.py`, which is included in this
folder.

## Example Analysis Commands

Bootstrap intervals:

```bash
python code/bootstrap_biabav_ci.py \
  --results-allmodels results/allmodels
```

Label-prior diagnostics:

```bash
python code/label_prior_diagnostic.py \
  --results-allmodels results/allmodels
```

Checkpoint plots:

```bash
python code/plot_genba_checkpoint_trajectories.py \
  --results-dir results/genba_checkpoints \
  --out-dir images
```

## Result JSON Shape

Evaluation runners write one JSON file per target, dataset/category, and
strategy. Each file contains:

- top-level run metadata: `run_timestamp`, `target_name`, `target_kind`,
  `dataset`, `filters`, `generation_config`, and `diagnostic_config`
- `results`: item-level outputs with fields such as `id`, `lang`, `type`,
  `category`, `prompt`, `answer`, `gold`, `correct`, `label_logprobs`, and
  strategy-specific diagnostics
