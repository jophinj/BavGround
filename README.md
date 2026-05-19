# BavGround Artifact

This repository contains the benchmark data, evaluation code, model outputs, and
interactive dashboard for the paper:

**BavGround: A Benchmark for Regional Cultural Grounding and Dialect Competence
in Bavarian**

BavGround evaluates Bavarian regional cultural grounding and dialect competence
across English, German, and Bavarian. The benchmark contains 206 source
multiple-choice questions across eight cultural domains. Each source question is
available in three languages, giving 618 evaluated instances.

## Artifact Map

```text
dataset/
  BavGround benchmark items, split by cultural category.

code/
  Evaluation runners and analysis scripts used to produce the reported tables,
  diagnostics, bootstrap intervals, and checkpoint plots.

results_allmodels/
  Item-level outputs for 15 open-weight instruction-tuned models across 12
  evaluation protocols.

results_closedmodels/
  Item-level outputs for the closed-model reference run.

results_genba_checkpoints/
  Item-level outputs for GENBA-10B continued-pretraining checkpoints across
  12 evaluation protocols.

dashboard/
  Streamlit dashboard for browsing result files interactively.
```

Each folder has its own README with schema details and reviewer-oriented usage
notes.

## Paper Connection

The artifact supports the paper sections as follows:

| Paper component | Artifact location |
| --- | --- |
| BavGround benchmark construction and category counts | `dataset/` |
| Open-weight model comparison | `results_allmodels/` |
| Closed-model reference | `results_closedmodels/` |
| Protocol sensitivity analysis | `results_allmodels/`, `code/label_prior_diagnostic.py`, `code/bootstrap_biabav_ci.py` |
| GENBA checkpoint trajectories | `results_genba_checkpoints/`, `code/plot_genba_checkpoint_trajectories.py` |
| Interactive error analysis dashboard | `dashboard/` |

## Quick Start for Reviewers

Inspect the benchmark:

```bash
ls dataset
python -m json.tool dataset/arts_and_identity.json | head
```

Run the dashboard:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

In the sidebar, select one or more of:

- `results_allmodels`
- `results_closedmodels`
- `results_genba_checkpoints`

The result folders are large because they contain item-level outputs rather than
only aggregate scores.

## Dataset Summary

The evaluated data contains:

| Category | Source questions | Instances |
| --- | ---: | ---: |
| Arts and Identity | 25 | 75 |
| Building and Sacred Heritage | 29 | 87 |
| Culinary | 25 | 75 |
| Historical | 25 | 75 |
| Landscape | 25 | 75 |
| Language | 25 | 75 |
| Living Traditions and Customs | 27 | 81 |
| Politics | 25 | 75 |
| **Total** | **206** | **618** |

Item IDs follow:

```text
<category>_<language>_<number>
```

For example, `arts_and_identity_en_001`.

Languages are:

- `en`: English
- `de`: German
- `bar`: Bavarian

Question numbers `001` to `010` are the general-knowledge subset. Later numbers
are the grounded regional subset.

## Evaluation Protocols

The main open-weight and checkpoint runs include 12 protocols:

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

The closed-model reference includes:

- `letter`
- `generate_letter_and_text`

## Anonymous-Review Checklist

The following repository-specific anonymization steps have been applied.

1. Local cluster paths in shell scripts were replaced with placeholders.
   The remaining `code/job_*.sh` files use template values such as
   `/path/to/checkpoints`, `/path/to/container.sqsh`, and
   `<slurm-partition>`.

2. Result metadata paths were replaced with stable identifiers.
   Open-weight outputs now use `model://...`; checkpoint outputs now use
   `checkpoint://...`.

3. Run timestamps in result metadata were replaced with anonymous run labels
   such as `run_open_models_1`. Closed-model result filenames no longer include
   timestamps.

4. Check source citations in `dataset/*`.
   Grounded questions include public source references in `item.meta.source`,
   including news URLs and book references. These are part of benchmark
   transparency, but make sure none of the sources reveal annotator identity,
   private documents, or unpublished materials.

5. Remove private/generated caches.
   Do not include `.git/`, `__pycache__/`, local virtual environments, Streamlit
   caches, Slurm `.out` logs, or downloaded model weights in the anonymous
   package.

6. Review model identifiers.
   Public model names are necessary for reproducibility. If any internal model
   alias could identify the authors or institution, replace it with the
   anonymized name used in the paper.

7. Verify semantic runner availability.
   The semantic job scripts currently refer to
   `run_checkpoints_eval_v4_genba_v2_semantic_embed.py`. Include that runner,
   regenerate the scripts to call the available runner, or mark semantic
   outputs as precomputed-only.

8. Do a final secret scan.
   Run a final scan for emails, usernames, tokens, absolute paths, and
   institution-specific paths before packaging:

```bash
rg -n "(private_path_placeholder|@|api[_-]?key|secret|password|SBATCH|partition=)" .
```

## Notes

The artifact is intended for inspection and reproducibility of reported
analyses. Re-running all model evaluations requires access to the evaluated
models and to the GENBA checkpoint sequence, which are not included as weights
in this repository.
