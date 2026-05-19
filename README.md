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

results/
  Item-level model outputs, grouped into allmodels, closedmodels, and
  genba_checkpoints.

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
| Open-weight model comparison | `results/allmodels/` |
| Closed-model reference | `results/closedmodels/` |
| Protocol sensitivity analysis | `results/allmodels/`, `code/label_prior_diagnostic.py`, `code/bootstrap_biabav_ci.py` |
| GENBA checkpoint trajectories | `results/genba_checkpoints/`, `code/plot_genba_checkpoint_trajectories.py` |
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

- `results/allmodels`
- `results/closedmodels`
- `results/genba_checkpoints`

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

## Notes

The artifact is intended for inspection and reproducibility of reported
analyses. Re-running all model evaluations requires access to the evaluated
models and to the GENBA checkpoint sequence, which are not included as weights
in this repository.
