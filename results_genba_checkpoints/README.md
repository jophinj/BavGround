# GENBA Checkpoint Results README

This folder contains item-level BavGround outputs for the GENBA-10B checkpoint
analysis in the paper.

## Contents

There are 8,160 JSON files:

```text
85 checkpoints x 8 categories x 12 evaluation protocols
```

The checkpoints range from `checkpoint0` through `checkpoint41707`.

## Filename Pattern

```text
answers_checkpoint<step>_<category>_<strategy>.json
```

Example:

```text
answers_checkpoint14500_language_option_text_avg.json
```

## Strategies

Each checkpoint-category pair is evaluated with:

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

## Paper Usage

This folder supports the longitudinal GENBA checkpoint analysis:

- protocol trajectories across continued pretraining
- category-level checkpoint trajectories
- language-level checkpoint trajectories
- checkpoint error analysis

Generate checkpoint figures with:

```bash
python code/plot_genba_checkpoint_trajectories.py \
  --results-dir results_genba_checkpoints \
  --out-dir images
```

## JSON Schema

Top-level fields include:

- `run_timestamp`
- `target_name`
- `target_kind`
- `checkpoint`
- `hf_dir`
- `source`
- `dataset`
- `filters`
- `generation_config`
- `diagnostic_config`
- `seconds`
- `results`

Each `results` row stores item-level predictions, correctness, and
strategy-specific diagnostics.

## Anonymous-Review Notes

The `hf_dir` and `source` fields currently preserve relative checkpoint paths
from the original evaluation environment. These should be replaced with
anonymous placeholders or public checkpoint identifiers before artifact
submission if the paths reveal private project structure.
