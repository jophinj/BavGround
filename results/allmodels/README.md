# Open-Weight Results README

This folder contains item-level BavGround outputs for the 15 open-weight
instruction-tuned models reported in the paper.

## Contents

There are 1,440 JSON files:

```text
15 models x 8 categories x 12 evaluation protocols
```

Each file contains all language versions for one category, so a typical file
contains 75, 81, or 87 item-level rows depending on the category.

## Filename Pattern

```text
answers_<model>_<category>_<strategy>.json
```

Example:

```text
answers_Qwen2.5-7B-Instruct_language_option_text_avg.json
```

## Models

The folder contains outputs for the open-weight models listed in the paper,
including:

- `EuroLLM-9B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Llama-3.1-8B-Instruct`
- `granite-3.0-8b-instruct`
- `occiglot-7b-eu5-instruct`
- `aya-expanse-8b`
- `Mistral-7B-Instruct-v0.3`
- `OLMo-2-1124-7B-Instruct`
- `salamandra-7b-instruct`
- `gemma-2-9b-it`
- `leo-hessianai-7b-chat`
- `Teuken-7B-instruct-research-v0.4`
- `Pharia-1-LLM-7B-control-hf`
- `genba-10b-it`
- `LLaMmlein_7B_chat`

## Strategies

Each model-category pair is evaluated with:

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

Each `results` row includes the item ID, language, prompt, predicted answer,
gold answer, correctness, and strategy-specific diagnostics.

## Paper Usage

This folder supports:

- main open-weight model comparison
- language and category breakdowns
- GEN versus GRD comparisons
- protocol-sensitivity tables
- label-prior diagnostics

Useful scripts:

```bash
python code/bootstrap_biabav_ci.py --results-allmodels results/allmodels
python code/label_prior_diagnostic.py --results-allmodels results/allmodels
```

## Anonymous-Review Notes

The result files include anonymized metadata fields such as `hf_dir` and
`source`, using stable `model://...` identifiers rather than local paths.
