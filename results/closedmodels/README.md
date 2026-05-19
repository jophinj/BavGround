# Closed-Model Results README

This folder contains the closed-model reference outputs reported in the paper.

## Contents

There are 16 JSON files:

```text
1 model x 8 categories x 2 protocols
```

The stored target name is `gpt-5.4-mini`.

## Filename Pattern

```text
answers_gpt-5.4-mini_<category>_<strategy>.json
```

Example:

```text
answers_gpt-5.4-mini_language_letter.json
```

## Strategies

The closed-model run includes:

- `letter`: first-token answer-letter scoring through API log probabilities
- `generate_letter_and_text`: generated answer parsing with letter plus
  optional option text

## JSON Schema

Top-level fields include:

- `run_timestamp`
- `target_name`
- `target_kind`
- `dataset`
- `filters`
- `generation_config`
- `diagnostic_config`
- `api_usage_total`
- `seconds`
- `results`

The `api_usage_total` field stores token counts for the run.

## Paper Usage

These files provide the closed-model reference point used to contextualize the
open-weight 7B to 10B results.
