# Bias Evaluation Dashboard

This repository contains evaluation outputs and a Streamlit dashboard for
exploring model/checkpoint behavior across cultural, political, language, and
heritage-style benchmark datasets.

The dashboard reads existing JSON result files from folders such as
`out_ckpts`, `out_ckpts_it`, `out_ckpts_genba_diag`, and `logs`. It does not
run model evaluations itself; it helps inspect and compare completed runs.

## What You Can Do

- Load one or more result folders or individual JSON files.
- Upload JSON result files directly in the Streamlit UI.
- Compare accuracy by target, dataset, language, and checkpoint step.
- Compare repeated run timestamps and evaluation strategies, such as `letter`,
  `option_text`, `letter_shuffled`, and `generate`.
- Inspect checkpoint trends in numeric order, for example `checkpoint0`,
  `checkpoint500`, `checkpoint1000`, and so on.
- Explore individual questions, chosen answers, gold labels, correctness, and
  stored label scores.

## Project Layout

```text
dashboard/
  app.py              Streamlit application
  data.py             JSON loading and normalization helpers
  requirements.txt    Dashboard Python dependencies
  README.md           Dashboard-specific notes
logs/                 Existing evaluation JSON outputs
out_ckpts/            Base/checkpoint evaluation outputs
out_ckpts_it/         Instruction-tuned evaluation outputs
out_ckpts_genba_diag/ Strategy evaluation outputs
```

## Setup

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip
```

Install the dashboard requirements:

```bash
pip install -r dashboard/requirements.txt
```

## Run the Dashboard

Start Streamlit from the repository root:

```bash
streamlit run dashboard/app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Using Data Sources

In the dashboard sidebar:

- Use **Quick folders** to select `out_ckpts`, `out_ckpts_it`,
  `out_ckpts_genba_diag`, or `logs`.
- Use **Additional folders or JSON files** for comma-separated paths.
- Use **Upload JSON result files** to inspect local JSON outputs without moving
  them into the repository.
- Use **Run timestamp(s)** and **Strategy(s)** to isolate or compare repeated
  runs and evaluation strategies.

The default quick folders are `out_ckpts` and `out_ckpts_it`.

## Notes

- JSON files should follow the output shape produced by the evaluation scripts,
  with top-level metadata and a `results` list.
- Older checkpoint logs are supported. If a file does not include
  `target_name`, the dashboard falls back to the checkpoint name or filename.
- `numpy<2` is pinned in `dashboard/requirements.txt` to avoid compatibility
  issues with older scientific Python packages commonly present in shared
  environments.

## Evaluation Strategies

The strategy evaluations are intended as first-class experimental conditions
for the paper, not merely as debugging checks. Multiple-choice evaluation can
itself introduce artifacts: a model may prefer a particular answer letter, may
be sensitive to option order, may tokenize option texts unevenly, or may behave
differently when asked to generate rather than score candidates. Reporting
multiple strategies helps distinguish content knowledge and cultural/political
bias from answer-format effects.

### `letter`

This is the standard multiple-choice scoring setup. The prompt shows options
`A`, `B`, `C`, and `D`, then the evaluator scores the log probability of each
answer letter directly and selects the highest-scoring letter.

This strategy is simple, stable, and easy to compare across models. It is also
closest to many standard MCQ benchmarks. Its main pitfall is letter bias: some
models may have a prior preference for a letter such as `A` or `C`, independent
of the option content. Letter tokenization can also differ across tokenizers,
especially across multilingual models.

### `option_text`

This strategy scores the full option text instead of only the answer letter. For
example, instead of scoring `A`, `B`, `C`, and `D`, it scores candidate strings
such as `Richard Wagner` or `A Bavarian cheese spread made with Camembert`.

This helps test whether poor MCQ performance is caused by answer-letter bias or
by actual lack of knowledge. If a model performs poorly with `letter` but much
better with `option_text`, the model may recognize the correct content but be
misled by the letter-only answer format. This is especially useful for studying
cultural and multilingual settings, where the semantic content of the option is
the real object of interest.

The main caveat is length and tokenization. Longer option texts naturally
accumulate more log probability terms, so total log probability can penalize
longer answers. This means `option_text` is informative but should be interpreted
alongside the other strategies.

### `option_text_avg`

This strategy is supported by the evaluation script and scores full option texts
by average log probability per token rather than total log probability.

It is designed to reduce length effects from `option_text`. If two answer
options differ greatly in token count, average token score can be a fairer
comparison than total score. The tradeoff is that averaging may overvalue short,
generic, or high-frequency phrases and undervalue longer but more specific
answers. When available, it is best reported next to `option_text` rather than
used as a replacement.

### `letter_shuffled`

This strategy deterministically shuffles the option positions for each question,
scores the displayed letters, and then maps the selected displayed letter back
to the original label. The shuffle is deterministic from the item id, so results
are reproducible.

This directly probes option-position and letter bias. If a model often chooses
the same displayed letter regardless of the underlying option, performance will
change under `letter_shuffled`. If performance remains stable, the model is more
likely using option content rather than a fixed letter prior.

This strategy is useful for paper reporting because it helps separate a model's
knowledge of the answer from artifacts of the benchmark format. It is not a
replacement for standard MCQ scoring; it is a complementary condition that shows
whether conclusions are robust to option order.

### `generate`

This strategy asks the model to generate a short answer and then parses the
predicted answer letter from the text. The prompt asks the model to start with
`A`, `B`, `C`, or `D`, optionally followed by the option text.

Generation is closer to how instruction-tuned models are often used in practice.
It can reveal whether a model knows the answer when it is not constrained to a
log-probability scoring interface. It can also expose instruction-following
failures, such as producing explanations, refusing to answer, or giving text
that cannot be parsed.

The main pitfall is parsing. A generated response may mention several letters or
include extra text, so the evaluator uses conservative parsing rules: it first
looks for a clear answer letter, then tries matching generated text to option
texts. Unparseable generations are treated cautiously. This means `generate`
measures both knowledge and answer-format compliance.

### Why Report Multiple Strategies

Using only one MCQ strategy can make model bias look stronger or weaker than it
really is. For example:

- A model with strong `A`/`B` letter preference may look biased on `letter` even
  when it recognizes the correct option text.
- A model that understands the content may improve on `option_text`, suggesting
  the issue is answer formatting rather than knowledge.
- A model whose performance drops under `letter_shuffled` may be relying on
  position or letter artifacts.
- A model that performs better under `generate` may need a natural instruction
  interface rather than forced candidate scoring.
- A model that performs worse under `generate` may know the answer but fail to
  follow concise answer instructions.

For the paper, these strategies should be read together. Agreement across
strategies strengthens a result. Disagreement across strategies is also
important: it identifies cases where apparent cultural, political, or language
bias may be entangled with evaluation format, tokenization, or
instruction-following behavior.
