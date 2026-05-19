# Evaluation Dashboard

This dashboard is a Streamlit app for exploring evaluation outputs from:

- `out_ckpts`
- `out_ckpts_it`
- `out_ckpts_genba_diag`
- any other folder or JSON file that follows the `run_checkpoints_eval_v4.py` output format

## Features

- Select result folders/files from friendly presets
- Upload one or many JSON result files from the browser
- Add extra folders or JSON files manually, such as `logs` or a single checkpoint file
- Filter by model/checkpoint target, dataset, language, and item type
- Filter by run timestamp and evaluation strategy
- Compare repeated runs and evaluation strategies as separate series
- Use regex to focus on checkpoint runs such as `checkpoint(\d+)`
- Compare target accuracy across datasets and languages
- Compare strategy accuracy across datasets
- Explore checkpoint trends over step number
- Inspect a single question across many models
- Inspect answer label scores, the chosen answer, and the runner-up score

## Run

From the repository root:

```bash
streamlit run dashboard/app.py
```

If you need dependencies:

```bash
pip install -r dashboard/requirements.txt
```

## Notes

- The app defaults to loading `out_ckpts,out_ckpts_it`
- The sidebar includes quick folder presets for `out_ckpts`, `out_ckpts_it`, `out_ckpts_genba_diag`, and `logs`
- Extra input paths are comma-separated in the sidebar
- Uploaded JSON files are combined with selected folders for the current session
- Strategy outputs with `diagnostic_config.mcq_strategy` are shown with strategies such as `letter`, `option_text`, `letter_shuffled`, and `generate`
- The app reads only existing JSON outputs; it does not run evaluations itself
- Older checkpoint logs such as `logs/answers_checkpoint500.json` are supported; if `target_name` is missing, the app uses the checkpoint name or filename
