# Anonymous Artifact Checklist

This file records the repository-specific anonymization work done before
submitting the BavGround artifact with an anonymous paper.

## Completed

1. **Cluster paths in job scripts**

   Files:

   - `code/job_checkpoints_eval_v4_genba_v2.sh`
   - `code/job_checkpoints_eval_v4_genba_checkpoints_v2.sh`
   - `code/job_checkpoints_eval_v4_genba_v2_embed_v3.sh`
   - `code/job_checkpoints_eval_v4_genba_v2_checkpoint_embed_v3.sh`
   - `code/job_checkpoints_eval_v4_genba_v2_semantic_embed.sh`
   - `code/job_checkpoints_eval_v4_genba_v2_semantic_embed_checkpoints.sh`

   Local Slurm/enroot paths and partition names were replaced with neutral
   placeholders.

2. **Result metadata paths**

   `results/allmodels/*.json` now stores `hf_dir` and `source` as
   `model://<target>`. `results/genba_checkpoints/*.json` now stores them as
   `checkpoint://<checkpoint>` identifiers.

3. **Run timestamps**

   Result metadata now uses labels such as `run_open_models_1`,
   `run_genba_checkpoints_1`, and `run_closed_model_1`. Closed-model result
   filenames no longer include timestamps.

4. **Packaging ignores**

   `.gitignore` excludes generated caches, virtual environments, Slurm logs,
   local model/checkpoint exports, model weight files, and container images.

## Still Review

5. **Closed-model API metadata**

   `results/closedmodels/*.json` stores API token usage totals. No API keys were
   found, but decide whether token counts should remain.

6. **Source metadata**

   Grounded dataset items include public references under `item.meta.source`.
   Keep them for transparency unless any source points to unpublished, private,
   or identifying material.

7. **Model aliases**

   Public model names are needed for reproducibility. Check any internal model
   aliases, especially `genba-10b-it` and checkpoint naming, against the
   anonymized terminology used in the submission.

## Packaging Hygiene

8. **Generated caches**

   Exclude:

   - `.git/`
   - `code/__pycache__/`
   - `.streamlit/`
   - virtual environments
   - Slurm `.out` files
   - downloaded model weights

9. **README consistency**

   Ensure the artifact title, model names, and release statements match the
   anonymous paper. Avoid author names, affiliations, acknowledgments, and
   repository URLs that identify the authors.

10. **Final scan command**

   Run:

   ```bash
   rg -n "(private_path_placeholder|@|api[_-]?key|secret|password|SBATCH|partition=)" .
   ```

   Expected non-sensitive matches after sanitization should be limited to
   benchmark content, public source URLs, and documentation examples.

## Current Scan Notes

- No email addresses were found in the scanned files.
- No obvious API keys or secrets were found. Many matches for `token` are
  expected because result files store token counts, token IDs, and token-level
  scores.
- The largest anonymity issue is local path metadata in result JSON files and
  Slurm/container paths in job scripts.
